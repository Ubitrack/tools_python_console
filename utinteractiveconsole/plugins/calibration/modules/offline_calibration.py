__author__ = 'jack'

import os
import time
import logging
from math import degrees, acos
import numpy as np
from numpy.linalg import norm
from collections import namedtuple

log = logging.getLogger(__name__)

from atom.api import Atom, Event, Bool, Str, Value, Typed, List, Float, Int, observe
from enaml.qt import QtCore
from enaml.application import deferred_call
from enaml.layout.api import InsertItem, FloatItem

import enaml

with enaml.imports():
    from .views.offline_calibration import OfflineCalibrationPanel, OfflineCalibrationResultPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

from ubitrack.core import util, measurement, math
from utinteractiveconsole.playback import (loadData, DSC, interpolatePoseList,
                                           interpolateVec3List, selectOnlyMatchingSamples,
                                           selectNearestNeighbour)

from utinteractiveconsole.plugins.calibration.algorithms.phantom_forward_kinematics import FWKinematicPhantom

from utinteractiveconsole.plugins.calibration.algorithms.streamfilters import (
    RelativeOrienationDistanceStreamFilter, StaticPointDistanceStreamFilter,
    RelativePointDistanceStreamFilter, TwoPointDistanceStreamFilter
)

from utinteractiveconsole.plugins.calibration.algorithms.offline_calibration import (
    TooltipCalibrationProcessor, AbsoluteOrientationCalibrationProcessor,
    JointAngleCalibrationProcessor, ReferenceOrientationProcessor,
    GimbalAngleCalibrationProcessor
)


angle_null_correction = np.array([[0.0, 1.0, 0.0, ], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

class BackgroundCalculationThread(QtCore.QThread):
    def __init__(self, ctrl):
        log.info("Init Background Calculation")
        super(BackgroundCalculationThread, self).__init__()
        self.ctrl = ctrl

    def run(self):
        log.info("BackgroundCalculationThread.run()")
        deferred_call(self.set_is_working, True)
        try:
            self.ctrl.process()
        except Exception, e:
            log.error("Error in BackgroundCalculationThread:")
            log.exception(e)
        finally:
            deferred_call(self.set_is_working, False)

    def set_is_working(self, v):
        self.ctrl.is_working = v


class CalibrationResults(Atom):
    boxplot_position_errors = Value()
    boxplot_orientation_errors = Value()


def compute_position_errors(stream,
                            tooltip_offset=None,
                            absolute_orientation=None,
                            forward_kinematics=None):

    if tooltip_offset is None:
        raise ValueError("TooltipOffset not supplied")

    if absolute_orientation is None:
        raise ValueError("AbsoluteOrientation not supplied")

    if forward_kinematics is None:
        raise ValueError("ForwardKinematics not supplied")

    len_data = len(stream)
    absolute_orientation_inv = absolute_orientation.invert()
    position_errors = np.zeros((len_data,), dtype=np.double)
    for i, record in enumerate(stream):
        haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
        hip_reference_pose = (
        absolute_orientation_inv * record.externaltracker_pose * math.Pose(math.Quaternion(), tooltip_offset))
        position_errors[i] = norm(hip_reference_pose.translation() - haptic_pose.translation())

    return position_errors


def compute_orientation_errors(stream,
                               tooltip_offset=None,
                               absolute_orientation=None,
                               forward_kinematics=None,
                               zref_axis=None):

    if tooltip_offset is None:
        raise ValueError("TooltipOffset not supplied")

    if absolute_orientation is None:
        raise ValueError("AbsoluteOrientation not supplied")

    if forward_kinematics is None:
        raise ValueError("ForwardKinematics not supplied")

    if zref_axis is None:
        raise ValueError("ZRefAxis not supplied")

    zaxis = np.array([0, 0, 1])

    len_data = len(stream)
    absolute_orientation_inv = absolute_orientation.invert()
    orientation_errors = np.zeros((len_data,), dtype=np.double)

    for i, record in enumerate(stream):
        haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
        hip_reference_pose = (
        absolute_orientation_inv * record.externaltracker_pose * math.Pose(math.Quaternion(), tooltip_offset))

        z_fwk = math.Quaternion(haptic_pose.rotation()).transformVector(zaxis)
        z_ref = math.Quaternion(hip_reference_pose.rotation()).transformVector(zref_axis)

        # unit vector
        z_fwk /= norm(z_fwk)
        z_ref /= norm(z_ref)

        orientation_errors[i] = degrees(acos(z_ref.dot(z_fwk)))

    return orientation_errors



class OfflineCalibrationController(CalibrationController):
    bgThread = Typed(BackgroundCalculationThread)
    is_working = Bool(False)

    has_result = Bool(False)

    # system configuration options
    # configuration parameters
    tt_minimal_angle_between_measurements = Float(0.1)

    ao_inital_maxdistance_from_origin = Float(0.03)
    ao_minimal_distance_between_measurements = Float(0.01)

    ja_minimal_distance_between_measurements = Float(0.005)
    ja_maximum_distance_to_reference = Float(0.02)
    ja_refinement_min_difference = Float(0.00001)
    ja_refinement_max_iterations = Int(3)

    ro_minimal_angle_between_measurements = Float(0.1)

    ga_minimal_angle_between_measurements = Float(0.1)

    refinement_shrink_factor = Float(0.8)

    joint_lengths = Value(np.array([0.13335, 0.13335]))
    origin_offset = Value(np.array([0.0, -0.11, -0.035]))


    # intermediate results
    theta6_correction_result = Value(np.array([0, 1, 0]))
    zaxis_reference_result = Value(np.array([0, 0, 1]))
    zaxis_points_result = Value([])

    # results generated (iteratively)
    tooltip_calibration_result = Value(np.array([0, 0, 0]))
    absolute_orientation_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    jointangles_correction_result = Value(angle_null_correction.copy())
    gimbalangles_correction_result = Value(angle_null_correction.copy())

    source_tooltip_calibration_result = Value()
    source_absolute_orientation_result = Value()
    source_jointangles_correction_result = Value()
    source_gimbalangles_correction_result = Value()

    source_zaxis_points_result = Value()

    position_errors = List()
    orientation_errors = List()

    def setupController(self, active_widgets=None):
        active_widgets[0].find("btn_start_calibration").visible = False
        active_widgets[0].find("btn_stop_calibration").visible = False
        wiz_cfg = self.wizard_state.config
        gbl_cfg = self.context.get("config")

        # load all parameters from the configuration file
        try:
            haptidevice_name = wiz_cfg.get("haptic_device").strip()
            hd_cfg = dict(gbl_cfg.items("ubitrack.devices.%s" % haptidevice_name))
            self.joint_lengths = np.array([float(hd_cfg["joint_length1"]),
                                           float(hd_cfg["joint_length2"]), ])

            self.origin_offset = np.array([float(hd_cfg["origin_offset_x"]),
                                           float(hd_cfg["origin_offset_y"]),
                                           float(hd_cfg["origin_offset_z"]),])
        except Exception, e:
            log.error("Error reading Haptic device configuration. Make sure, the configuration file is correct.")
            log.exception(e)

        parameters_sname = "%s.modules.%s.parameters" % (self.config_ns, self.module_name)
        if gbl_cfg.has_section(parameters_sname):
            self.tt_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "tt_minimal_angle_between_measurements")
            self.ao_inital_maxdistance_from_origin = gbl_cfg.getfloat(parameters_sname, "ao_inital_maxdistance_from_origin")
            self.ao_minimal_distance_between_measurements = gbl_cfg.getfloat(parameters_sname, "ao_minimal_distance_between_measurements")
            self.ja_minimal_distance_between_measurements = gbl_cfg.getfloat(parameters_sname, "ja_minimal_distance_between_measurements")
            self.ja_maximum_distance_to_reference = gbl_cfg.getfloat(parameters_sname, "ja_maximum_distance_to_reference")
            self.ja_refinement_min_difference = gbl_cfg.getfloat(parameters_sname, "ja_refinement_min_difference")
            self.ja_refinement_max_iterations = gbl_cfg.getint(parameters_sname, "ja_refinement_max_iterations")
            self.ro_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "ro_minimal_angle_between_measurements")
            self.ga_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "ga_minimal_angle_between_measurements")
            self.refinement_shrink_factor = gbl_cfg.getfloat(parameters_sname, "refinement_shrink_factor")
        else:
            log.warn("No parameters found for offline calibration - using defaults. Define parameters in section: %s" % parameters_sname)



    def do_offline_calibration(self):
        self.bgThread = BackgroundCalculationThread(self)
        self.bgThread.start()

    def do_tooltip_calibration(self, tt_data):
        log.info("Tooltip Calibration")
        tt_processor = TooltipCalibrationProcessor()

        tt_selector = RelativeOrienationDistanceStreamFilter("externaltracker_pose",
                                                             min_distance=self.tt_minimal_angle_between_measurements)
        selected_tt_data = tt_selector.process(tt_data)
        log.info("Offline Tooltip Calibration (%d out of %d records selected)" % (len(selected_tt_data), len(tt_data)))

        tt_processor.data = selected_tt_data
        tt_processor.facade = self.facade

        self.tooltip_calibration_result = tt_processor.run()

        tt_processor.facade = None
        log.info("Result for Tooltip Calibration: %s" % str(self.tooltip_calibration_result))

    def do_absolute_orientation(self, ao_data):
        log.info("Absolute Orientation")
        ao_processor = AbsoluteOrientationCalibrationProcessor()
        fwk = self.get_fwk(self.jointangles_correction_result, self.gimbalangles_correction_result)

        ao_data_ext = ao_processor.prepare_stream(ao_data,
                                                  tooltip_offset=self.tooltip_calibration_result,
                                                  absolute_orientation=self.absolute_orientation_result,
                                                  forward_kinematics=fwk)

        ao_selector1 = StaticPointDistanceStreamFilter("haptic_pose", np.array([0, 0, 0]),
                                                       max_distance=self.ao_inital_maxdistance_from_origin)

        ao_selector2 = RelativePointDistanceStreamFilter("haptic_pose",
                                                         min_distance=self.ao_minimal_distance_between_measurements)

        selected_ao_data = ao_selector2.process(ao_selector1.process(ao_data_ext))
        log.info(
            "Absolute Orientation Calibration (%d out of %d records selected)" % (len(selected_ao_data), len(ao_data)))

        ao_processor.data = selected_ao_data
        ao_processor.facade = self.facade

        self.absolute_orientation_result = ao_processor.run()
        ao_processor.facade = None
        log.info("Result for Absolute Orientation: %s" % str(self.absolute_orientation_result))

    def do_jointangle_correction(self, ja_data):
        log.info("Joint-Angle Correction")
        ja_processor = JointAngleCalibrationProcessor()
        fwk = self.get_fwk(angle_null_correction, angle_null_correction)

        ja_data_ext = ja_processor.prepare_stream(ja_data,
                                                  tooltip_offset=self.tooltip_calibration_result,
                                                  absolute_orientation=self.absolute_orientation_result,
                                                  forward_kinematics=fwk)

        # simple way to avoid outliers from the external tracker: limit distance to reference ...
        ja_selector1 = TwoPointDistanceStreamFilter("hip_reference_pose", "haptic_pose",
                                                    max_distance=self.ja_maximum_distance_to_reference)

        # only use a subset of the dataset
        ja_selector2 = RelativePointDistanceStreamFilter("haptic_pose",
                                                         min_distance=self.ja_minimal_distance_between_measurements)

        selected_ja_data = ja_selector2.process(ja_selector1.process(ja_data_ext))
        log.info("Joint-Angles Calibration (%d out of %d records selected)" % (len(selected_ja_data), len(ja_data)))

        ja_processor.data = selected_ja_data
        ja_processor.facade = self.facade

        self.jointangles_correction_result = ja_processor.run()
        ja_processor.facade = None
        log.info("Result for Joint-Angles Correction: %s" % str(self.jointangles_correction_result))

    def do_gimbalangle_correction(self, ga_data):
        log.info("Gimbal-Angle Correction")
        ga_processor = GimbalAngleCalibrationProcessor()
        fwk = self.get_fwk(self.jointangles_correction_result, angle_null_correction)

        ga_data_ext = ga_processor.prepare_stream(ga_data,
                                                  tooltip_offset=self.tooltip_calibration_result,
                                                  absolute_orientation=self.absolute_orientation_result,
                                                  forward_kinematics=fwk,
                                                  zrefaxis_calib=self.zaxis_reference_result)

        ga_selector = RelativeOrienationDistanceStreamFilter("haptic_pose",
                                                             min_distance=self.ga_minimal_angle_between_measurements)

        selected_ga_data = ga_selector.process(ga_data_ext)
        log.info("Gimbal-Angles Calibration (%d out of %d records selected)" % (len(selected_ga_data), len(ga_data)))

        ga_processor.data_joint_angle_correction = self.jointangles_correction_result
        ga_processor.data = selected_ga_data
        ga_processor.facade = self.facade

        gimbalangle_correction = ga_processor.run()

        # add theta6 correction here
        gimbalangle_correction[2, 0] = self.theta6_correction_result[0]
        gimbalangle_correction[2, 1] = self.theta6_correction_result[1]
        gimbalangle_correction[2, 2] = self.theta6_correction_result[2]

        self.gimbalangles_correction_result = gimbalangle_correction
        ga_processor.facade = None
        log.info("Result for Gimbal-Angles Correction: %s" % str(self.gimbalangles_correction_result))

    def do_reference_orientation(self, ro_data):
        log.info("Calculate Reference Orientation")
        ro_processor = ReferenceOrientationProcessor()
        fwk = self.get_fwk(self.jointangles_correction_result, self.gimbalangles_correction_result)
        fwk_5dof = self.get_fwk(self.jointangles_correction_result, self.gimbalangles_correction_result, disable_theta6=True)

        ro_data_ext = ro_processor.prepare_stream(ro_data,
                                                  tooltip_offset=self.tooltip_calibration_result,
                                                  absolute_orientation=self.absolute_orientation_result,
                                                  forward_kinematics=fwk,
                                                  forward_kinematics_5dof=fwk_5dof,
                                                  use_markers=True)
        # no filtering for now
        selected_ro_data = ro_data_ext

        log.info("Reference Orientation (%d out of %d records selected)" % (len(selected_ro_data), len(ro_data)))

        ro_processor.data = selected_ro_data
        ro_processor.facade = self.facade

        self.zaxis_reference_result, self.zaxis_points_result, theta6_correction = ro_processor.run(use_markers=True)
        if theta6_correction is not None:
            self.theta6_correction_result = theta6_correction

        ro_processor.facade = None
        log.info("Result for ReferenceOrientation: %s" % str(self.zaxis_reference_result))
        log.info("Result for Theta6-Correction: %s" % str(self.theta6_correction_result))

    def compute_position_errors(self, ja_data):
        fwk = self.get_fwk(self.jointangles_correction_result, self.gimbalangles_correction_result)
        position_errors = compute_position_errors(ja_data,
                                                  tooltip_offset=self.tooltip_calibration_result,
                                                  absolute_orientation=self.absolute_orientation_result,
                                                  forward_kinematics=fwk)
        log.info("Resulting position error: %s" % position_errors.mean())
        return position_errors

    def compute_orientation_errors(self, ga_data):
        fwk = self.get_fwk(self.jointangles_correction_result, self.gimbalangles_correction_result)
        orientation_errors = compute_orientation_errors(ga_data,
                                                  tooltip_offset=self.tooltip_calibration_result,
                                                  absolute_orientation=self.absolute_orientation_result,
                                                  forward_kinematics=fwk,
                                                  zref_axis=self.zaxis_reference_result)
        log.info("Resulting orientation error: %s" % orientation_errors.mean())
        return orientation_errors



    def process(self):
        fname = os.path.join(self.dfg_dir, self.dfg_filename)
        if not os.path.isfile(fname):
            log.error("DFG file not found: %s" % fname)
            return

        self.facade.loadDataflow(fname)
        self.facade.startDataflow()

        # connect result sources
        self.source_tooltip_calibration_result = self.facade.instance.getApplicationPushSourcePosition("result_calib_tooltip")
        self.source_absolute_orientation_result = self.facade.instance.getApplicationPushSourcePose("result_calib_absolute_orientation")
        self.source_jointangles_correction_result = self.facade.instance.getApplicationPushSourceMatrix3x3("result_calib_phantom_jointangle_correction")
        self.source_gimbalangles_correction_result = self.facade.instance.getApplicationPushSourceMatrix3x3("result_calib_phantom_gimbalangle_correction")

        self.source_zaxis_points_result = self.facade.instance.getApplicationPushSourcePositionList("result_calib_zrefaxis_points")


        log.info("Loading recorded streams for Offline Calibration")
        data01 = self.load_data_step01()
        data02 = self.load_data_step02()
        data03 = self.load_data_step03()
        data04 = self.load_data_step04()


        # 1st step: Tooltip Calibration (uses step01 data)
        self.do_tooltip_calibration(data01)

        # 2nd step: initial absolute orientation (uses step03  data)
        self.do_absolute_orientation(data03)

        # compute initial errors
        self.position_errors.append(self.compute_position_errors(data04))

        # 3nd step: initial jointangle correction
        self.do_jointangle_correction(data04)

        # compute initial position errors
        last_error = self.compute_position_errors(data04)
        self.position_errors.append(last_error)

        iterations = 0
        while True:
            # modify the frame selector parameters
            self.ao_inital_maxdistance_from_origin /= self.refinement_shrink_factor
            self.ao_minimal_distance_between_measurements *= self.refinement_shrink_factor
            self.ja_minimal_distance_between_measurements *= self.refinement_shrink_factor

            # redo the calibration
            self.do_absolute_orientation(data03)
            self.do_jointangle_correction(data04)

            # recalculate the error
            error = self.compute_position_errors(data04)
            self.position_errors.append(error)

            if (last_error.mean() - error.mean()) < self.ja_refinement_min_difference:
                break

            last_error = error
            iterations += 1

            if iterations > self.ja_refinement_max_iterations:
                log.warn("Terminating iterative optimization after %d cycles" % self.ja_refinement_max_iterations)
                break

        # 4th step: reference orientation
        self.do_reference_orientation(data02)

        self.orientation_errors.append(self.compute_orientation_errors(data01))

        # 5th step: gimbalangle correction
        self.do_gimbalangle_correction(data01)

        # compute errors after calibration
        self.orientation_errors.append(self.compute_orientation_errors(data01))

        # do iterative refinement for orientation as well ?

        # finally store or send the results
        ts = measurement.now()
        self.source_tooltip_calibration_result.send(measurement.Position(ts, self.tooltip_calibration_result))
        self.source_absolute_orientation_result.send(measurement.Pose(ts, self.absolute_orientation_result))
        self.source_jointangles_correction_result.send(measurement.Matrix3x3(ts, self.jointangles_correction_result))
        self.source_gimbalangles_correction_result.send(measurement.Matrix3x3(ts, self.gimbalangles_correction_result))

        if len(self.zaxis_points_result) > 0:
            self.source_zaxis_points_result.send(measurement.PositionList(ts, math.PositionList.fromList(self.zaxis_points_result)))


        # wait a bit before shutting down to allow ubitrack to process the data
        time.sleep(0.1)
        # display nice result graphs or at least text??
        self.has_result = True

        # teardown

        self.source_tooltip_calibration_result = None
        self.source_absolute_orientation_result = None
        self.source_jointangles_correction_result = None
        self.source_gimbalangles_correction_result = None

        self.facade.stopDataflow()
        self.facade.clearDataflow()



    def do_visualize_results(self):
        # create figures
        from matplotlib.figure import Figure

        poserr = Figure()
        ax1 = poserr.add_subplot(111)
        ax1.boxplot(self.position_errors)
        ax1.set_title("Position Errors")

        ornerr = Figure()
        ax2 = ornerr.add_subplot(111)
        ax2.boxplot(self.orientation_errors)
        ax2.set_title("Orientation Errors")

        results = CalibrationResults(boxplot_position_errors=poserr,
                                     boxplot_orientation_errors=ornerr)
        # create results panel
        panel = OfflineCalibrationResultPanel(name="utic.ismar14.offline_calibration_result.%s" % time.time(),
                                              results=results)


        # add to layout
        wizard_controller = self.wizard_state.controller
        parent = wizard_controller.wizview.parent

        panel.set_parent(parent)
        op = FloatItem(item=panel.name,)
        parent.update_layout(op)


    def load_data_step01(self):
        return loadData(
            os.path.expanduser(self.config.get("data_step01")),
            DSC('externaltracker_pose', 'externaltracker_hiptarget_pose.log',
                util.PoseStreamReader),
            items=(DSC('jointangles', 'phantom_joint_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
                   DSC('gimbalangles', 'phantom_gimbal_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
            )
        )

    def load_data_step02(self):
        return loadData(
            os.path.expanduser(self.config.get("data_step02")),
            DSC('externaltracker_pose', 'externaltracker_hiptarget_pose.log',
                util.PoseStreamReader),
            items=(DSC('jointangles', 'phantom_joint_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
                   DSC('gimbalangles', 'phantom_gimbal_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
                   DSC('externaltracker_markers', 'externaltracker_hiptarget_markers.log',
                       util.PositionListStreamReader, selectOnlyMatchingSamples),
            )
        )

    def load_data_step03(self):
        return loadData(
            os.path.expanduser(self.config.get("data_step03")),
            DSC('externaltracker_pose', 'externaltracker_hiptarget_pose.log',
                util.PoseStreamReader),
            items=(DSC('jointangles', 'phantom_joint_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
                   DSC('gimbalangles', 'phantom_gimbal_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
            )
        )

    def load_data_step04(self):
        return loadData(
            os.path.expanduser(self.config.get("data_step04")),
            DSC('externaltracker_pose', 'externaltracker_hiptarget_pose.log',
                util.PoseStreamReader),
            items=(DSC('jointangles', 'phantom_joint_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
                   DSC('gimbalangles', 'phantom_gimbal_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
            )
        )

    def get_fwk(self, jointangle_calib, gimbalangle_calib, disable_theta6=False):
        return FWKinematicPhantom(self.joint_lengths,
                                  jointangle_calib,
                                  gimbalangle_calib,
                                  self.origin_offset,
                                  disable_theta6=disable_theta6)


class OfflineCalibrationModule(ModuleBase):
    def get_category(self):
        return "Calibration"

    def get_widget_class(self):
        return OfflineCalibrationPanel

    def get_controller_class(self):
        return OfflineCalibrationController