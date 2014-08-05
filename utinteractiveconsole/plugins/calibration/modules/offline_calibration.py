__author__ = 'jack'

import os
import logging
import numpy as np
from numpy.linalg import norm
from collections import namedtuple

log = logging.getLogger(__name__)

from atom.api import Event, Bool, Str, Value, Typed, Float, Int, observe
from enaml.qt import QtCore
from enaml.application import deferred_call

import enaml

with enaml.imports():
    from .views.offline_calibration import OfflineCalibrationPanel

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
    TooltipCalibrationProcessor, AbsoluteOrientationCalibrationProcessor, JointAngleCalibrationProcessor
)


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


class OfflineCalibrationController(CalibrationController):
    bgThread = Typed(BackgroundCalculationThread)
    is_working = Bool(False)

    # system configuration options
    # configuration parameters
    tt_minimal_angle_between_measurements = Float(0.1)

    ao_inital_maxdistance_from_origin = Float(0.1)
    ao_minimal_distance_between_measurements = Float(0.01)

    ja_minimal_distance_between_measurements = Float(0.01)
    ja_maximum_distance_to_reference = Float(0.02)
    ja_refinement_min_difference = Float(0.00001)
    ja_refinement_max_iterations = Int(10)

    refinement_shrink_factor = Float(0.8)

    joint_lengths = Value(np.array([0.13335, 0.13335]))
    origin_offset = Value(np.array([0.0, -0.11, -0.035]))

    # load all above values from the configuration file
    # try:
    # haptidevice_name = wiz_cfg.get("haptic_device").strip()
    #     hd_cfg = dict(gbl_cfg.items("ubitrack.devices.%s" % haptidevice_name))
    #     joint_lengths = np.array([float(hd_cfg["joint_length1"]), float(hd_cfg["joint_length2"]), ])
    #     origin_offset = np.array([float(hd_cfg["origin_offset_x"]),
    #                               float(hd_cfg["origin_offset_y"]),
    #                               float(hd_cfg["origin_offset_z"]),])
    # except Exception, e:
    #     log.error("Error reading Haptic device configuration. Make sure, the configuration file is correct.")
    #     log.exception(e)

    # results generated (iteratively)
    tooltip_calibration_result = Value(np.array([0, 0, 0]))
    absolute_orientation_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    jointangles_correction_result = Value(np.array([[0.0, 1.0, 0.0, ], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
    gimbalangles_correction_result = Value(np.array([[0.0, 1.0, 0.0, ], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))

    def setupController(self, active_widgets=None):
        active_widgets[0].find("btn_start_calibration").visible = False
        active_widgets[0].find("btn_stop_calibration").visible = False

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

        tt_processor.data_tracker_poses = [r.externaltracker_pose for r in selected_tt_data]
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

        ao_processor.data_tracker_hip_positions = [r.externaltracker_hip_position for r in selected_ao_data]
        ao_processor.data_fwk_hip_positions = [r.haptic_pose.translation() for r in selected_ao_data]
        ao_processor.facade = self.facade

        self.absolute_orientation_result = ao_processor.run()
        ao_processor.facade = None
        log.info("Result for Absolute Orientation: %s" % str(self.absolute_orientation_result))

    def do_jointangle_correction(self, ja_data):
        log.info("Joint-Angle Correction")
        ja_processor = JointAngleCalibrationProcessor()
        fwk = self.get_fwk(self.jointangles_correction_result, self.gimbalangles_correction_result)

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

        ja_processor.data_tracker_hip_positions = [r.hip_reference_pose.translation() for r in selected_ja_data]
        ja_processor.data_joint_angles = [r.jointangles for r in selected_ja_data]
        ja_processor.facade = self.facade

        self.jointangles_correction_result = ja_processor.run()
        ja_processor.facade = None
        log.info("Result for Joint-Angles Correction: %s" % str(self.jointangles_correction_result))

    def compute_position_errors(self, ja_data):
        fwk = self.get_fwk(self.jointangles_correction_result, self.gimbalangles_correction_result)
        position_errors = compute_position_errors(ja_data,
                                                  tooltip_offset=self.tooltip_calibration_result,
                                                  absolute_orientation=self.absolute_orientation_result,
                                                  forward_kinematics=fwk)
        log.info("Resulting position error: %s" % position_errors.mean())
        return position_errors

    def process(self):

        fname = os.path.join(self.dfg_dir, self.dfg_filename)
        if not os.path.isfile(fname):
            log.error("DFG file not found: %s" % fname)
            return

        self.facade.loadDataflow(fname)
        self.facade.startDataflow()

        data01 = self.load_data_step01()
        data03 = self.load_data_step03()
        data04 = self.load_data_step04()

        # 1st step: Tooltip Calibration (uses step01 data)
        self.do_tooltip_calibration(data01)

        # 2nd step: initial absolute orientation (uses step03  data)
        self.do_absolute_orientation(data03)

        # 3nd step: initial jointangle correction
        self.do_jointangle_correction(data04)

        # compute initial position errors
        last_error = self.compute_position_errors(data04)

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

            if (last_error.mean() - error.mean()) < self.ja_refinement_min_difference:
                break

            last_error = error
            iterations += 1

            if iterations > self.ja_refinement_max_iterations:
                log.warn("Terminating iterative optimization after %d cycles" % self.ja_refinement_max_iterations)
                break

        # continue with orientation calibration here

        # finally store or send the results somehow

        # display nice result graphs ??

        self.facade.stopDataflow()
        self.facade.clearDataflow()

    def load_data_step01(self):
        return loadData(
            os.path.expanduser(self.config.get("data_step01")),
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

    def load_data_step02(self):
        return loadData(
            os.path.expanduser(self.config.get("data_step02")),
            DSC('externaltracker_pose', 'externaltracker_hiptarget_pose.log',
                util.PoseStreamReader),
            items=(DSC('jointangles', 'phantom_joint_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
                   DSC('gimbalangles', 'phantom_gimbal_angles.log',
                       util.PositionStreamReader, interpolateVec3List),
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