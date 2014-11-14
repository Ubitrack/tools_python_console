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
    from .views.offline_calibration2 import OfflineCalibrationPanel, OfflineCalibrationResultPanel

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

from utinteractiveconsole.plugins.calibration.algorithms.offline_calibration2 import (
    TooltipCalibrationProcessor, AbsoluteOrientationCalibrationProcessor,
    JointAngleCalibrationProcessor, ReferenceOrientationProcessor,
    GimbalAngleCalibrationProcessor
)


available_interpolators = dict(interpolatePoseList=interpolatePoseList,
                               interpolateVec3List=interpolateVec3List,
                               selectOnlyMatchingSamples=selectOnlyMatchingSamples,
                               selectNearestNeighbour=selectNearestNeighbour,)


# Initialization of disabled steps
tooltip_null_calibration =  math.Pose(math.Quaternion(), np.array([0., 0., 0.]))
absolute_orientation_null_calibration = math.Pose(math.Quaternion(), np.array([0., 0., 0.]))
reference_orientation_null_calibration = np.array([0., 0., 1.])
angle_null_correction = np.array([[0.0, 1.0, 0.0, ], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])



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
        absolute_orientation_inv * record.externaltracker_pose * tooltip_offset)
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
        absolute_orientation_inv * record.externaltracker_pose * tooltip_offset)

        z_fwk = math.Quaternion(haptic_pose.rotation()).transformVector(zaxis)
        z_ref = math.Quaternion(hip_reference_pose.rotation()).transformVector(zref_axis)

        # unit vector
        z_fwk /= norm(z_fwk)
        z_ref /= norm(z_ref)

        orientation_errors[i] = degrees(acos(z_ref.dot(z_fwk)))

    return orientation_errors



class BackgroundCalculationThread(QtCore.QThread):
    def __init__(self, processor):
        log.info("Init Background Calculation")
        super(BackgroundCalculationThread, self).__init__()
        self.processor = processor

    def run(self):
        log.info("BackgroundCalculationThread.run()")
        deferred_call(self.set_is_working, True)
        try:
            self.processor.process()
        except Exception, e:
            log.error("Error in BackgroundCalculationThread:")
            log.exception(e)
        finally:
            deferred_call(self.set_is_working, False)

    def set_is_working(self, v):
        self.processor.is_working = v




class OfflineCalibrationResults(Atom):

    has_result = Bool(False)

    # calibration results
    theta6_correction_result = Value(np.array([0, 1, 0]))
    zaxis_reference_result = Value(np.array([0, 0, 1]))
    zaxis_points_result = Value([])

    tooltip_calibration_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    absolute_orientation_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    jointangles_correction_result = Value(np.array(angle_null_correction))
    gimbalangles_correction_result = Value(np.array(angle_null_correction))

    # results evaluation
    position_errors = List()
    orientation_errors = List()

    def reset(self):
        self.has_result = False

        self.theta6_correction_result = np.array([0, 1, 0])
        self.zaxis_reference_result = np.array([0, 0, 1])
        self.zaxis_points_result = []

        self.tooltip_calibration_result = math.Pose(math.Quaternion(), np.array([0, 0, 0]))
        self.absolute_orientation_result = math.Pose(math.Quaternion(), np.array([0, 0, 0]))
        self.jointangles_correction_result = np.array(angle_null_correction)
        self.gimbalangles_correction_result = np.array(angle_null_correction)

        self.position_errors = []
        self.orientation_errors = []


class OfflineCalibrationParameters(Atom):
    # tooltip
    tooltip_enabled = Bool(False)
    tooltip_datasource = Str()
    tt_minimal_angle_between_measurements = Float(0.1)

    # absolute orientation
    absolute_orientation_enabled = Bool(False)
    absolute_orientation_datasource = Str()
    ao_inital_maxdistance_from_origin = Float(0.03)
    ao_minimal_distance_between_measurements = Float(0.01)
    ao_refinement_expand_coverage = Float(1.2)
    ao_refinement_shrink_distance = Float(0.8)

    # joint-angle correction
    joint_angle_calibration_enabled = Bool(False)
    joint_angle_calibration_datasource = Str()
    ja_minimal_distance_between_measurements = Float(0.005)
    ja_maximum_distance_to_reference = Float(0.02)
    ja_refinement_min_difference = Float(0.00001)
    ja_refinement_max_iterations = Int(3)
    ja_refinement_shrink_distance = Float(0.8)

    # reference orientation
    reference_orientation_enabled = Bool(False)
    reference_orientation_datasource = Str()
    ro_minimal_angle_between_measurements = Float(0.1)

    # gimbal-angle correction
    gimbal_angle_calibration_enabled = Bool(False)
    gimbal_angle_calibration_datasource = Str()
    ga_minimal_angle_between_measurements = Float(0.1)

    # haptic device
    joint_lengths = Value(np.array([0.13335, 0.13335]))
    origin_offset = Value(np.array([0.0, -0.11, -0.035]))


class OfflineCalibrationProcessor(Atom):

    context = Value()
    config = Value()
    facade = Value()
    dfg_dir = Value()
    dfg_filename = Value()

    is_working = Bool(False)

    parameters = Typed(OfflineCalibrationParameters)
    result = Typed(OfflineCalibrationResults)

    # refinement vars
    ao_maxdistance_from_origin = Float(0.0)
    ao_minimal_distance_between_measurements = Float(0.0)
    ja_minimal_distance_between_measurements = Float(0.0)


    source_tooltip_calibration_result = Value()
    source_absolute_orientation_result = Value()
    source_jointangles_correction_result = Value()
    source_gimbalangles_correction_result = Value()

    source_zaxis_points_result = Value()
    source_zaxis_reference_result = Value()

    def _default_result(self):
        return OfflineCalibrationResults()


    def do_tooltip_calibration(self, tt_data):
        log.info("Tooltip Calibration")
        tt_processor = TooltipCalibrationProcessor()

        tt_selector = RelativeOrienationDistanceStreamFilter("externaltracker_pose",
                                                             min_distance=self.parameters.tt_minimal_angle_between_measurements)
        selected_tt_data = tt_selector.process(tt_data)
        log.info("Offline Tooltip Calibration (%d out of %d records selected)" % (len(selected_tt_data), len(tt_data)))

        tt_processor.data = selected_tt_data
        tt_processor.facade = self.facade

        self.result.tooltip_calibration_result = tt_processor.run()

        tt_processor.facade = None
        log.info("Result for Tooltip Calibration: %s" % str(self.result.tooltip_calibration_result))

        return True

    def do_absolute_orientation(self, ao_data):
        log.info("Absolute Orientation")
        ao_processor = AbsoluteOrientationCalibrationProcessor()
        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result)

        ao_data_ext = ao_processor.prepare_stream(ao_data,
                                                  tooltip_offset=self.result.tooltip_calibration_result,
                                                  absolute_orientation=self.result.absolute_orientation_result,
                                                  forward_kinematics=fwk)

        ao_selector1 = StaticPointDistanceStreamFilter("haptic_pose", np.array([0, 0, 0]),
                                                       max_distance=self.ao_maxdistance_from_origin)

        ao_selector2 = RelativePointDistanceStreamFilter("haptic_pose",
                                                         min_distance=self.ao_minimal_distance_between_measurements)

        selected_ao_data = ao_selector2.process(ao_selector1.process(ao_data_ext))
        log.info(
            "Absolute Orientation Calibration (%d out of %d records selected)" % (len(selected_ao_data), len(ao_data)))

        if len(selected_ao_data) == 0:
            log.error("No Records selected for Absolute Orientation Calibration - please redo Step03 and provide valid data.")
            return False

        ao_processor.data = selected_ao_data
        ao_processor.facade = self.facade

        self.result.absolute_orientation_result = ao_processor.run()

        ao_processor.facade = None
        log.info("Result for Absolute Orientation: %s" % str(self.result.absolute_orientation_result))

        return True

    def do_jointangle_correction(self, ja_data):
        log.info("Joint-Angle Correction")
        ja_processor = JointAngleCalibrationProcessor()
        fwk = self.get_fwk(angle_null_correction, angle_null_correction)

        ja_data_ext = ja_processor.prepare_stream(ja_data,
                                                  tooltip_offset=self.result.tooltip_calibration_result,
                                                  absolute_orientation=self.result.absolute_orientation_result,
                                                  forward_kinematics=fwk)

        # simple way to avoid outliers from the external tracker: limit distance to reference ...
        ja_selector1 = TwoPointDistanceStreamFilter("hip_reference_pose", "haptic_pose",
                                                    max_distance=self.parameters.ja_maximum_distance_to_reference)

        # only use a subset of the dataset
        ja_selector2 = RelativePointDistanceStreamFilter("haptic_pose",
                                                         min_distance=self.ja_minimal_distance_between_measurements)

        selected_ja_data = ja_selector2.process(ja_selector1.process(ja_data_ext))
        log.info("Joint-Angles Calibration (%d out of %d records selected)" % (len(selected_ja_data), len(ja_data)))

        ja_processor.data = selected_ja_data
        ja_processor.facade = self.facade

        self.result.jointangles_correction_result = ja_processor.run()

        ja_processor.facade = None
        log.info("Result for Joint-Angles Correction: %s" % str(self.result.jointangles_correction_result))

        return True

    def do_gimbalangle_correction(self, ga_data):
        log.info("Gimbal-Angle Correction")
        ga_processor = GimbalAngleCalibrationProcessor()
        fwk = self.get_fwk(self.result.jointangles_correction_result, angle_null_correction)

        ga_data_ext = ga_processor.prepare_stream(ga_data,
                                                  tooltip_offset=self.result.tooltip_calibration_result,
                                                  absolute_orientation=self.result.absolute_orientation_result,
                                                  forward_kinematics=fwk,
                                                  zrefaxis_calib=self.result.zaxis_reference_result)

        ga_selector = RelativeOrienationDistanceStreamFilter("haptic_pose",
                                                             min_distance=self.parameters.ga_minimal_angle_between_measurements)

        selected_ga_data = ga_selector.process(ga_data_ext)
        log.info("Gimbal-Angles Calibration (%d out of %d records selected)" % (len(selected_ga_data), len(ga_data)))

        ga_processor.data_joint_angle_correction = self.result.jointangles_correction_result
        ga_processor.data = selected_ga_data
        ga_processor.facade = self.facade

        gimbalangle_correction = ga_processor.run()

        # add theta6 correction here
        gimbalangle_correction[2, 0] = self.result.theta6_correction_result[0]
        gimbalangle_correction[2, 1] = self.result.theta6_correction_result[1]
        gimbalangle_correction[2, 2] = self.result.theta6_correction_result[2]

        self.result.gimbalangles_correction_result = gimbalangle_correction

        ga_processor.facade = None
        log.info("Result for Gimbal-Angles Correction: %s" % str(self.result.gimbalangles_correction_result))

        return True

    def do_reference_orientation(self, ro_data):
        log.info("Calculate Reference Orientation")
        ro_processor = ReferenceOrientationProcessor()
        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result)
        fwk_5dof = self.get_fwk(self.result.jointangles_correction_result, angle_null_correction, disable_theta6=True)

        ro_data_ext = ro_processor.prepare_stream(ro_data,
                                                  tooltip_offset=self.result.tooltip_calibration_result,
                                                  absolute_orientation=self.result.absolute_orientation_result,
                                                  forward_kinematics=fwk,
                                                  forward_kinematics_5dof=fwk_5dof,
                                                  use_markers=True)
        # no filtering for now
        selected_ro_data = ro_data_ext

        log.info("Reference Orientation (%d out of %d records selected)" % (len(selected_ro_data), len(ro_data)))

        ro_processor.data = selected_ro_data
        ro_processor.facade = self.facade

        zaxis_reference, zaxis_points, theta6_correction = ro_processor.run(use_markers=True)
        self.result.zaxis_reference_result = zaxis_reference
        self.result.zaxis_points_result = zaxis_points

        if theta6_correction is not None:
            self.result.theta6_correction_result =  theta6_correction

        ro_processor.facade = None
        log.info("Result for ReferenceOrientation: %s" % str(self.result.zaxis_reference_result))
        log.info("Result for Theta6-Correction: %s" % str(self.result.theta6_correction_result))

        return True

    def compute_position_errors(self, ja_data):
        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result)
        position_errors = compute_position_errors(ja_data,
                                                  tooltip_offset=self.result.tooltip_calibration_result,
                                                  absolute_orientation=self.result.absolute_orientation_result,
                                                  forward_kinematics=fwk)
        log.info("Resulting position error: %s" % position_errors.mean())
        return position_errors

    def compute_orientation_errors(self, ga_data):
        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result)
        orientation_errors = compute_orientation_errors(ga_data,
                                                  tooltip_offset=self.result.tooltip_calibration_result,
                                                  absolute_orientation=self.result.absolute_orientation_result,
                                                  forward_kinematics=fwk,
                                                  zref_axis=self.result.zaxis_reference_result)
        log.info("Resulting orientation error: %s" % orientation_errors.mean())
        return orientation_errors

    def reset(self, parameters):
        self.result.reset()

        self.parameters = parameters
        # mutable parameters are copied
        self.ao_maxdistance_from_origin = self.parameters.ao_inital_maxdistance_from_origin
        self.ao_minimal_distance_between_measurements = self.parameters.ao_minimal_distance_between_measurements
        self.ja_minimal_distance_between_measurements = self.parameters.ja_minimal_distance_between_measurements


    def process(self):

        fname = os.path.join(self.dfg_dir, self.dfg_filename)
        if not os.path.isfile(fname):
            log.error("DFG file not found: %s" % fname)
            return


        self.facade.loadDataflow(fname)
        self.facade.startDataflow()

        # connect result sources
        if self.parameters.tooltip_enabled:
            self.source_tooltip_calibration_result = self.facade.instance.getApplicationPushSourcePose("result_calib_tooltip")

        if self.parameters.absolute_orientation_enabled:
            self.source_absolute_orientation_result = self.facade.instance.getApplicationPushSourcePose("result_calib_absolute_orientation")

        if self.parameters.joint_angle_calibration_enabled:
            self.source_jointangles_correction_result = self.facade.instance.getApplicationPushSourceMatrix3x3("result_calib_phantom_jointangle_correction")

        if self.parameters.gimbal_angle_calibration_enabled:
            self.source_gimbalangles_correction_result = self.facade.instance.getApplicationPushSourceMatrix3x3("result_calib_phantom_gimbalangle_correction")

        if self.parameters.reference_orientation_enabled:
            self.source_zaxis_points_result = self.facade.instance.getApplicationPushSourcePositionList("result_calib_zrefaxis_points")
            self.source_zaxis_reference_result = self.facade.instance.getApplicationPushSourcePosition("result_calib_zrefaxis_reference")



        log.info("Loading recorded streams for Offline Calibration")
        datasources = self.load_datasources()


        if self.parameters.tooltip_enabled:
            # 1st step: Tooltip Calibration (uses step01 data)
            self.do_tooltip_calibration(datasources.get(self.parameters.tooltip_datasource, None))
        else:
            # skipped tooltip calibration, defaults to no offset
            self.result.tooltip_calibration_result = tooltip_null_calibration



        # 2nd step: initial absolute orientation (uses step03  data)
        if self.parameters.absolute_orientation_enabled:
            if not self.do_absolute_orientation(datasources.get(self.parameters.absolute_orientation_datasource, None)):
                return
        else:
            log.warn("Absolute Orientation Calibration is disabled - Are you sure this is correct ????")
            self.result.absolute_orientation_result = absolute_orientation_null_calibration

        # compute initial errors
        self.result.position_errors.append(self.compute_position_errors(datasources.get(self.parameters.joint_angle_calibration_datasource, None)))

        last_error = np.array([0.,])

        # 3nd step: initial jointangle correction
        if self.parameters.joint_angle_calibration_enabled:
            self.do_jointangle_correction(datasources.get(self.parameters.joint_angle_calibration_datasource, None))

            # compute initial position errors
            last_error = self.compute_position_errors(datasources.get(self.parameters.joint_angle_calibration_datasource, None))
            self.result.position_errors.append(last_error)
        else:
            self.result.jointangles_correction_result = angle_null_correction

        # Iterative refinement makes only sense if absolute orientation and joint angle calibration are enabled
        iterative_refinement_enabled = self.parameters.absolute_orientation_enabled and self.parameters.joint_angle_calibration_enabled
        iterations = 0
        while iterative_refinement_enabled:
            # modify the frame selector parameters
            self.ao_maxdistance_from_origin *= self.parameters.ao_refinement_expand_coverage
            self.ao_minimal_distance_between_measurements *= self.parameters.ao_refinement_shrink_distance
            self.ja_minimal_distance_between_measurements *= self.parameters.ja_refinement_shrink_distance

            # redo the calibration
            if self.parameters.absolute_orientation_enabled:
                if not self.do_absolute_orientation(datasources.get(self.parameters.absolute_orientation_datasource, None)):
                    break

            if self.parameters.joint_angle_calibration_enabled:
                self.do_jointangle_correction(datasources.get(self.parameters.joint_angle_calibration_datasource, None))

                # recalculate the error
                error = self.compute_position_errors(datasources.get(self.parameters.joint_angle_calibration_datasource, None))
                self.result.position_errors.append(error)

                if (last_error.mean() - error.mean()) < self.parameters.ja_refinement_min_difference:
                    break

                last_error = error

            iterations += 1
            if iterations >= self.parameters.ja_refinement_max_iterations:
                log.info("Terminating iterative optimization after %d cycles" % self.parameters.ja_refinement_max_iterations)
                break

        # 4th step: reference orientation
        if self.parameters.reference_orientation_enabled:
            self.do_reference_orientation(datasources.get(self.parameters.reference_orientation_datasource, None))
        else:
            self.result.zaxis_reference_result = reference_orientation_null_calibration
            self.result.zaxis_points_result = []


        self.result.orientation_errors.append(self.compute_orientation_errors(datasources.get(self.parameters.gimbal_angle_calibration_datasource, None)))

        # 5th step: gimbalangle correction
        if self.parameters.gimbal_angle_calibration_enabled:
            self.do_gimbalangle_correction(datasources.get(self.parameters.gimbal_angle_calibration_datasource, None))

            # compute errors after calibration
            self.result.orientation_errors.append(self.compute_orientation_errors(datasources.get(self.parameters.gimbal_angle_calibration_datasource, None)))

        else:
            self.result.gimbalangles_correction_result = angle_null_correction

        # do iterative refinement for orientation as well ?

        # finally store or send the results
        ts = measurement.now()
        if self.source_tooltip_calibration_result is not None:
            self.source_tooltip_calibration_result.send(measurement.Pose(ts, self.result.tooltip_calibration_result))
        if self.source_absolute_orientation_result is not None:
            self.source_absolute_orientation_result.send(measurement.Pose(ts, self.result.absolute_orientation_result))
        if self.source_jointangles_correction_result is not None:
            self.source_jointangles_correction_result.send(measurement.Matrix3x3(ts, self.result.jointangles_correction_result))
        if self.source_gimbalangles_correction_result is not None:
            self.source_gimbalangles_correction_result.send(measurement.Matrix3x3(ts, self.result.gimbalangles_correction_result))

        if len(self.result.zaxis_points_result) > 0 and self.source_zaxis_points_result is not None:
            self.source_zaxis_points_result.send(measurement.PositionList(ts, math.PositionList.fromList(self.result.zaxis_points_result)))

        if self.source_zaxis_reference_result is not None:
            self.source_zaxis_reference_result.send(measurement.Position(ts, self.result.zaxis_reference_result))

        # wait a bit before shutting down to allow ubitrack to process the data
        time.sleep(0.1)
        # display nice result graphs or at least text??
        self.result.has_result = True

        # teardown

        self.source_tooltip_calibration_result = None
        self.source_absolute_orientation_result = None
        self.source_jointangles_correction_result = None
        self.source_gimbalangles_correction_result = None

        self.facade.stopDataflow()
        self.facade.clearDataflow()

    def load_datasource(self, config, datasource_sname):
        log.info("Load Datasource: %s" % datasource_sname)
        ds_cfg = dict(config.items(datasource_sname))
        data_directory = ds_cfg["data_directory"]
        reference_data = [(k.replace("reference.", ""), v) for k, v in ds_cfg.items() if k.startswith("reference.")][0]
        items = [(k.replace("item.", ""), v) for k, v in ds_cfg.items() if k.startswith("item.")]

        def mkDSC(name, spec):
            spec_items = [si.strip() for si in spec.split(",")]
            if len(spec_items) < 2:
                raise ValueError("Invalid Configuration for datasource element: %s" % name)
            filename = spec_items[0]
            reader = getattr(util, spec_items[1], None)
            if reader is None:
                raise ValueError("Invalid Configuration for datasource element: %s -> reader not found: %s" % (name, spec_items[1]))
            interpolator = None
            if len(spec_items) > 2:
                interpolator = available_interpolators.get(spec_items[2], None)
                if interpolator is None:
                    raise ValueError("Invalid Configuration for datasource element: %s -> interpolator not found: %s" % (name, spec_items[2]))
            return DSC(name, filename, reader, interpolator=interpolator)

        return loadData(data_directory,
                        mkDSC(*reference_data),
                        items=(mkDSC(*i) for i in items))

    def load_datasources(self):
        all_datasources = set()
        if self.parameters.tooltip_enabled:
            all_datasources.add(self.parameters.tooltip_datasource)
        if self.parameters.absolute_orientation_enabled:
            all_datasources.add(self.parameters.absolute_orientation_datasource)
        if self.parameters.joint_angle_calibration_enabled:
            all_datasources.add(self.parameters.joint_angle_calibration_datasource)
        if self.parameters.reference_orientation_enabled:
            all_datasources.add(self.parameters.reference_orientation_datasource)
        if self.parameters.gimbal_angle_calibration_enabled:
            all_datasources.add(self.parameters.gimbal_angle_calibration_datasource)

        config = self.context.get("config")
        result = {}
        for datasource_sname in sorted(all_datasources):
            data = self.load_datasource(config, datasource_sname)
            record_count = len(data)
            if record_count > 0:
                log.info('Loaded %d records with fieldnames: %s' % (record_count, ','.join(data[0]._fields)))
                result[datasource_sname] = data
            else:
                log.warn('No records loaded!')

        return result

    def get_fwk(self, jointangle_calib, gimbalangle_calib, disable_theta6=False):
        log.info("ForwardKinematics:\njoint_lengths=%s\norigin_offset=%s\njointangle_correction=%s\ngimbalangle_correction=%s" %
                 (self.parameters.joint_lengths, self.parameters.origin_offset, jointangle_calib, gimbalangle_calib))
        return FWKinematicPhantom(self.parameters.joint_lengths,
                                  jointangle_calib,
                                  gimbalangle_calib,
                                  self.parameters.origin_offset,
                                  disable_theta6=disable_theta6)









class OfflineCalibrationController(CalibrationController):
    processor = Typed(OfflineCalibrationProcessor)
    parameters = Typed(OfflineCalibrationParameters)

    bgThread = Typed(BackgroundCalculationThread)

    is_working = Bool(False)
    has_result = Bool(False)

    # results received from background thread
    tooltip_calibration_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    absolute_orientation_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    jointangles_correction_result = Value(angle_null_correction.copy())
    gimbalangles_correction_result = Value(angle_null_correction.copy())

    def _default_parameters(self):
        return OfflineCalibrationParameters()

    def setupController(self, active_widgets=None):
        active_widgets[0].find("btn_start_calibration").visible = False
        active_widgets[0].find("btn_stop_calibration").visible = False
        self.do_reset_parameters()

    def do_reset_parameters(self):
        wiz_cfg = self.wizard_state.config
        gbl_cfg = self.context.get("config")

        # load all parameters from the configuration file
        try:
            haptidevice_name = wiz_cfg.get("haptic_device").strip()
            hd_cfg = dict(gbl_cfg.items("ubitrack.devices.%s" % haptidevice_name))
            self.parameters.joint_lengths = np.array([float(hd_cfg["joint_length1"]),
                                           float(hd_cfg["joint_length2"]), ])

            self.parameters.origin_offset = np.array([float(hd_cfg["origin_offset_x"]),
                                           float(hd_cfg["origin_offset_y"]),
                                           float(hd_cfg["origin_offset_z"]),])
        except Exception, e:
            log.error("Error reading Haptic device configuration. Make sure, the configuration file is correct.")
            log.exception(e)

        if self.module.parent.config_version < 2:
            log.warn("This controller (%s) requires config_version >= 2" % self.module_name)

        parameters_sname = "%s.parameters.%s" % (self.config_ns, self.module_name)
        datasource_sname_prefix = "%s.datasources." % self.config_ns

        if gbl_cfg.has_section(parameters_sname):
            self.parameters.tooltip_enabled = gbl_cfg.getboolean(parameters_sname, "tooltip_enabled")
            if self.parameters.tooltip_enabled:
                log.info("Tooltip Calibration Enabled")
                self.parameters.tooltip_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "tooltip_datasource")
                self.parameters.tt_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "tt_minimal_angle_between_measurements")

            self.parameters.absolute_orientation_enabled = gbl_cfg.getboolean(parameters_sname, "absolute_orientation_enabled")
            if self.parameters.absolute_orientation_enabled:
                log.info("Absolute Orientation Calibration Enabled")
                self.parameters.absolute_orientation_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "absolute_orientation_datasource")
                self.parameters.ao_inital_maxdistance_from_origin = gbl_cfg.getfloat(parameters_sname, "ao_inital_maxdistance_from_origin")
                self.parameters.ao_minimal_distance_between_measurements = gbl_cfg.getfloat(parameters_sname, "ao_minimal_distance_between_measurements")
                self.parameters.ao_refinement_expand_coverage = gbl_cfg.getfloat(parameters_sname, "ao_refinement_expand_coverage")
                self.parameters.ao_refinement_shrink_distance = gbl_cfg.getfloat(parameters_sname, "ao_refinement_shrink_distance")

            self.parameters.joint_angle_calibration_enabled = gbl_cfg.getboolean(parameters_sname, "joint_angle_calibration_enabled")
            if self.parameters.joint_angle_calibration_enabled:
                log.info("Joint-Angle Calibration Enabled")
                self.parameters.joint_angle_calibration_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "joint_angle_calibration_datasource")
                self.parameters.ja_minimal_distance_between_measurements = gbl_cfg.getfloat(parameters_sname, "ja_minimal_distance_between_measurements")
                self.parameters.ja_maximum_distance_to_reference = gbl_cfg.getfloat(parameters_sname, "ja_maximum_distance_to_reference")
                self.parameters.ja_refinement_min_difference = gbl_cfg.getfloat(parameters_sname, "ja_refinement_min_difference")
                self.parameters.ja_refinement_max_iterations = gbl_cfg.getint(parameters_sname, "ja_refinement_max_iterations")
                self.parameters.ja_refinement_shrink_distance = gbl_cfg.getfloat(parameters_sname, "ja_refinement_shrink_distance")

            self.parameters.reference_orientation_enabled = gbl_cfg.getboolean(parameters_sname, "reference_orientation_enabled")
            if self.parameters.reference_orientation_enabled:
                log.info("Reference Orientation Calibration Enabled")
                self.parameters.reference_orientation_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "reference_orientation_datasource")
                self.parameters.ro_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "ro_minimal_angle_between_measurements")

            self.parameters.gimbal_angle_calibration_enabled = gbl_cfg.getboolean(parameters_sname, "gimbal_angle_calibration_enabled")
            if self.parameters.gimbal_angle_calibration_enabled:
                log.info("Gimbal-Angle Calibration Enabled")
                self.parameters.gimbal_angle_calibration_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "gimbal_angle_calibration_datasource")
                self.parameters.ga_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "ga_minimal_angle_between_measurements")

        else:
            log.warn("No parameters found for offline calibration - using defaults. Define parameters in section: %s" % parameters_sname)

    def do_offline_calibration(self):
        if self.processor is None:
            self.processor = OfflineCalibrationProcessor(
                context=self.context,
                config=self.config,
                facade=self.facade,
                dfg_dir=self.dfg_dir,
                dfg_filename=self.dfg_filename,
            )

            self.processor.observe("is_working", self.defer_update_attr)
            self.processor.result.observe(["has_result",
                                           "tooltip_calibration_result",
                                           "absolute_orientation_result",
                                           "jointangles_correction_result",
                                           "gimbalangles_correction_result"],
                                           self.defer_update_attr)

        self.processor.reset(self.parameters)
        self.bgThread = BackgroundCalculationThread(self.processor)
        self.bgThread.start()

    # called from background thread
    def defer_update_attr(self, change):
        if change is not None:
            deferred_call(setattr, self, change['name'], change['value'])


    def do_visualize_results(self):
        # create figures
        from matplotlib.figure import Figure

        result = self.processor.result

        poserr = Figure()
        ax1 = poserr.add_subplot(111)
        ax1.boxplot(result.position_errors)
        ax1.set_title("Position Errors")

        ornerr = Figure()
        ax2 = ornerr.add_subplot(111)
        ax2.boxplot(result.orientation_errors)
        ax2.set_title("Orientation Errors")

        # create results panel
        panel = OfflineCalibrationResultPanel(name="utic.ismar14.offline_calibration_result.%s" % time.time(),
                                              boxplot_position_errors=poserr,
                                              boxplot_orientation_errors=ornerr,
                                              )


        # add to layout
        wizard_controller = self.wizard_state.controller
        parent = wizard_controller.wizview.parent

        panel.set_parent(parent)
        op = FloatItem(item=panel.name,)
        parent.update_layout(op)


class OfflineCalibrationModule(ModuleBase):
    def get_category(self):
        return "Calibration"

    def get_widget_class(self):
        return OfflineCalibrationPanel

    def get_controller_class(self):
        return OfflineCalibrationController