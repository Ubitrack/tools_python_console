__author__ = 'jack'

import os
import logging
import numpy as np
from collections import namedtuple

log = logging.getLogger(__name__)

from atom.api import Event, Bool, Str, Value, Typed, Float, observe
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
    RelativeOrienationDistanceStreamFilter, StaticPointDistanceStreamFilter, RelativePointDistanceStreamFilter
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



def refine_datastream(stream, tooltip_offset, absolute_orientation, forward_kinematics):
    """
    expects a stream with the following attributes:
    - externaltracker_pose
    - joint_angles
    - gimbal_angles (optional)
    """

    stream_fields = stream[0]._fields

    has_gimbalangles = False
    if "gimbalangles" in stream_fields:
        has_gimbalangles = True

    data_fieldnames = list(stream_fields)
    data_fieldnames.append("haptic_pose")
    data_fieldnames.append("externaltracker_hip_position")
    data_fieldnames.append("hip_reference_pose")

    DataSet = namedtuple('DataSet', data_fieldnames)

    absolute_orientation_inv = absolute_orientation.invert()

    result = []

    # placeholder if not available in dataset
    gimbalangles = np.array([0, 0, 0])

    for record in stream:
        if has_gimbalangles:
            gimbalangles = record.gimbalangles

        haptic_pose = forward_kinematics.calculate_pose(record.jointangles, gimbalangles)
        externaltracker_hip_position = record.externaltracker_pose * tooltip_offset
        hip_reference_pose = (absolute_orientation_inv * record.externaltracker_pose * tooltip_offset)

        values = list(record) + [haptic_pose, externaltracker_hip_position, hip_reference_pose]
        result.append(DataSet(*values))

    return result


class OfflineCalibrationController(CalibrationController):

    bgThread = Typed(BackgroundCalculationThread)
    is_working = Bool(False)


    # system configuration options
    record_dir = Value()

    # configuration parameters
    tt_minimal_angle_between_measurements = Float(0.1)
    ao_inital_maxdistance_from_origin = Float(0.1)
    ao_minimal_distance_between_measurements = Float(0.01)

    joint_lengths = Value(np.array([0.13335, 0.13335]))
    origin_offset = Value(np.array([0.0, -0.11, -0.035]))




    # results generated (iteratively)
    tooltip_calibration_result = Value()
    absolute_orientation_result = Value()

    def _default_record_dir(self):
        root = self.config.get("offline_root_temp")
        return os.path.join(root, "input_data")


    def setupController(self, active_widgets=None):
        active_widgets[0].find("btn_start_calibration").visible = False
        active_widgets[0].find("btn_stop_calibration").visible = False
        # super(OfflineCalibrationController, self).setupController(active_widgets=active_widgets)


    def do_offline_calibration(self):
        self.bgThread = BackgroundCalculationThread(self)
        self.bgThread.start()

    def process(self):

        fname = os.path.join(self.dfg_dir, self.dfg_filename)
        if not os.path.isfile(fname):
            log.error("DFG file not found: %s" % fname)
            return

        self.facade.loadDataflow(fname)
        self.facade.startDataflow()

        # 1st step: Tooltip Calibration (uses step01 data)
        tt_data = self.load_data_step01()
        tt_selector = RelativeOrienationDistanceStreamFilter("externaltracker_pose",
                                                             min_distance=self.tt_minimal_angle_between_measurements)
        selected_tt_data = tt_selector.process(tt_data)
        log.info("Offline Tooltip Calibration (%d out of %d records selected)" % (len(selected_tt_data), len(tt_data)))
        tt_processor = TooltipCalibrationProcessor()

        tt_processor.data_tracker_poses = [r.externaltracker_pose for r in selected_tt_data]
        tt_processor.facade = self.facade

        self.tooltip_calibration_result = tt_processor.run()

        tt_processor.facade = None
        log.info("Result for Tooltip Calibration: %s" % str(self.tooltip_calibration_result))


        # 2nd step: initial absolute orientation (uses step03  data)
        ao_data = self.load_data_step03()

        calib_null = np.array([[0.0, 1.0, 0.0,], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        absolute_orientation_null = math.Pose(math.Quaternion(), np.array([0,0,0]))

        fwk_null = self.get_fwk(calib_null, calib_null)

        ao_data_ext = refine_datastream(ao_data, self.tooltip_calibration_result, absolute_orientation_null, fwk_null)

        ao_selector1 = StaticPointDistanceStreamFilter("haptic_pose", np.array([0,0,0]),
                                                       max_distance=self.ao_inital_maxdistance_from_origin)

        ao_selector2 = RelativePointDistanceStreamFilter("haptic_pose",
                                                         min_distance= self.ao_minimal_distance_between_measurements)

        selected_ao_data = ao_selector2.process(ao_selector1.process(ao_data_ext))
        log.info("Absolute Orientation Calibration (%d out of %d records selected)" % (len(selected_ao_data), len(ao_data)))

        ao_processor = AbsoluteOrientationCalibrationProcessor()
        ao_processor.data_tracker_hip_positions = [r.externaltracker_hip_position for r in selected_ao_data]
        ao_processor.data_fwk_hip_positions = [r.haptic_pose.translation() for r in selected_ao_data]
        ao_processor.facade = self.facade

        self.absolute_orientation_result = ao_processor.run()
        ao_processor.facade = None
        log.info("Result for Absolute Orientation: %s" % str(self.absolute_orientation_result))




        self.facade.stopDataflow()
        self.facade.clearDataflow()




    def load_data_step01(self):
        return loadData(
                os.path.join(self.record_dir, "step01"),
                DSC('externaltracker_pose', 'tracker_pose.log',
                    util.PoseStreamReader),
                    items=(DSC('jointangles', 'joint_angles.log',
                               util.PositionStreamReader, interpolateVec3List),
                           DSC('gimbalangles', 'gimbal_angles.log',
                               util.PositionStreamReader, interpolateVec3List),
                           )
                )

    def load_data_step02(self):
        return loadData(
                os.path.join(self.record_dir, "step02"),
                DSC('externaltracker_pose', 'tracker_pose.log',
                    util.PoseStreamReader),
                    items=(DSC('jointangles', 'joint_angles.log',
                               util.PositionStreamReader, interpolateVec3List),
                           DSC('gimbalangles', 'gimbal_angles.log',
                               util.PositionStreamReader, interpolateVec3List),
                           )
                )

    def load_data_step03(self):
        return loadData(
                os.path.join(self.record_dir, "step03"),
                DSC('externaltracker_pose', 'tracker_pose.log',
                    util.PoseStreamReader),
                    items=(DSC('jointangles', 'joint_angles.log',
                               util.PositionStreamReader, interpolateVec3List),
                           # DSC('gimbalangles', 'gimbal_angles.log',
                           #     util.PositionStreamReader, interpolateVec3List),
                           )
                )

    def load_data_step04(self):
        return loadData(
                os.path.join(self.record_dir, "step04"),
                DSC('externaltracker_pose', 'tracker_pose.log',
                    util.PoseStreamReader),
                    items=(DSC('jointangles', 'joint_angles.log',
                               util.PositionStreamReader, interpolateVec3List),
                           DSC('gimbalangles', 'gimbal_angles.log',
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