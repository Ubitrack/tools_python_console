__author__ = 'jack'
import logging

import numpy as np
from numpy.linalg import norm
from collections import namedtuple

from atom.api import Atom, Value, Float, Int, List, observe
import time

from ubitrack.core import measurement, math, util

log = logging.getLogger(__name__)


class CalibrationProcessor(Atom):

    # the facade will be set from the controller
    facade = Value()

    def prepare_stream(self, stream,
                       tooltip_offset=None,
                       absolute_orientation=None,
                       forward_kinematics=None):
        return stream


class TooltipCalibrationProcessor(CalibrationProcessor):

    max_computation_time = Float(30)

    # input data will be set from the controller
    data_tracker_poses = List()

    # resulting tooltip offset is received from dataflow
    result_tooltip_offset = Value(None)
    result_last_timestamp = Value(None)

    # dataflow components
    sink_result_tooltip_offset = Value()
    source_tracker_poses = Value()

    @observe("facade")
    def handle_facade_change(self, change):
        facade = change['value']
        if facade is None:
            if self.sink_result_tooltip_offset is not None:
                self.sink_result_tooltip_offset.setCallback(None)
            self.source_tracker_poses = None
            self.sink_result_tooltip_offset = None
        else:
            self.sink_result_tooltip_offset = facade.instance.getApplicationPushSinkPosition("calib_tooltip_out")
            self.sink_result_tooltip_offset.setCallback(self.handler_result_tooltip_offset)
            self.source_tracker_poses = facade.instance.getApplicationPushSourcePose("tt_calib_et_pose")

    def handler_result_tooltip_offset(self, m):
        self.result_tooltip_offset = m.get()
        self.result_last_timestamp = m.time()

    def run(self):
        ts = measurement.now()
        for p in self.data_tracker_poses:
            ts += 1
            self.source_tracker_poses.send(measurement.Pose(ts, p))

        wait_time = 0
        while True:
            time.sleep(0.5)
            wait_time += 0.5
            if wait_time > self.max_computation_time:
                log.warn("No result produced for Tooltip Calibration due to timeout")
                break
            if self.result_last_timestamp is not None:
                if self.result_last_timestamp >= ts:
                    break

        return self.result_tooltip_offset


class AbsoluteOrientationCalibrationProcessor(CalibrationProcessor):

    max_computation_time = Float(30)

    # input data will be set from the controller
    data_tracker_hip_positions = List()
    data_fwk_hip_positions = List()

    # resulting absolute orientation transform is received from dataflow
    result_absolute_orientation = Value()
    result_last_timestamp = Value()

    # dataflow components
    sink_result_absolute_orientation = Value()
    source_tracker_hip_positions = Value()
    source_fwk_hip_positions = Value()

    @observe("facade")
    def handle_facade_change(self, change):
        facade = change['value']
        if facade is None:
            if self.sink_result_absolute_orientation is not None:
                self.sink_result_absolute_orientation.setCallback(None)
            self.sink_result_absolute_orientation = None
            self.source_tracker_hip_positions = None
            self.source_fwk_hip_positions = None
        else:
            self.sink_result_absolute_orientation = facade.instance.getApplicationPushSinkPose("calib_absolute_orientation_out")
            self.sink_result_absolute_orientation.setCallback(self.handler_result_absolute_orientation)
            self.source_tracker_hip_positions = facade.instance.getApplicationPushSourcePositionList("ao_calib_et_positions")
            self.source_fwk_hip_positions = facade.instance.getApplicationPushSourcePositionList("ao_calib_fwk_positions")

    def handler_result_absolute_orientation(self, m):
        self.result_absolute_orientation = m.get()
        self.result_last_timestamp = m.time()

    def run(self):
        ts = measurement.now()

        acfp = measurement.PositionList(ts, math.PositionList.fromList([p for p in self.data_fwk_hip_positions]))
        acep = measurement.PositionList(ts, math.PositionList.fromList([p for p in self.data_tracker_hip_positions]))

        self.source_tracker_hip_positions.send(acep)
        self.source_fwk_hip_positions.send(acfp)

        wait_time = 0
        while True:
            time.sleep(0.5)
            wait_time += 0.5
            if wait_time > self.max_computation_time:
                log.warn("No result produced for Absolute Orientation due to timeout")
                break
            if self.result_last_timestamp is not None:
                if self.result_last_timestamp >= ts:
                    break

        return self.result_absolute_orientation

    def prepare_stream(self, stream,
                       tooltip_offset=None,
                       absolute_orientation=None,
                       forward_kinematics=None):

        if tooltip_offset is None:
            raise ValueError("TooltipOffset not supplied")

        if forward_kinematics is None:
            raise ValueError("ForwardKinematics not supplied")


        stream_fields = stream[0]._fields

        data_fieldnames = list(stream_fields)
        data_fieldnames.append("haptic_pose")
        data_fieldnames.append("externaltracker_hip_position")

        DataSet = namedtuple('DataSet', data_fieldnames)

        result = []

        for record in stream:
            haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
            externaltracker_hip_position = record.externaltracker_pose * tooltip_offset

            values = list(record) + [haptic_pose, externaltracker_hip_position]
            result.append(DataSet(*values))

        return result


class JointAngleCalibrationProcessor(CalibrationProcessor):

    # input data will be set from the controller
    data_tracker_hip_positions = List()
    data_joint_angles = List()

    # dataflow components
    sink_result_jointangle_correction = Value()
    source_tracker_hip_positions = Value()
    source_joint_angles = Value()

    @observe("facade")
    def handle_facade_change(self, change):
        facade = change['value']
        if facade is None:
            self.sink_result_jointangle_correction = None
            if self.source_tracker_hip_positions is not None:
                self.source_tracker_hip_positions.setCallback(None)
            self.source_tracker_hip_positions = None
            if self.source_joint_angles is not None:
                self.source_joint_angles.setCallback(None)
            self.source_joint_angles = None
        else:
            self.sink_result_jointangle_correction = facade.instance.getApplicationPullSinkMatrix3x3("calib_phantom_jointangle_correction_out")

            self.source_tracker_hip_positions = facade.instance.getApplicationPullSourcePositionList("ja_calib_hip_positions")
            self.source_tracker_hip_positions.setCallback(self.handler_input_hip_positions)

            self.source_joint_angles = facade.instance.getApplicationPullSourcePositionList("ja_calib_jointangles")
            self.source_joint_angles.setCallback(self.handler_input_joint_angles)

    def handler_input_hip_positions(self, ts):
        pl = math.PositionList.fromList(self.data_tracker_hip_positions)
        return measurement.PositionList(ts, pl)

    def handler_input_joint_angles(self, ts):
        pl = math.PositionList.fromList(self.data_joint_angles)
        return measurement.PositionList(ts, pl)

    def run(self):
        ts = measurement.now()
        return self.sink_result_jointangle_correction.get(ts).get()

    def prepare_stream(self, stream,
                       tooltip_offset=None,
                       absolute_orientation=None,
                       forward_kinematics=None):

        if tooltip_offset is None:
            raise ValueError("TooltipOffset not supplied")

        if absolute_orientation is None:
            raise ValueError("AbsoluteOrientation not supplied")

        if forward_kinematics is None:
            raise ValueError("ForwardKinematics not supplied")


        stream_fields = stream[0]._fields

        data_fieldnames = list(stream_fields)
        data_fieldnames.append("haptic_pose")
        data_fieldnames.append("hip_reference_pose")

        DataSet = namedtuple('DataSet', data_fieldnames)

        absolute_orientation_inv = absolute_orientation.invert()

        result = []

        for record in stream:
            haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
            hip_reference_pose = (
            absolute_orientation_inv * record.externaltracker_pose * math.Pose(math.Quaternion(), tooltip_offset))

            values = list(record) + [haptic_pose, hip_reference_pose]
            result.append(DataSet(*values))

        return result
