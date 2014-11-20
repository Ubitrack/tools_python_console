__author__ = 'jack'

from atom.api import Atom, Value, List, Bool
from collections import namedtuple
import logging

from ubitrack.core import math
import numpy as np

log = logging.getLogger(__name__)



class BaseStreamProcessor(Atom):

    name = None
    # data and setup
    raw_data = Value()

    # static attributes, set by subclasses
    required_fields = None
    required_attributes = None

    # process attributes
    input_field_names = List()
    output_field_names = List()


    def _default_input_field_names(self):
        if self.raw_data:
            return list(self.raw_data[0]._fields)
        return []

    def check_input(self):
        input_ok = True
        if self.required_fields is not None:
            fieldnames = self.input_field_names
            # check for required fields
            for name in self.required_fields:
                if name not in fieldnames:
                    log.warn("Required field: %s not in input_stream" % name)
                    input_ok = False

        if self.required_attributes is not None:
            attributes = [a for a in self.members().keys() if not a.startswith("_")]
            for name in self.required_attributes:
                if name not in attributes:
                    log.warn("Required attribute: %s not set" % name)
                    input_ok = False

        return input_ok

    def emit(self):
        raise NotImplementedError


class NullStreamProcessor(BaseStreamProcessor):

    name = "Null"
    required_fields = []
    required_attributes = []

    def _default_output_field_names(self):
        return list(self.input_field_names)

    def emit(self):
        if self.check_input():
            return self.raw_data
        return None


class TooltipStreamProcessor(NullStreamProcessor):

    name = "Tooltip"
    required_fields = ['externaltracker_pose',]
    required_attributes = []


class AbsoluteOrientationStreamProcessor(BaseStreamProcessor):

    name = "AbsoluteOrientation"
    required_fields = ['externaltracker_pose', 'jointangles', 'gimbalangles']
    required_attributes = ['tooltip_offset', 'forward_kinematics']


    def _default_output_field_names(self):
        data_fieldnames = list(self.input_field_names)
        data_fieldnames.append("haptic_pose")
        data_fieldnames.append("externaltracker_hip_position")
        return data_fieldnames


    tooltip_offset     = Value()
    forward_kinematics = Value()

    def emit(self):

        if not self.check_input():
            return None


        DataSet = namedtuple('DataSet', self.output_field_names)

        result = []

        for record in self.raw_data:
            haptic_pose = self.forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
            externaltracker_hip_position = (record.externaltracker_pose * self.tooltip_offset).translation()

            values = list(record) + [haptic_pose, externaltracker_hip_position]
            result.append(DataSet(*values))

        return result


class JointAngleCalibrationStreamProcessor(BaseStreamProcessor):

    name = "JointAngleCalibration"
    required_fields = ['externaltracker_pose', 'jointangles', 'gimbalangles']
    required_attributes = ['tooltip_offset', 'absolute_orientation', 'forward_kinematics']

    def _default_output_field_names(self):
        data_fieldnames = list(self.input_field_names)
        data_fieldnames.append("haptic_pose")
        data_fieldnames.append("hip_reference_pose")
        return data_fieldnames


    tooltip_offset       = Value()
    absolute_orientation = Value()
    forward_kinematics   = Value()

    def emit(self):

        if not self.check_input():
            return None

        DataSet = namedtuple('DataSet', self.output_field_names)

        absolute_orientation_inv = self.absolute_orientation.invert()
        result = []

        for record in self.raw_data:
            haptic_pose = self.forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
            hip_reference_pose = (
            absolute_orientation_inv * record.externaltracker_pose * self.tooltip_offset)

            values = list(record) + [haptic_pose, hip_reference_pose]
            result.append(DataSet(*values))

        return result


class GimbalAngleCalibrationStreamProcessor(BaseStreamProcessor):

    name = "GimbalAngleCalibration"
    required_fields = ['externaltracker_pose', 'jointangles', 'gimbalangles']
    required_attributes = ['zrefaxis_calib', 'absolute_orientation', 'forward_kinematics']

    def _default_output_field_names(self):
        data_fieldnames = list(self.input_field_names)
        data_fieldnames.append("haptic_pose")
        data_fieldnames.append("zrefaxis")
        return data_fieldnames


    absolute_orientation = Value()
    zrefaxis_calib       = Value()
    forward_kinematics   = Value()

    def emit(self):

        if not self.check_input():
            return None

        DataSet = namedtuple('DataSet', self.output_field_names)

        absolute_orientation_inv = self.absolute_orientation.invert()

        result = []

        for record in self.raw_data:
            haptic_pose = self.forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)

            # HIP target pose in HDorigin
            hiptarget_rotation = math.Quaternion((absolute_orientation_inv * record.externaltracker_pose).rotation())
            ht_pose_no_trans = math.Pose(hiptarget_rotation, np.array([0, 0, 0]))

            # re-orient zrefaxis_calib using hiptarget pose
            zrefaxis = ht_pose_no_trans * self.zrefaxis_calib
            zrefaxis = zrefaxis / np.linalg.norm(zrefaxis)

            values = list(record) + [haptic_pose, zrefaxis]
            result.append(DataSet(*values))

        return result


class ReferenceOrientationStreamProcessor(BaseStreamProcessor):

    name = "ReferenceOrientation"
    required_fields = ['externaltracker_pose', 'externaltracker_markers', 'jointangles', 'gimbalangles']
    required_attributes = ['tooltip_offset', 'absolute_orientation', 'forward_kinematics', 'forward_kinematics_5dof']

    def _default_output_field_names(self):
        data_fieldnames = list(self.input_field_names)
        data_fieldnames.append("haptic_pose")
        data_fieldnames.append("mean_marker_error")
        data_fieldnames.append("target_position")
        data_fieldnames.append("target_markers")
        data_fieldnames.append("device_to_stylus_5dof")
        data_fieldnames.append("hiptarget_pose")
        return data_fieldnames


    tooltip_offset       = Value()
    absolute_orientation = Value()
    forward_kinematics   = Value()
    forward_kinematics_5dof = Value()

    use_markers = Bool(True)

    def emit(self):

        if not self.check_input():
            return None

        DataSet = namedtuple('DataSet', self.output_field_names)

        absolute_orientation_inv = self.absolute_orientation.invert()

        # find marker count and verify that it is constant for the complete dataset
        nmarkers = 0
        if self.use_markers:
            nmarkers = len(self.raw_data[0].externaltracker_markers)
            assert (np.asarray([len(d.externaltracker_markers) for d in self.raw_data
                                if d.externaltracker_markers is not None]) == nmarkers).all()

        rel_marker_positions = []
        skipped_markers = 0

        result = []

        for record in self.raw_data:
            # fwk pose in HDorigin
            haptic_pose = self.forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)

            # HIP target pose in HDorigin
            hiptarget_pose = absolute_orientation_inv * record.externaltracker_pose
            hiptarget_pose_inv = hiptarget_pose.invert()

            mean_marker_error = None
            markers = []

            if self.use_markers:
                if record.externaltracker_markers is not None:
                    # HIP target markers in HDorigin
                    hiptarget_markers = [absolute_orientation_inv * m for m in record.externaltracker_markers]
                    if not rel_marker_positions:
                        # initialize positions
                        for m in hiptarget_markers:
                            rel_marker_positions.append(hiptarget_pose_inv * m)
                            markers.append(m)
                    else:
                        # validate positions
                        markers = [None, ] * nmarkers
                        dbg_dists = []
                        for m in hiptarget_markers:
                            dist_ = []
                            for i, relm in enumerate(rel_marker_positions):
                                dist_.append((i, norm(relm - (hiptarget_pose_inv * m))))
                            dist_ = sorted(dist_, lambda x, y: cmp(x[1], y[1]))
                            markers[dist_[0][0]] = m
                            dbg_dists.append(dist_)

                        mean_marker_error = np.asarray([d[0][1] for d in dbg_dists]).mean()

                        # XXX remove print statements or improve to make useful logging output
                        if None in markers:
                            log.warn("Incomplete marker dataset received!")
                            # print "Markers:"
                            # for m in markers:
                            #     print m
                            # print "Distances:"
                            # for i, dist_ in enumerate(dbg_dists):
                            #     print "Item%d" % i
                            #     for i,m in dist_:
                            #         print i, m
                            skipped_markers += 1
                            markers = []

            # calculate the stylus pose (5DOF) based on the calibrated correction_factors and the measured angles
            device_to_stylus_5dof = self.forward_kinematics_5dof.calculate_pose(record.jointangles, record.gimbalangles)
            device_to_stylus_5dof_inv = device_to_stylus_5dof.invert()

            # multiply the inverse of the externaltracker to stylus transform with the externaltracker to stylus target
            target_position = (device_to_stylus_5dof_inv * hiptarget_pose).translation()

            # back-project markers using the corrected stylus pose (5dof)
            if use_markers and markers:
                target_markers = np.asarray([device_to_stylus_5dof_inv * m for m in markers])
            else:
                target_markers = None

            values = list(record) + [haptic_pose, mean_marker_error,
                                     target_position, target_markers, device_to_stylus_5dof,
                                     hiptarget_pose]
            result.append(DataSet(*values))

        return result