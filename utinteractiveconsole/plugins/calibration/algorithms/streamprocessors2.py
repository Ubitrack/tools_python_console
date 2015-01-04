__author__ = 'jack'

from atom.api import Atom, Value, List, Bool, Typed

import new
import logging

from ubitrack.core import math
import numpy as np
from numpy.linalg import norm

log = logging.getLogger(__name__)

from utinteractiveconsole.persistence.recordsource import BaseRecordSource
from utinteractiveconsole.persistence.recordschema import RequiredField, Field, RecordSchema

class BaseStreamProcessor(Atom):

    name = None
    # data and setup
    recordsource = Typed(BaseRecordSource)

    schema = Typed(RecordSchema)

    # static attributes, set by subclasses
    required_fields = None
    additional_fields = None
    required_attributes = None


    @property
    def input_fieldnames(self):
        return self.recordsource.output_fieldnames

    @property
    def output_fieldnames(self):
        fieldnames = self.input_fieldnames
        if self.additional_fields is not None:
            for field in self.additional_fields:
                fieldnames.append(field.name)
        return fieldnames

    def _default_schema(self):
        return RecordSchema(fields=self.recordsource.schema.fields + (self.additional_fields or []))

    def make_record_class(self):
        rcls = self.recordsource.record_class
        if not self.additional_fields:
            return rcls
        attrs = {}
        for field in self.additional_fields:
            if field.name in attrs:
                log.warn("Duplicate field: %s defined for recordsource: %s" % (field.name, self.name))
            # XXX could use Coerced with datatype here ...
            attrs[field.name] = Value()
        return new.classobj('%s_%s' % (rcls.__name__, self.name), (rcls,), attrs)

    def check_input(self):
        input_ok = True
        if self.required_fields is not None:
            # XXX could use schema here to do datatype checking
            fieldnames = self.input_fieldnames
            # check for required fields
            for field in self.required_fields:
                if field.name not in fieldnames:
                    log.warn("Required field: %s not in input_stream" % field.name)
                    input_ok = False

        if self.required_attributes is not None:
            for name in self.required_attributes:
                if getattr(self, name, None) is None:
                    log.warn("Required attribute: %s not set" % name)
                    input_ok = False

        return input_ok

    def __iter__(self):
        raise NotImplementedError


class NullStreamProcessor(BaseStreamProcessor):

    name = "Null"
    required_fields = []
    additional_fields = []

    required_attributes = []

    def __iter__(self):
        rcls = self.make_record_class()
        parent_fields = self.input_fieldnames
        
        if self.check_input():
            for record in self.recordsource:
                attrs = dict((k, getattr(record, k)) for k in parent_fields)
                
                yield rcls(**attrs)


class TooltipStreamProcessor(NullStreamProcessor):

    name = "Tooltip"
    required_fields = [RequiredField(name='externaltracker_pose', datatype='pose'),]
    additional_fields = []

    required_attributes = []


class AbsoluteOrientationStreamProcessor(BaseStreamProcessor):

    name = "AbsoluteOrientation"
    required_fields = [RequiredField(name='externaltracker_pose', datatype='pose'),
                       RequiredField(name='jointangles', datatype='position3d'),
                       #RequiredField(name='gimbalangles', datatype='position3d'),
                       ]
    additional_fields = [Field(name='haptic_pose', datatype='pose', is_computed=True),
                         Field(name='externaltracker_hip_position', datatype='position3d', is_computed=True)]

    required_attributes = ['tooltip_offset', 'forward_kinematics']

    tooltip_offset = Value()
    forward_kinematics = Value()

    def __iter__(self):

        if not self.check_input():
            raise StopIteration()

        rcls = self.make_record_class()
        parent_fields = self.input_fieldnames

        for record in self.recordsource:
            attrs = dict((k, getattr(record, k)) for k in parent_fields)
            gimbalangles = getattr(record, 'gimbalangles', np.array([0., 0., 0.]))
            attrs['haptic_pose'] = self.forward_kinematics.calculate_pose(record.jointangles, gimbalangles)
            attrs['externaltracker_hip_position'] = (record.externaltracker_pose * self.tooltip_offset).translation()
            
            yield rcls(**attrs)



class JointAngleCalibrationStreamProcessor(BaseStreamProcessor):

    name = "JointAngleCalibration"
    required_fields = [RequiredField(name='externaltracker_pose', datatype='pose'),
                       RequiredField(name='jointangles', datatype='position3d'),
                       #RequiredField(name='gimbalangles', datatype='position3d'),
                       ]
    additional_fields = [Field(name='haptic_pose', datatype='pose', is_computed=True),
                         Field(name='hip_reference_pose', datatype='pose', is_computed=True)]

    required_attributes = ['tooltip_offset', 'absolute_orientation', 'forward_kinematics']

    tooltip_offset = Value()
    absolute_orientation = Value()
    forward_kinematics = Value()

    def __iter__(self):

        if not self.check_input():
            raise StopIteration()
        
        rcls = self.make_record_class()
        parent_fields = self.input_fieldnames

        absolute_orientation_inv = self.absolute_orientation.invert()

        for record in self.recordsource:
            attrs = dict((k, getattr(record, k)) for k in parent_fields)
            gimbalangles = getattr(record, 'gimbalangles', np.array([0., 0., 0.]))
            attrs['haptic_pose'] = self.forward_kinematics.calculate_pose(record.jointangles, gimbalangles)
            attrs['hip_reference_pose'] = (absolute_orientation_inv * record.externaltracker_pose * self.tooltip_offset)
            
            yield rcls(**attrs)


class GimbalAngleCalibrationStreamProcessor(BaseStreamProcessor):

    name = "GimbalAngleCalibration"
    required_fields = [RequiredField(name='externaltracker_pose', datatype='pose'),
                       RequiredField(name='jointangles', datatype='position3d'),
                       RequiredField(name='gimbalangles', datatype='position3d')]
    additional_fields = [Field(name='haptic_pose', datatype='pose', is_computed=True),
                         Field(name='zrefaxis', datatype='position3d', is_computed=True)]

    required_attributes = ['tooltip_offset', 'zrefaxis_calib', 'absolute_orientation', 'forward_kinematics']

    tooltip_offset = Value()
    absolute_orientation = Value()
    zrefaxis_calib = Value()
    forward_kinematics = Value()

    def __iter__(self):

        if not self.check_input():
            raise StopIteration()
        
        rcls = self.make_record_class()
        parent_fields = self.input_fieldnames

        absolute_orientation_inv = self.absolute_orientation.invert()

        for record in self.recordsource:
            attrs = dict((k, getattr(record, k)) for k in parent_fields)
            attrs['haptic_pose'] = self.forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)

            # HIP target pose in HDorigin
            hiptarget_rotation = math.Quaternion((absolute_orientation_inv * record.externaltracker_pose * self.tooltip_offset).rotation())
            ht_pose_no_trans = math.Pose(hiptarget_rotation, np.array([0, 0, 0]))

            # re-orient zrefaxis_calib using hiptarget pose
            zrefaxis = ht_pose_no_trans * self.zrefaxis_calib
            # zrefaxis = hiptarget_rotation.transformVector(self.zrefaxis_calib)
            attrs['zrefaxis'] = zrefaxis / np.linalg.norm(zrefaxis)

            yield rcls(**attrs)


class ReferenceOrientationStreamProcessor(BaseStreamProcessor):

    name = "ReferenceOrientation"
    required_fields = [RequiredField(name='externaltracker_pose', datatype='pose'),
                       RequiredField(name='jointangles', datatype='position3d'),
                       RequiredField(name='gimbalangles', datatype='position3d'),
                       RequiredField(name='externaltracker_markers', datatype='position3d', is_array=True)]

    additional_fields = [Field(name='haptic_pose', datatype='pose', is_computed=True),
                         Field(name='mean_marker_error', datatype='distance', is_computed=True),
                         Field(name='target_position', datatype='position3d', is_computed=True),
                         Field(name='target_markers', datatype='position3d', is_array=True, is_computed=True),
                         Field(name='device_to_stylus_5dof', datatype='pose', is_computed=True),
                         Field(name='hiptarget_pose', datatype='pose', is_computed=True)]

    required_attributes = ['tooltip_offset', 'absolute_orientation', 'forward_kinematics', 'forward_kinematics_5dof']

    tooltip_offset = Value()
    absolute_orientation = Value()
    forward_kinematics = Value()
    forward_kinematics_5dof = Value()

    use_markers = Bool(True)

    def __iter__(self):

        if not self.check_input():
            raise StopIteration()
        
        rcls = self.make_record_class()
        parent_fields = self.input_fieldnames

        absolute_orientation_inv = self.absolute_orientation.invert()

        # find marker count and verify that it is constant for the complete dataset
        nmarkers = 0
        rel_marker_positions = []
        skipped_markers = 0


        for record in self.recordsource:
            attrs = dict((k, getattr(record, k)) for k in parent_fields)
            if nmarkers == 0 and self.use_markers and record.externaltracker_markers is not None:
                nmarkers = len(record.externaltracker_markers)

            if len(record.externaltracker_markers) != nmarkers:
                log.warn("Skipping record with invalid number of markers: %s (expected: %s)" % 
                         (len(record.externaltracker_markers), nmarkers))
                skipped_markers += 1
                continue
                
            # fwk pose in HDorigin
            attrs['haptic_pose'] = self.forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)

            # HIP target pose in HDorigin
            attrs['hiptarget_pose'] = hiptarget_pose = absolute_orientation_inv * record.externaltracker_pose
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
                            skipped_markers += 1
                            continue
            attrs['mean_marker_error'] = mean_marker_error

            # calculate the stylus pose (5DOF) based on the calibrated correction_factors and the measured angles
            attrs['device_to_stylus_5dof'] = device_to_stylus_5dof = \
                self.forward_kinematics_5dof.calculate_pose(record.jointangles, record.gimbalangles)
            device_to_stylus_5dof_inv = device_to_stylus_5dof.invert()

            # multiply the inverse of the externaltracker to stylus transform with the externaltracker to stylus target
            attrs['target_position'] = (device_to_stylus_5dof_inv * hiptarget_pose).translation()

            # back-project markers using the corrected stylus pose (5dof)
            if self.use_markers and markers:
                attrs['target_markers'] = np.asarray([device_to_stylus_5dof_inv * m for m in markers])
            else:
                attrs['target_markers'] = None

            yield rcls(**attrs)
