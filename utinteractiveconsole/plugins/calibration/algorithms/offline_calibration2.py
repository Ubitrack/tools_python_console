__author__ = 'jack'
import logging

import os

from math import sin, cos, acos, atan2, sqrt, fabs, radians, degrees
import numpy as np
import pandas as pd
from numpy.linalg import norm

from scipy import odr
from scipy.signal import correlate
from scipy import interpolate
from scipy import optimize
from scipy import stats

from atom.api import Atom, Bool, Str, Value, Typed, List, Dict, Float, Int, Enum, observe
import time

from ubitrack.core import measurement, math, util, calibration

from utinteractiveconsole.persistence.dataset import DataSet
from utinteractiveconsole.persistence.recordschema import DataType
from utinteractiveconsole.persistence.recordsource import RecordSource, StreamInterpolator
from utinteractiveconsole.persistence.streamfile import StreamFileSpec

from utinteractiveconsole.plugins.calibration.algorithms.phantom_forward_kinematics import (
    FWKinematicPhantom,
    FWKinematicPhantom2
)
from utinteractiveconsole.plugins.calibration.algorithms.scale_forward_kinematics import (
    FWKinematicVirtuose,
    FWKinematicScale
)

from utinteractiveconsole.plugins.calibration.algorithms.streamfilters2 import (
    RelativeOrienationDistanceStreamFilter, StaticPointDistanceStreamFilter,
    RelativePointDistanceStreamFilter, TwoPointDistanceStreamFilter,
    NClustersPositionStreamFilter, NClustersOrientationStreamFilter,
    SkipFrontStreamFilter, ExcludeTimestampsStreamFilter
)

from utinteractiveconsole.plugins.calibration.algorithms.streamprocessors2 import (
    TooltipStreamProcessor, AbsoluteOrientationStreamProcessor, JointAngleCalibrationStreamProcessor,
    GimbalAngleCalibrationStreamProcessor, ReferenceOrientationStreamProcessor
)

# Initialization of disabled steps
tooltip_null_calibration = math.Pose(math.Quaternion(), np.array([0., 0., 0.]))
absolute_orientation_null_calibration = math.Pose(math.Quaternion(), np.array([0., 0., 0.]))
reference_orientation_null_calibration = np.array([0., 0., 1.])
angle_null_correction = np.array([[0.0, 1.0, 0.0, ], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
fwkbase_position_null_calibration = np.array([0., 0., 0.])

log = logging.getLogger(__name__)


class CalibrationProcessor(Atom):

    # the facade will be set from the controller
    facade = Value()

    # the data stream to be consumed by this processor
    dataset = Typed(DataSet)


class TooltipCalibrationProcessor(CalibrationProcessor):

    # data extracted from stream
    data_tracker_poses = List()

    # resulting tooltip offset is received from dataflow
    result_tooltip_offset = Value(None)
    use_tooltip_pose = Bool(False)

    def run(self):
        self.data_tracker_poses = []
        record_count = 0
        for record in self.dataset:
            self.data_tracker_poses.append(record.externaltracker_pose)
            record_count += 1
        log.info("Offline Tooltip Calibration (%d records selected)" % (record_count,))

        if record_count < 5:
            log.error("Insufficient poses for tooltip calibration")
            raise ValueError("Insufficient poses for tooltip calibration")

        result = calibration.tipCalibrationPose(math.PoseList.fromList(self.data_tracker_poses))
        if self.use_tooltip_pose:
            self.result_tooltip_offset = result
        else:
            self.result_tooltip_offset = math.Pose(math.Quaternion(), result.translation())

        # XXX calculate RMS Error here and report it.
        return self.result_tooltip_offset


class TooltipAbsolutePositionCalibrationProcessor(TooltipCalibrationProcessor):

    result_absolute_position = Value()

    def run(self):
        tt_position = super(TooltipAbsolutePositionCalibrationProcessor, self).run().translation()

        # now transform tt_position with all poses to compute the ErrorPosition
        results = []
        for pose in self.data_tracker_poses:
            results.append(pose * tt_position)

        self.result_absolute_position = math.averagePositionListError(results)
        log.info("Result for TooltipAbsolutePosition Calibration: %s with RMS: %s" %
                 (self.result_absolute_position.value(), self.result_absolute_position.getRMS()))
        return self.result_absolute_position.value()


class AbsoluteOrientationCalibrationProcessor(CalibrationProcessor):

    # data extracted from stream
    data_tracker_hip_positions = List()
    data_fwk_hip_positions = List()

    # resulting absolute orientation transform is received from dataflow
    result_absolute_orientation = Value()

    def run(self):
        self.data_tracker_hip_positions = []
        self.data_fwk_hip_positions = []
        record_count = 0
        for record in self.dataset:
            self.data_tracker_hip_positions.append(record.externaltracker_hip_position)
            self.data_fwk_hip_positions.append(record.haptic_pose.translation())
            record_count += 1

        log.info(
            "Absolute Orientation Calibration (%d records selected)" % (record_count,))

        if record_count < 3:
            log.error("Invalid number of correspondences for absolute orientation: %d" % record_count)
            raise ValueError("Invalid number of correspondences for absolute orientation")

        acep = math.PositionList.fromList(self.data_tracker_hip_positions)
        acfp = math.PositionList.fromList(self.data_fwk_hip_positions)

        # XXX should be absoluteOrientationError
        self.result_absolute_orientation = calibration.absoluteOrientation(acfp, acep)

        return self.result_absolute_orientation



class AbsoluteOrientationFWKBaseCalibrationProcessor(CalibrationProcessor):


    # data extracted from stream
    data_tracker_hip_positions = List()
    data_theta1_angles = List()

    # configuration parameters
    negate_upvector = Bool(False)
    joint_lengths = Value(np.array([0.20955, 0.20955]))
    origin_offset = Value(np.array([0., 0., 0.]))

    # parameters from previous steps
    fwkbase_position = Value()
    fwkbase_position2 = Value()

    # resulting absolute orientation transform is received from dataflow
    result_absolute_orientation = Value()

    def run(self):
        self.data_tracker_hip_positions = []
        self.data_theta1_angles = []

        record_count = 0
        for record in self.dataset:
            self.data_tracker_hip_positions.append(record.externaltracker_hip_position)
            self.data_theta1_angles.append(record.jointangles[0])
            record_count += 1

        if record_count == 0:
            log.error("No Records selected for Absolute Orientation Calibration - please redo data-collection and provide valid data.")
            raise ValueError("Invalid number of correspondences for absolute orientation")

        # find a record that is closest to theta1 == 0
        # XXX this could be changed to use the center of the measured range for theta1
        # but this would require the user to cover extremes in both directions (not really an issue)
        idx = np.argmin(np.abs(np.array(self.data_theta1_angles)))
        point_on_zaxis = self.data_tracker_hip_positions[idx]

        # compute up-vector
        if self.negate_upvector:
            up_vector = self.fwkbase_position - self.fwkbase_position2
        else:
            up_vector = self.fwkbase_position2 - self.fwkbase_position

        up_vector /= norm(up_vector)

        # project point onto plane defined by fwkbase_position and fwkbase_position2 to find normalized z_vector
        point_on_zaxis_plane = point_on_zaxis - np.dot(point_on_zaxis - self.fwkbase_position, up_vector) * up_vector
        z_vector = point_on_zaxis_plane - self.fwkbase_position
        z_vector /= norm(z_vector)

        # calculate x_vector
        x_vector = np.cross(up_vector, z_vector)

        # combine into rotation matrix
        rotation_matrix = np.array([x_vector, up_vector, z_vector]).T
        log.info("Calculated rotation ET2HD: %s" % (rotation_matrix,))

        # compute translation using fwkbasepos and joint-lengths
        hd_origin = self.fwkbase_position + self.joint_lengths[0] * z_vector + self.joint_lengths[1] * (-1 * up_vector) - self.origin_offset
        log.info("Calculated translation ET2HD: %s" % (hd_origin,))

        self.result_absolute_orientation = math.Pose(math.Quaternion.fromMatrix(rotation_matrix), hd_origin)
        return self.result_absolute_orientation


class JointAngleCalibrationProcessor(CalibrationProcessor):

    # data extracted from stream
    data_tracker_hip_positions = List()
    data_joint_angles = List()

    # configuration
    optimizationStepSize = Float(1.0)
    optimizationStepFactor = Float(10.0)
    #
    joint_lengths = Value(np.array([0.20955, 0.20955]))
    origin_offset = Value(np.array([0., 0., 0.]))
    #
    # def run(self):
    #     self.data_tracker_hip_positions = [r.hip_reference_pose.translation() for r in self.data]
    #     self.data_joint_angles = [r.jointangles for r in self.data]
    #
    #     hp = math.PositionList.fromList(self.data_tracker_hip_positions)
    #     ja = math.PositionList.fromList(self.data_joint_angles)
    #
    #     result = haptics.computePhantomLMCalibration(ja, hp, self.joint_lengths, self.origin_offset,
    #                                                  self.optimizationStepSize, self.optimizationStepFactor )
    #     return result

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

        self.data_tracker_hip_positions = []
        self.data_joint_angles = []

        record_count = 0
        for record in self.dataset:
            self.data_tracker_hip_positions.append(record.hip_reference_pose.translation())
            self.data_joint_angles.append(record.jointangles)
            record_count += 1

        log.info("Joint-Angles Calibration (%d records selected)" % (record_count,))

        return self.sink_result_jointangle_correction.get(ts).get()


class GimbalAngleCalibrationProcessor(CalibrationProcessor):

    # input data will be set from the controller
    data_joint_angle_correction = Value()

    # data extracted from stream
    data_zrefaxis = List()
    data_joint_angles = List()
    data_gimbal_angles = List()

    # configuration
    optimizationStepSize = Float(1.0)
    optimizationStepFactor = Float(10.0)
    #
    joint_lengths = Value(np.array([0.20955, 0.20955]))
    origin_offset = Value(np.array([0., 0., 0.]))
    #
    #
    # def run(self):
    #     ts = measurement.now()
    #
    #     self.data_zrefaxis = [r.zrefaxis for r in self.data]
    #     self.data_joint_angles = [r.jointangles for r in self.data]
    #     self.data_gimbal_angles = [r.gimbalangles for r in self.data]
    #
    #     zr = math.PositionList.fromList(self.data_zrefaxis)
    #     ja = math.PositionList.fromList(self.data_joint_angles)
    #     ga = math.PositionList.fromList(self.data_gimbal_angles)
    #
    #
    #     result = haptics.computePhantomLMGimbalCalibration(ja, ga, zr, self.data_joint_angle_correction,
    #                                                        self.joint_lengths, self.origin_offset,
    #                                                        self.optimizationStepSize, self.optimizationStepFactor)
    #
    #     return result

    # dataflow components
    sink_result_gimbalangle_correction = Value()
    source_joint_angle_correction = Value()
    source_zrefaxis = Value()
    source_joint_angles = Value()
    source_gimbal_angles = Value()

    @observe("facade")
    def handle_facade_change(self, change):
        facade = change['value']
        if facade is None:
            self.sink_result_gimbalangle_correction = None
            if self.source_joint_angle_correction is not None:
                self.source_joint_angle_correction.setCallback(None)
            self.source_joint_angle_correction = None
            if self.source_zrefaxis is not None:
                self.source_zrefaxis.setCallback(None)
            self.source_zrefaxis = None
            if self.source_joint_angles is not None:
                self.source_joint_angles.setCallback(None)
            self.source_joint_angles = None
            if self.source_gimbal_angles is not None:
                self.source_gimbal_angles.setCallback(None)
            self.source_gimbal_angles = None
        else:
            self.sink_result_gimbalangle_correction = facade.instance.getApplicationPullSinkMatrix3x3("calib_phantom_gimbalangle_correction_out")

            self.source_joint_angle_correction = facade.instance.getApplicationPullSourceMatrix3x3("ga_calib_jointangle_correction")
            self.source_joint_angle_correction.setCallback(self.handler_input_joint_angle_correction)

            self.source_zrefaxis = facade.instance.getApplicationPullSourcePositionList("ga_calib_zrefaxis")
            self.source_zrefaxis.setCallback(self.handler_input_zrefaxis)

            self.source_joint_angles = facade.instance.getApplicationPullSourcePositionList("ga_calib_jointangles")
            self.source_joint_angles.setCallback(self.handler_input_joint_angles)

            self.source_gimbal_angles = facade.instance.getApplicationPullSourcePositionList("ga_calib_gimbalangles")
            self.source_gimbal_angles.setCallback(self.handler_input_gimbal_angles)

    def handler_input_joint_angle_correction(self, ts):
        return measurement.Matrix3x3(ts, self.data_joint_angle_correction)

    def handler_input_zrefaxis(self, ts):
        pl = math.PositionList.fromList(self.data_zrefaxis)
        return measurement.PositionList(ts, pl)

    def handler_input_joint_angles(self, ts):
        pl = math.PositionList.fromList(self.data_joint_angles)
        return measurement.PositionList(ts, pl)

    def handler_input_gimbal_angles(self, ts):
        pl = math.PositionList.fromList(self.data_gimbal_angles)
        return measurement.PositionList(ts, pl)

    def run(self):
        ts = measurement.now()

        self.data_zrefaxis = []
        self.data_joint_angles = []
        self.data_gimbal_angles = []

        record_count = 0
        for record in self.dataset:
            self.data_zrefaxis.append(record.zrefaxis)
            self.data_joint_angles.append(record.jointangles)
            self.data_gimbal_angles.append(record.gimbalangles)
            record_count += 1

        log.info("Gimbal-Angles Calibration (%d records selected)" % (record_count,))

        return self.sink_result_gimbalangle_correction.get(ts).get()


class ReferenceOrientationProcessor(CalibrationProcessor):

    # XXX ISMAR14 debugging ...
    #negate_zaxis = Bool(False)


    # simple line fitting
    def fit_line(self, v1, v2):
        slope, intercept, r_value, p_value, std_err = stats.linregress(v1, v2)
        return slope, intercept

    # fit a circle on the xy-part of the data
    # circle fitting from scipy howto
    def fit_circle(self, x, y):

        x_m = np.mean(x)
        y_m = np.mean(y)

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)


        def f_3(beta, x):
            """ implicit definition of the circle """
            return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 - beta[2]**2

        # initial guess for parameters
        R_m = calc_R(x_m, y_m).mean()
        beta0 = [x_m, y_m, R_m]

        # for implicit function :
        #       data.x contains both coordinates of the points (data.x = [x, y])
        #       data.y is the dimensionality of the response
        lsc_data = odr.Data(np.row_stack([x, y]), y=1)
        lsc_model = odr.Model(f_3, implicit=True)
        lsc_odr = odr.ODR(lsc_data, lsc_model, beta0)
        lsc_out = lsc_odr.run()

        xc_3, yc_3, R_3 = lsc_out.beta
        Ri_3 = calc_R(xc_3, yc_3)
        residu_3 = sum((Ri_3 - R_3)**2)
        return xc_3, yc_3, R_3, residu_3

    def find_center(self, data):
        x_data = data[:, 0]
        y_data = data[:, 1]
        z_data = data[:, 2]

        # fit a line onto the measured points
        xz_slope, xz_intercept = self.fit_line(x_data,z_data)

        # fit a line onto the measured points
        yz_slope, yz_intercept = self.fit_line(y_data,z_data)

        #log.info("distance difference between xz and yz line fitting: %f" % (xz_intercept - yz_intercept))
        log.info("xz slope: %f, yz slope: %f, xz intercept: %f, yz intercept: %f, z_mean: %f" % (xz_slope, yz_slope, xz_intercept, yz_intercept, z_data.mean()))

        # Fit the circle
        xc, yc, radius, residual = self.fit_circle(x_data, y_data)
        log.info("Center of circle: [%f, %f], radius %f, residual: %f" % (xc, yc, radius, residual))

        center = [xc, yc, np.array([xz_intercept,yz_intercept]).mean()]
        log.info("Center: %s" % (center,))
        # return x/z line, y/z line, x/y circle
        return center, ((xz_slope, xz_intercept), (yz_slope, yz_intercept), (xc, yc, radius, residual))


    def find_zaxis(self, corrected_zaxis_points):
        # Calculate the mean of the points, i.e. the 'center' of the cloud
        data = np.asarray(corrected_zaxis_points)
        datamean = data.mean(axis=0)

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data - datamean)

        # Now vv[0] contains the first principal component, i.e. the direction
        # vector of the 'best fit' line in the least squares sense.
        result = np.asarray(vv[0])
        # if result[2] > 0:
        #     result *= -1.0

        return result

    def compute_theta6_correction(self, circle_data, theta6_angles):
        log.info("Start Theta6 correction")
        def rad_norm(angle):
            if angle > np.pi:
                angle -= 2*np.pi
            elif angle <= -np.pi:
                angle += 2*np.pi
            return angle

        rad_norm_vec = np.vectorize(rad_norm)

        def toangle(v):
            x, y = v
            cv, sv = np.arccos(x), np.arcsin(y)
            if y <= 0:
                cv = -cv
            if cv <= -np.pi:
                cv = cv + 2*np.pi
            return cv

        points = circle_data["points"]
        xc = circle_data["xc"]
        yc = circle_data["yc"]
        radius = circle_data["radius"]

        data = (points[:, 0:2] - np.array([xc, yc])) / radius
        data_normalized = (data.T / np.linalg.norm(data, axis=1)).T

        angles = np.apply_along_axis(toangle, 1, data_normalized)

        # find the gap using a histogram
        n, _ = np.histogram(angles, bins=np.linspace(np.pi/180.0-np.pi, np.pi, 361)) # range -179 to +180 degrees
        null_n = (n == 0)

        n_edges = (null_n ^ np.roll(null_n,1))
        n_edges_idx = np.arange(len(n_edges))[n_edges]

        if len(n_edges_idx) != 2:
            nei1 = np.abs(n_edges_idx - np.roll(n_edges_idx, 1)) > 2
            nei2 = np.abs(n_edges_idx - np.roll(n_edges_idx, -1)) > 2
            n_edges_idx = n_edges_idx[nei1 & nei2]
            if len(n_edges_idx) != 2:
                log.warn("Theta6 correction: unexpected gap in dataset, found edges at: %s, will skip" % (n_edges_idx,))
                # needs to find a way to get the "right" gap, or less resolution for the bins ?
                return None

        min_angle = None
        max_angle = None
        for v, i in zip(null_n[n_edges], n_edges_idx):
            if v:
                max_angle = rad_norm(radians(float(i-180)))
            else:
                min_angle = rad_norm(radians(float(i-181)))

        gap_angle = max_angle - min_angle
        if gap_angle < 0:
            center = (max_angle + min_angle) / 2.0
        else:
            center = (max_angle + min_angle) / 2.0 - np.pi

        theta6_center = rad_norm(center + np.pi)

        # compensate time-delays
        theta6_ref = rad_norm_vec(-angles + theta6_center)

        ta = theta6_angles.copy()
        tr = theta6_ref.copy()

        ta -= ta.mean()
        ta /= ta.std()
        tr -= tr.mean()
        tr /= tr.std()

        N = len(theta6_ref)
        dt = np.arange(1-N, N)

        cross_correlation = correlate(ta, tr)
        orn_delay = dt[cross_correlation.argmax()]

        if orn_delay > 0:
            theta6_ref = theta6_ref[:-orn_delay]
            theta6_angles = theta6_angles[orn_delay:]
        elif orn_delay < 0:
            theta6_ref = theta6_ref[orn_delay:]
            theta6_angles = theta6_angles[:-orn_delay]

        # 2nd order variant
        # p0 = [0.0, 1.0, 0.0]
        # O6_fitfunc = lambda p, x: rad_norm_vec(p[0]*x**2+p[1]*x+p[2])
        # O6_errfunc = lambda p, x, y: rad_norm_vec(O6_fitfunc(p, x) - rad_norm_vec(y))

        # minimize the error using least-squares optimization
        p0 = [1.0, 0.0]
        O6_fitfunc = lambda p, x: rad_norm_vec(p[0]*x+p[1])
        O6_errfunc = lambda p, x, y: rad_norm_vec(O6_fitfunc(p, x) - rad_norm_vec(y))

        p1, cov, info, msg, success = optimize.leastsq(O6_errfunc, p0[:], args=(theta6_angles, theta6_ref), full_output=True)
        theta6_correction = [0.0, p1[0], p1[1]]
        log.info("Theta6-Correction result: %s" % (theta6_correction,))
        return theta6_correction

    def run(self, use_markers=True):
        # haptic interface point as origin in the correct coordinate system
        zaxis_points = [np.array([0.0, 0.0, 0.0])]

        # bulk load here, since we need to iterate multiple times over the dataset
        data = list(self.dataset)

        log.info("Reference Orientation (%d records selected)" % (len(data),))


        # the center of the circle described by the center-of-mass of the tracking target
        target_position = np.asarray([d.target_position for d in data])

        # center info consists of: ((xz_slope, xz_intercept), (yz_slope, yz_intercept), (xc, yc, radius, residual))
        target_circle_center, target_center_info = self.find_center(target_position)
        zaxis_points.append(target_circle_center)

        # theta6 fitting helpers
        theta6_angles = np.asarray([d.gimbalangles[2] for d in data])
        best_residual_radius = target_center_info[2][2] / target_center_info[2][3]
        theta6_data = dict(points=target_position, radius=target_center_info[2][2], residual=target_center_info[2][3],
                           xc=target_circle_center[0], yc=target_circle_center[1])


        if use_markers:
            num_markers = len(data[0].target_markers)
            target_markers = np.asarray([d.target_markers for d in data])
            # individual markers travel on circles as well - find their centers
            for i in range(num_markers):
                m_circle_center, m_center_info = self.find_center(target_markers[:, i, :])
                zaxis_points.append(m_circle_center)

                residual_radius = m_center_info[2][2] / m_center_info[2][3]
                if residual_radius > best_residual_radius:
                    best_residual_radius = residual_radius
                    theta6_data = dict(points=target_markers[:, i, :], radius=m_center_info[2][2], residual=m_center_info[2][3],
                                       xc=m_circle_center[0], yc=m_circle_center[1])

        # zaxis = self.find_zaxis(zaxis_points)

        # project zaxis back into OTtarget coordinates
        corrected_zaxis_points_ot = []

        for record in data:
            hiptarget_pose_inv = record.hiptarget_pose.invert()
            # un-project markers using the corrected stylus pose (5dof)
            # corrected_zaxis_ot.append(hiptarget_pose_inv * (record.device_to_stylus_5dof * zaxis))

            # un-project the found centers for debugging
            otp = []
            for i, p in enumerate(zaxis_points):
                otp.append(hiptarget_pose_inv * (record.device_to_stylus_5dof * p))
            corrected_zaxis_points_ot.append(otp)

        corrected_zaxis_points_ot = np.asarray(corrected_zaxis_points_ot)
        corrected_zaxis_points_ot_mean = []
        for i in range(corrected_zaxis_points_ot.shape[1]):
            corrected_zaxis_points_ot_mean.append(corrected_zaxis_points_ot[:, i, :].mean(axis=0))

        zref = np.asarray(self.find_zaxis(corrected_zaxis_points_ot_mean))
        zref = zref / np.linalg.norm(zref)

        # XXX ismar14 debugging
        #sign = -1. if self.negate_zaxis else 1.


        # compute corrections for theta6 here since data is all available
        theta6_correction = self.compute_theta6_correction(theta6_data, theta6_angles)


        return zref, corrected_zaxis_points_ot_mean, theta6_correction


class TimeDelayEstimationCalibrationProcessor(CalibrationProcessor):

    # data extracted from stream
    data_tracker_hip_distance_to_origin = List()
    data_haptic_device_distance_to_origin = List()

    def run(self):
        interval = self.dataset.interval

        len_dataset = 0
        start_time = 0.
        for record in self.dataset:
            self.data_tracker_hip_distance_to_origin.append(norm(record.hip_reference_pose.translation()))
            self.data_haptic_device_distance_to_origin.append(norm(record.haptic_pose.translation()))
            if len_dataset == 0:
                start_time = record.timestamp
            len_dataset += 1

        stop_time = record.timestamp
        nr_samples = int(len_dataset * interval)

        x = np.linspace(0, len_dataset, num=len_dataset)

        et_distance_interpolator = interpolate.interp1d(x, np.asarray(self.data_tracker_hip_distance_to_origin))
        hd_distance_interpolator = interpolate.interp1d(x, np.asarray(self.data_haptic_device_distance_to_origin))

        xs = np.linspace(0, len_dataset, num=nr_samples)

        a, b, N = 0., float(stop_time-start_time), nr_samples
        dt = np.arange(1-N, N)

        log.info("Correlating dataset for timedelay estimation")
        cross_correlation = correlate(et_distance_interpolator(xs),
                                      hd_distance_interpolator(xs))
        shift_calculated = (dt[cross_correlation.argmax()] * 1.0 * b / float(N)) / interval

        log.info("Timedelay between external_tracker and haptic device: %f, interval: %f" % (shift_calculated, interval))

        return shift_calculated

# offline calibration controller implementation below


class ProcessData(Atom):
    dataset = Typed(DataSet)
    attributes = Dict()
    results = Dict()


class OfflineCalibrationResults(Atom):

    has_result = Bool(False)

    # calibration results
    theta6_correction_result = Value(np.array([0, 1, 0]))
    zaxis_reference_result = Value(np.array([0, 0, 1]))
    zaxis_points_result = Value([])

    tooltip_calibration_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    fwkbase_position_calibration_result = Value(np.array([0, 0, 0]))
    fwkbase_position2_calibration_result = Value(np.array([0, 1, 0]))
    absolute_orientation_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    jointangles_correction_result = Value(np.array(angle_null_correction))
    gimbalangles_correction_result = Value(np.array(angle_null_correction))

    timedelay_estimation_result = Float(0.0)

    # results evaluation
    position_errors = List()
    orientation_errors = List()

    process_data = Dict()

    def reset(self):
        self.has_result = False

        self.theta6_correction_result = np.array([0, 1, 0])
        self.zaxis_reference_result = np.array([0, 0, 1])
        self.zaxis_points_result = []

        self.tooltip_calibration_result = math.Pose(math.Quaternion(), np.array([0, 0, 0]))
        self.fwkbase_position_calibration_result = np.array([0, 0, 0])
        self.fwkbase_position2_calibration_result = np.array([0, 0, 0])
        self.absolute_orientation_result = math.Pose(math.Quaternion(), np.array([0, 0, 0]))
        self.jointangles_correction_result = np.array(angle_null_correction)
        self.gimbalangles_correction_result = np.array(angle_null_correction)

        self.timedelay_estimation_result = 0.0

        self.position_errors = []
        self.orientation_errors = []

        self.process_data = {}


class OfflineCalibrationParameters(Atom):
    # global
    stream_skip_first_nseconds = Float(0.0)
    haptidevice_name = Str()

    # tooltip
    tooltip_enabled = Bool(False)
    tooltip_datasource = Str()
    tt_minimal_angle_between_measurements = Float(0.1)
    tt_use_pose = Bool(False)

    # fwkbase_position
    fwkbase_position_enabled = Bool(False)
    fwkbase_position_datasource = Str()

    # fwkbase_position2
    fwkbase_position2_enabled = Bool(False)
    fwkbase_position2_datasource = Str()

    # absolute orientation
    absolute_orientation_enabled = Bool(False)
    absolute_orientation_datasource = Str()

    # known methods: fwkpose, externaltracker
    ao_method = Enum('fwkpose', 'fwkbase')
    # parameters for using externaltracker
    ao_negate_upvector = Bool(False)

    # custom initialization for angle correction
    ao_initialize_anglecorrection_calibsource = Value(None)

    # parameters for using fwkpose
    ao_inital_maxdistance_from_origin = Float(0.03)
    ao_minimal_distance_between_measurements = Float(0.01)
    ao_refinement_expand_coverage = Float(1.2)
    ao_refinement_shrink_distance = Float(0.8)
    ao_number_of_clusters = Int(0)

    # joint-angle correction
    joint_angle_calibration_enabled = Bool(False)
    joint_angle_calibration_datasource = Str()
    ja_minimal_distance_between_measurements = Float(0.005)
    ja_maximum_distance_to_reference = Float(0.02)
    ja_refinement_min_difference = Float(0.00001)
    ja_refinement_max_iterations = Int(3)
    ja_refinement_shrink_distance = Float(0.8)
    ja_number_of_clusters = Int(0)
    ja_use_2nd_order = Bool(False)
    ja_exclude_calibration_samples_from_evaluation = Bool(True)

    # reference orientation
    reference_orientation_enabled = Bool(False)
    reference_orientation_datasource = Str()
    ro_minimal_angle_between_measurements = Float(0.1)

    # gimbal-angle correction
    gimbal_angle_calibration_enabled = Bool(False)
    gimbal_angle_calibration_datasource = Str()
    ga_minimal_angle_between_measurements = Float(0.1)
    ga_use_tooltip_offset = Bool(False)
    ga_number_of_clusters = Int(0)
    ga_use_2nd_order = Bool(False)
    ga_exclude_calibration_samples_from_evaluation = Bool(True)

    # time-delay estimation
    timedelay_estimation_enabled = Bool(False)
    timedelay_estimation_datasource = Str()

    # result evaluation
    result_evaluation_enabled = Bool(False)
    result_evaluation_datasource = Str()

    # haptic device
    joint_lengths = Value(np.array([0.13335, 0.13335]))
    origin_offset = Value(np.array([0.0, -0.11, -0.035]))


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

    absolute_orientation_inv = absolute_orientation.invert()
    position_errors = []
    timestamps = []
    for record in stream:
        timestamps.append(record.timestamp)
        gimbalangles = getattr(record, 'gimbalangles', np.array([0., 0., 0.]))
        haptic_pose = forward_kinematics.calculate_pose(record.jointangles, gimbalangles, record=record)
        hip_reference_pose = (absolute_orientation_inv * record.externaltracker_pose * tooltip_offset)
        position_errors.append(norm(hip_reference_pose.translation() - haptic_pose.translation()))

    return pd.Series(np.array(position_errors), index=timestamps)


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

    absolute_orientation_inv = absolute_orientation.invert()
    orientation_errors = []

    timestamps = []
    for record in stream:
        timestamps.append(record.timestamp)
        haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles, record=record)
        hip_reference_pose = (absolute_orientation_inv * record.externaltracker_pose * tooltip_offset)

        z_fwk = math.Quaternion(haptic_pose.rotation()).transformVector(zaxis)
        z_ref = math.Quaternion(hip_reference_pose.rotation()).transformVector(zref_axis)

        # unit vector
        z_fwk /= norm(z_fwk)
        z_ref /= norm(z_ref)

        orientation_errors.append(degrees(acos(z_ref.dot(z_fwk))))

    return pd.Series(np.array(orientation_errors), index=timestamps)


class OfflineCalibrationProcessor(Atom):

    context = Value()
    config = Value()
    facade = Value()
    dfg_dir = Value()
    dfg_filename = Value()
    publish_results = Bool(True)

    is_working = Bool(False)

    parameters = Typed(OfflineCalibrationParameters)
    result = Typed(OfflineCalibrationResults)

    datasources = Dict()
    calibsources = Dict()

    # refinement vars
    ao_maxdistance_from_origin = Float(0.0)
    ao_minimal_distance_between_measurements = Float(0.0)
    ao_number_of_clusters = Int(0)

    ja_minimal_distance_between_measurements = Float(0.0)
    ja_number_of_clusters = Int(0)

    ga_number_of_clusters = Int(0)

    source_tooltip_calibration_result = Value()
    source_absolute_orientation_result = Value()
    source_jointangles_correction_result = Value()
    source_gimbalangles_correction_result = Value()

    source_zaxis_points_result = Value()
    source_zaxis_reference_result = Value()

    #helpers
    fwk_classes = Value()

    def _default_result(self):
        return OfflineCalibrationResults()

    def _default_fwk_classes(self):
        config = self.context.get("config")
        section = "ubitrack.devices.%s" % self.parameters.hapticdevice_name
        if config.has_section(section):
            hd_cfg = config.items(section)
            model_family = hd_cfg.get("model_family", "phantom")
            model_type = hd_cfg.get("model_type", "omni")

            if model_family == 'phantom':
                return FWKinematicPhantom, FWKinematicPhantom2
            elif model_family == 'virtuose':
                # XXX no second order models yet (are they needed ??)
                if model_type == 'virtuose':
                    return FWKinematicVirtuose, None
                elif model_type == 'scale1':
                    return FWKinematicScale, None
                else:
                    raise ValueError("Invalid model_type %s for virtuose familiy" % model_type)
            else:
                raise ValueError("Invalid model_family: %s" % model_family)
        raise ValueError("Missing Configuration for Haptic Device.")


    def do_tooltip_calibration(self, tt_data):
        log.info("Tooltip Calibration")

        stream_filters = []

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        if self.parameters.tt_minimal_angle_between_measurements > 0:
            tt_selector = RelativeOrienationDistanceStreamFilter("externaltracker_pose",
                                                                 min_distance=self.parameters.tt_minimal_angle_between_measurements)
            stream_filters.append(tt_selector)

        ds = DataSet(name="tooltip_calibration_data",
                     title="Tooltip Calibration DataSet",
                     recordsource=tt_data,
                     processor_factory=TooltipStreamProcessor,
                     stream_filters=stream_filters,
                     )

        tt_processor = TooltipCalibrationProcessor(dataset=ds,
                                                   use_tooltip_pose=self.parameters.tt_use_pose)
        self.result.tooltip_calibration_result = tt_processor.run()
        log.info("Result for Tooltip Calibration: %s" % str(self.result.tooltip_calibration_result))

        pd = self.result.process_data.setdefault('tooltip_calibration', [])
        pd.append(ProcessData(dataset=ds,
                              results=dict(tooltip_calibration=self.result.tooltip_calibration_result)))

        return True

    def do_fwkbase_position_calibration(self, tt_data):
        log.info("FWKBase Position Calibration")

        stream_filters = []

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        ds = DataSet(name="fwkbase_position_calibration_data",
                     title="FWKBase Position Calibration DataSet",
                     recordsource=tt_data,
                     processor_factory=TooltipStreamProcessor,
                     stream_filters=stream_filters,
                     )

        tt_processor = TooltipAbsolutePositionCalibrationProcessor(dataset=ds)
        self.result.fwkbase_position_calibration_result = tt_processor.run()

        log.info("Result for FWKBase Position Calibration: %s" % str(self.result.fwkbase_position_calibration_result))

        pd = self.result.process_data.setdefault('fwkbase_position', [])
        pd.append(ProcessData(dataset=ds,
                              results=dict(fwkbase_position=self.result.fwkbase_position_calibration_result)))

        return True

    def do_fwkbase_position2_calibration(self, tt_data):
        log.info("FWKBase Position2 Calibration")

        stream_filters = []

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        ds = DataSet(name="fwkbase_position_calibration_data",
                     title="FWKBase Position Calibration DataSet",
                     recordsource=tt_data,
                     processor_factory=TooltipStreamProcessor,
                     stream_filters=stream_filters,
                     )

        tt_processor = TooltipAbsolutePositionCalibrationProcessor(dataset=ds)
        self.result.fwkbase_position2_calibration_result = tt_processor.run()

        log.info("Result for FWKBase Position2 Calibration: %s" % str(self.result.fwkbase_position2_calibration_result))

        pd = self.result.process_data.setdefault('fwkbase_position2', [])
        pd.append(ProcessData(dataset=ds,
                              results=dict(fwkbase_position2=self.result.fwkbase_position2_calibration_result)))

        return True

    def do_absolute_orientation(self, ao_data, ao_method='fwkpose'):
        log.info("Absolute Orientation: %s" % ao_method)

        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result,
                           enable_2ndorder=self.parameters.ja_use_2nd_order)

        ao_streamproc_factory = AbsoluteOrientationStreamProcessor
        ao_streamproc_attributes = dict(tooltip_offset=self.result.tooltip_calibration_result,
                                        forward_kinematics=fwk)

        ao_processor_attributes = dict()
        stream_filters = []

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        if ao_method == "fwkpose":
            ao_processor_factory = AbsoluteOrientationCalibrationProcessor

            # no attributes
            if self.ao_maxdistance_from_origin > 0:
                ao_selector1 = StaticPointDistanceStreamFilter("haptic_pose", np.array([0, 0, 0]),
                                                               max_distance=self.ao_maxdistance_from_origin)
                stream_filters.append(ao_selector1)
            if self.ao_minimal_distance_between_measurements > 0:
                ao_selector2 = RelativePointDistanceStreamFilter("haptic_pose",
                                                                 min_distance=self.ao_minimal_distance_between_measurements)
                stream_filters.append(ao_selector2)

            if self.ao_number_of_clusters > 0:
                ao_selector3 = NClustersPositionStreamFilter('haptic_pose', self.ao_number_of_clusters)
                stream_filters.append(ao_selector3)

        elif ao_method == "fwkbase":
            ao_processor_factory = AbsoluteOrientationFWKBaseCalibrationProcessor

            ao_processor_attributes['fwkbase_position'] = self.result.fwkbase_position_calibration_result
            ao_processor_attributes['fwkbase_position2'] = self.result.fwkbase_position2_calibration_result
            ao_processor_attributes['negate_upvector'] = self.parameters.ao_negate_upvector
            ao_processor_attributes['joint_lengths'] = self.parameters.joint_lengths
            ao_processor_attributes['origin_offset'] = self.parameters.origin_offset

            # no stream filters

        else:
            raise ValueError("Invalid method for Absolute Orientation: %s" % ao_method)

        ds = DataSet(name="absolute_orientation_calibration_data",
                     title="Absolute Orientation Calibration DataSet",
                     recordsource=ao_data,
                     processor_factory=ao_streamproc_factory,
                     attributes=ao_streamproc_attributes,
                     stream_filters=stream_filters,
                     )

        ao_processor = ao_processor_factory(dataset=ds, **ao_processor_attributes)

        self.result.absolute_orientation_result = ao_processor.run()

        log.info("Result for Absolute Orientation: %s" % str(self.result.absolute_orientation_result))

        pd = self.result.process_data.setdefault('absolute_orientation', [])
        pd.append(ProcessData(dataset=ds,
                             attributes=ao_processor_attributes,
                             results=dict(absolute_orientation=self.result.absolute_orientation_result)))

        return True

    def do_jointangle_correction(self, ja_data):
        log.info("Joint-Angle Correction")

        fwk = self.get_fwk(angle_null_correction, angle_null_correction,
                           enable_2ndorder=self.parameters.ja_use_2nd_order)
        ja_streamproc_attributes = dict(tooltip_offset=self.result.tooltip_calibration_result,
                                        absolute_orientation=self.result.absolute_orientation_result,
                                        forward_kinematics=fwk)

        stream_filters = []

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        if self.parameters.ja_maximum_distance_to_reference > 0:
            # simple way to avoid outliers from the external tracker: limit distance to reference ...
            ja_selector1 = TwoPointDistanceStreamFilter("hip_reference_pose", "haptic_pose",
                                                        max_distance=self.parameters.ja_maximum_distance_to_reference)
            stream_filters.append(ja_selector1)

        if self.ja_minimal_distance_between_measurements > 0:
            # only use a subset of the dataset
            ja_selector2 = RelativePointDistanceStreamFilter("haptic_pose",
                                                             min_distance=self.ja_minimal_distance_between_measurements)
            stream_filters.append(ja_selector2)

        if self.ja_number_of_clusters > 0:
            ja_selector3 = NClustersPositionStreamFilter('haptic_pose', self.ja_number_of_clusters)
            stream_filters.append(ja_selector3)

        ds = DataSet(name="joint_angles_calibration_data",
                     title="Joint Angles Calibration DataSet",
                     recordsource=ja_data,
                     processor_factory=JointAngleCalibrationStreamProcessor,
                     attributes=ja_streamproc_attributes,
                     stream_filters=stream_filters,
                     )

        ja_processor_attributes = dict(joint_lengths=self.parameters.joint_lengths,
                                       origin_offset=self.parameters.origin_offset,)
        ja_processor = JointAngleCalibrationProcessor(dataset=ds,
                                                      **ja_processor_attributes)

        ja_processor.facade = self.facade
        self.result.jointangles_correction_result = ja_processor.run()
        ja_processor.facade = None

        log.info("Result for Joint-Angles Correction: %s" % str(self.result.jointangles_correction_result))

        pd = self.result.process_data.setdefault('jointangles_correction', [])
        pd.append(ProcessData(dataset=ds,
                              attributes=ja_processor_attributes,
                              results=dict(jointangles_correction=self.result.jointangles_correction_result)))

        return True

    def do_gimbalangle_correction(self, ga_data):
        log.info("Gimbal-Angle Correction")
        par = self.parameters

        fwk = self.get_fwk(self.result.jointangles_correction_result, angle_null_correction,
                           enable_2ndorder=self.parameters.ga_use_2nd_order)

        ga_streamproc_attributes = dict(absolute_orientation=self.result.absolute_orientation_result,
                                        tooltip_offset=self.result.tooltip_calibration_result if par.ga_use_tooltip_offset else tooltip_null_calibration,
                                        forward_kinematics=fwk,
                                        zrefaxis_calib=self.result.zaxis_reference_result)

        stream_filters = []

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        if par.ga_minimal_angle_between_measurements > 0:
            ga_selector1 = RelativeOrienationDistanceStreamFilter("haptic_pose",
                                                                  min_distance=par.ga_minimal_angle_between_measurements)

            stream_filters.append(ga_selector1)

        if self.ga_number_of_clusters > 0:
            ga_selector2 = NClustersOrientationStreamFilter('haptic_pose', self.ga_number_of_clusters)
            stream_filters.append(ga_selector2)

        ds = DataSet(name="gimbal_angles_calibration_data",
                     title="Gimbal Angles Calibration DataSet",
                     recordsource=ga_data,
                     processor_factory=GimbalAngleCalibrationStreamProcessor,
                     attributes=ga_streamproc_attributes,
                     stream_filters=stream_filters,
                     )

        ga_processor_attributes = dict(data_joint_angle_correction=self.result.jointangles_correction_result,
                                       joint_lengths=par.joint_lengths,
                                       origin_offset=par.origin_offset,)
        ga_processor = GimbalAngleCalibrationProcessor(dataset=ds,
                                                       **ga_processor_attributes)

        ga_processor.facade = self.facade
        gimbalangle_correction = ga_processor.run()

        # add theta6 correction here
        gimbalangle_correction[2, 0] = self.result.theta6_correction_result[0]
        gimbalangle_correction[2, 1] = self.result.theta6_correction_result[1]
        gimbalangle_correction[2, 2] = self.result.theta6_correction_result[2]

        self.result.gimbalangles_correction_result = gimbalangle_correction

        ga_processor.facade = None
        log.info("Result for Gimbal-Angles Correction: %s" % str(self.result.gimbalangles_correction_result))

        pd = self.result.process_data.setdefault('gimbalangles_correction', [])
        pd.append(ProcessData(dataset=ds,
                             attributes=ga_processor_attributes,
                             results=dict(gimbalangles_correction=self.result.gimbalangles_correction_result)))

        return True

    def do_reference_orientation(self, ro_data):
        log.info("Calculate Reference Orientation")

        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result,
                           enable_2ndorder=self.parameters.ga_use_2nd_order)
        fwk_5dof = self.get_fwk(self.result.jointangles_correction_result, angle_null_correction, disable_theta6=True,
                                enable_2ndorder=self.parameters.ga_use_2nd_order)

        ro_streamproc_attributes = dict(tooltip_offset=self.result.tooltip_calibration_result,
                                        absolute_orientation=self.result.absolute_orientation_result,
                                        forward_kinematics=fwk,
                                        forward_kinematics_5dof=fwk_5dof)

        stream_filters = []

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        ds = DataSet(name="reference_orientation_calibration_data",
                     title="Reference Orientation Calibration DataSet",
                     recordsource=ro_data,
                     processor_factory=ReferenceOrientationStreamProcessor,
                     attributes=ro_streamproc_attributes,
                     stream_filters=stream_filters,
                     )

        ro_processor = ReferenceOrientationProcessor(dataset=ds)

        ro_processor.facade = self.facade

        zaxis_reference, zaxis_points, theta6_correction = ro_processor.run(use_markers=True)
        self.result.zaxis_reference_result = zaxis_reference
        self.result.zaxis_points_result = zaxis_points

        if theta6_correction is not None:
            self.result.theta6_correction_result = theta6_correction

        ro_processor.facade = None
        log.info("Result for ReferenceOrientation: %s" % str(self.result.zaxis_reference_result))
        log.info("Result for Theta6-Correction: %s" % str(self.result.theta6_correction_result))

        pd = self.result.process_data.setdefault('reference_orientation', [])
        pd.append(ProcessData(dataset=ds,
                              results=dict(zaxis_reference=self.result.zaxis_reference_result,
                                           zaxis_points=self.result.zaxis_points_result,
                                           theta6_correction=self.result.theta6_correction_result)))

        return True

    def do_timedelay_estimation(self, ja_data):
        if ja_data is None:
            log.warn("No data for time-delay estimation.")
            return False

        log.info("Time-Delay Estimation")

        fwk = self.get_fwk(self.result.jointangles_correction_result, angle_null_correction)
        ja_streamproc_attributes = dict(tooltip_offset=self.result.tooltip_calibration_result,
                                        absolute_orientation=self.result.absolute_orientation_result,
                                        forward_kinematics=fwk)

        stream_filters = []
        # XXX eventually filter time-slice with sufficient movement

        if self.parameters.stream_skip_first_nseconds > 0:
            sf_selector = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds)
            stream_filters.append(sf_selector)

        ds = DataSet(name="timedelay_estimation_calibration_data",
                     title="Time-Delay Estimation Calibration DataSet",
                     recordsource=ja_data,
                     processor_factory=JointAngleCalibrationStreamProcessor,
                     attributes=ja_streamproc_attributes,
                     stream_filters=stream_filters,
                     )

        tde_processor_attributes = dict()
        tde_processor = TimeDelayEstimationCalibrationProcessor(dataset=ds,
                                                                **tde_processor_attributes)

        tde_processor.facade = self.facade
        self.result.timedelay_estimation_result = tde_processor.run()
        tde_processor.facade = None

        log.info("Result for Time-Delay Estimation: %s" % str(self.result.timedelay_estimation_result))

        pd = self.result.process_data.setdefault('timedelay_estimation', [])
        pd.append(ProcessData(dataset=ds,
                              attributes=tde_processor_attributes,
                              results=dict(timedelay_estimation=self.result.timedelay_estimation_result)))

        return True

    def compute_position_errors(self, ja_data, excluded_timestamps=None):
        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result,
                           enable_2ndorder=self.parameters.ja_use_2nd_order)

        if self.parameters.stream_skip_first_nseconds > 0:
            ja_data = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds).process(ja_data)

        if excluded_timestamps is None:
            ja_pds = self.result.process_data.get('jointangles_correction', [])
            if ja_pds:
                ja_pd = ja_pds[-1]
                excluded_timestamps = [r.timestamp for r in ja_pd.dataset]

        if excluded_timestamps is not None and self.parameters.ja_exclude_calibration_samples_from_evaluation:
            ja_data = ExcludeTimestampsStreamFilter(excluded_timestamps).process(ja_data)

        position_errors = compute_position_errors(ja_data,
                                                  tooltip_offset=self.result.tooltip_calibration_result,
                                                  absolute_orientation=self.result.absolute_orientation_result,
                                                  forward_kinematics=fwk)
        log.info("Resulting position error: %s" % position_errors.mean())
        return position_errors

    def compute_orientation_errors(self, ga_data, excluded_timestamps=None):
        fwk = self.get_fwk(self.result.jointangles_correction_result, self.result.gimbalangles_correction_result,
                           enable_2ndorder=self.parameters.ga_use_2nd_order)

        if self.parameters.stream_skip_first_nseconds > 0:
            ga_data = SkipFrontStreamFilter(self.parameters.stream_skip_first_nseconds).process(ga_data)

        if excluded_timestamps is None:
            ga_pds = self.result.process_data.get('gimbalangles_correction', [])
            if ga_pds:
                ga_pd = ga_pds[-1]
                excluded_timestamps = [r.timestamp for r in ga_pd.dataset]

        if excluded_timestamps is not None and self.parameters.ga_exclude_calibration_samples_from_evaluation:
            ga_data = ExcludeTimestampsStreamFilter(excluded_timestamps).process(ga_data)
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
        self.ao_number_of_clusters = self.parameters.ao_number_of_clusters
        self.ja_number_of_clusters = self.parameters.ja_number_of_clusters
        self.ga_number_of_clusters = self.parameters.ga_number_of_clusters

    def process(self):
        fname = os.path.join(self.dfg_dir, self.dfg_filename)
        if not os.path.isfile(fname):
            log.error("DFG file not found: %s" % fname)
            return

        self.facade.loadDataflow(fname)
        self.facade.startDataflow()

        if self.publish_results:
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
        datasources = self.datasources

        if self.parameters.tooltip_enabled:
            # 1st step: Tooltip Calibration (uses step01 data)
            self.do_tooltip_calibration(datasources.get(self.parameters.tooltip_datasource, None))
        else:
            # skipped tooltip calibration, defaults to no offset
            self.result.tooltip_calibration_result = tooltip_null_calibration

        if self.parameters.fwkbase_position_enabled:
            # 1st step: FwKBase Position Calibration (uses fwkbase_position data)
            self.do_fwkbase_position_calibration(datasources.get(self.parameters.fwkbase_position_datasource, None))
        else:
            # skipped fwkbase_position calibration, defaults to no offset
            self.result.fwkbase_position_calibration_result = fwkbase_position_null_calibration

        if self.parameters.fwkbase_position2_enabled:
            # 1st step: FwKBase Position2 Calibration (uses fwkbase_position data)
            self.do_fwkbase_position2_calibration(datasources.get(self.parameters.fwkbase_position2_datasource, None))
        else:
            # skipped fkwbase_position2 calibration, defaults to no offset
            self.result.fwkbase_position2_calibration_result = fwkbase_position_null_calibration

        # 2nd step: initial absolute orientation (uses step03  data)
        if self.parameters.absolute_orientation_enabled:
            if self.parameters.ao_method == 'fwkbase' and not (self.parameters.fwkbase_position_enabled and self.parameters.fwkbase_position2_enabled):
                raise ValueError("FWKBase Position calibration must be enabled for Absolute Orientation fwkbase method.")
            else:
                # allow for custom initialization of joint/gimbal angles correction
                if self.parameters.ao_initialize_anglecorrection_calibsource:
                    self.load_defaults_from_calibsource(self.parameters.ao_initialize_anglecorrection_calibsource)

            if not self.do_absolute_orientation(datasources.get(self.parameters.absolute_orientation_datasource, None),
                                                ao_method=self.parameters.ao_method):
                return
        else:
            log.warn("Absolute Orientation Calibration is disabled - Are you sure this is correct ????")
            self.result.absolute_orientation_result = absolute_orientation_null_calibration

        # compute initial errors
        self.result.position_errors.append(self.compute_position_errors(datasources.get(self.parameters.joint_angle_calibration_datasource, None)))

        last_error = pd.Series(np.array([0., ]))

        # initial time-delay estimation
        if self.parameters.timedelay_estimation_enabled:
            self.do_timedelay_estimation(datasources.get(self.parameters.timedelay_estimation_datasource, None))

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
        retry_count = 0
        while iterative_refinement_enabled:
            # modify the frame selector parameters
            self.ao_maxdistance_from_origin *= self.parameters.ao_refinement_expand_coverage
            self.ao_minimal_distance_between_measurements *= self.parameters.ao_refinement_shrink_distance
            self.ja_minimal_distance_between_measurements *= self.parameters.ja_refinement_shrink_distance

            # redo the calibration
            if self.parameters.absolute_orientation_enabled:
                if not self.do_absolute_orientation(datasources.get(self.parameters.absolute_orientation_datasource, None)):
                    break

            # XXX it only makes sense to optimize the joint angles if the absolute orientation was updated as well.
            if self.parameters.joint_angle_calibration_enabled:
                self.do_jointangle_correction(datasources.get(self.parameters.joint_angle_calibration_datasource, None))

                # recalculate the error
                error = self.compute_position_errors(datasources.get(self.parameters.joint_angle_calibration_datasource, None))

                position_error_diff = (last_error.mean() - error.mean())
                if position_error_diff < self.parameters.ja_refinement_min_difference:
                    if position_error_diff >= 0:
                        self.result.position_errors.append(error)
                        break
                    else:
                        log.warn("Optimization yielded bad result - retrying again ..")
                        # Bad iteration - restore previous result
                        if len(self.result.process_data['absolute_orientation']) > 2 and len(self.result.process_data['jointangles_correction']) > 2:
                            # discard latest entries in process data
                            _ = self.result.process_data['absolute_orientation'].pop(-1)
                            _ = self.result.process_data['jointangles_correction'].pop(-1)

                            # restore previous results
                            self.result.absolute_orientation_result = self.result.process_data['absolute_orientation'][-1].results['absolute_orientation']
                            self.result.jointangles_correction_result = self.result.process_data['jointangles_correction'][-1].results['jointangles_correction']
                            retry_count += 1

                else:
                    self.result.position_errors.append(error)
                    last_error = error
                    retry_count = 0

            if retry_count >= 3:
                log.warn("Retried %d times without improvement - giving up" % retry_count)
                break

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

        # final time-delay estimation
        if self.parameters.timedelay_estimation_enabled:
            self.do_timedelay_estimation(datasources.get(self.parameters.timedelay_estimation_datasource, None))

        if self.publish_results:
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

    # XXX Needs refactoring !!!
    def load_datasource(self, config, datasource_sname):
        log.info("Load Datasource: %s" % datasource_sname)
        ds_cfg = dict(config.items(datasource_sname))
        data_directory = ds_cfg["data_directory"]
        reference_column = ds_cfg["reference"]
        columns = [(k.replace("item.", ""), [si.strip() for si in v.split(",")]) for k, v in ds_cfg.items() if k.startswith("item.")]

        fields = []

        for k, v in columns:
            filename, datatype, selector = v
            is_array = False
            # hack until we refator configuration system
            if datatype.endswith('-list'):
                is_array = True
                datatype = datatype.replace('-list', '')

            fs = StreamFileSpec(fieldname=k,
                                filename=os.path.join(data_directory, filename).strip(),
                                datatype=datatype.lower(),
                                is_array=is_array,
                                )
            f = StreamInterpolator(filespec=fs,
                                  is_reference=bool(k == reference_column),
                                  selector=selector.lower(),
                                  latency=0.0)
            fields.append(f)

        return RecordSource(name=datasource_sname,
                            title=datasource_sname,
                            fields=fields)

    # XXX Needs refactoring !!!
    def _default_datasources(self):
        all_datasources = set()
        if self.parameters.tooltip_enabled:
            all_datasources.add(self.parameters.tooltip_datasource)
        if self.parameters.fwkbase_position_enabled:
            all_datasources.add(self.parameters.fwkbase_position_datasource)
        if self.parameters.fwkbase_position2_enabled:
            all_datasources.add(self.parameters.fwkbase_position2_datasource)
        if self.parameters.absolute_orientation_enabled:
            all_datasources.add(self.parameters.absolute_orientation_datasource)
        if self.parameters.joint_angle_calibration_enabled:
            all_datasources.add(self.parameters.joint_angle_calibration_datasource)
        if self.parameters.reference_orientation_enabled:
            all_datasources.add(self.parameters.reference_orientation_datasource)
        if self.parameters.gimbal_angle_calibration_enabled:
            all_datasources.add(self.parameters.gimbal_angle_calibration_datasource)
        if self.parameters.timedelay_estimation_enabled:
            all_datasources.add(self.parameters.timedelay_estimation_datasource)
        if self.parameters.result_evaluation_enabled:
            all_datasources.add(self.parameters.result_evaluation_datasource)

        config = self.context.get("config")
        result = {}
        for datasource_sname in sorted(all_datasources):
            datasource = self.load_datasource(config, datasource_sname)
            result[datasource_sname] = datasource

        return result

    # XXX Needs refactoring !!!
    def _default_calibsources(self):
        config = self.context.get("config")
        # calibsources = {}
        # calibsource_config_prefix = '%s.calibsources.' % config_ns
        # for section_name in [sn for sn in ini_sections if sn.startswith(calibsource_config_prefix)]:
        #     calib_files = []
        #     for k, v in [i for i in ini_cfg.items(section_name) if i[0].startswith('item.')]:
        #         fname = k.replace('item.', '')
        #         sfarray = False
        #         sfname, sfdt = [e.strip() for e in v.split(',')]
        #         if sfdt.endswith('-list'):
        #             sfarray = True
        #             sfdt = sfdt.replace('-list', '')
        #         sf = CalibrationWizardCalibFile(
        #             fieldname=fname,
        #             filename=sfname,
        #             datatype=sfdt,
        #             is_array=sfarray,
        #         )
        #         calib_files.append(sf)
        #
        #     cs = CalibrationWizardCalibSource(
        #         name=section_name.replace(datasource_config_prefix, ''),
        #         calib_directory=ini_cfg.get(config_ns, 'calibdir'),
        #         calib_files=calib_files,
        #     )
        #     calibsources[ds.name] = cs

        log.error("Calibsources are not finally implemented for online calibration !!!!")
        print config
        return {}

    def load_defaults_from_calibsource(self, calibsource_sname):
        log.info("Load Calibsource: %s (UNFINISHED!!!)" % calibsource_sname)
        if calibsource_sname in self.calibsources:
            cs_cfg = self.calibsources[calibsource_sname]
            for k, v in cs_cfg.items():
                print "XXX %s -> %s" % (k, v)
        else:
            log.warn("Missing calibsource: %s" % calibsource_sname)

    def get_fwk(self, jointangle_calib, gimbalangle_calib, disable_theta6=False, enable_2ndorder=False):
        log.info("ForwardKinematics:\njoint_lengths=%s\norigin_offset=%s\njointangle_correction=%s\ngimbalangle_correction=%s" %
                 (self.parameters.joint_lengths, self.parameters.origin_offset, jointangle_calib, gimbalangle_calib))

        cls = self.fwk_classes[1] if enable_2ndorder else self.fwk_classes[0]
        if cls is None:
            log.error("Missing implementation for ForwardKinematics.")
            raise ValueError("Missing implementation for ForwardKinematics.")

        return cls(self.parameters.joint_lengths,
                   jointangle_calib,
                   gimbalangle_calib,
                   self.parameters.origin_offset,
                   disable_theta6=disable_theta6)

    def export_data(self, filename, metadata=None):
        import pandas as pd
        from utinteractiveconsole.persistence.pandas_converters import store_data, guess_type


        log.info("Open HDF5Store for writing: %s" % filename)
        store = pd.HDFStore(filename, mode='w')
        root = store.root

        # store metadata
        if metadata is not None:
            root._v_attrs.utic_metadata = metadata

        # store parameters
        log.info("Store calibration parameters.")
        parameters = dict([(k, getattr(self.parameters, k)) for k in self.parameters.members()])
        root._v_attrs.utic_parameters = parameters

        # store results
        log.info("Store calibration results.")
        store_data(store, '/results/theta6_correction', self.result.theta6_correction_result, datatype=DataType.position3d)
        store_data(store, '/results/zaxis_reference', self.result.zaxis_reference_result, datatype=DataType.position3d)
        if self.result.zaxis_points_result:
            store_data(store, '/results/zaxis_points', self.result.zaxis_points_result, datatype=DataType.position3d, is_array=True)

        store_data(store, '/results/tooltip_calibration', self.result.tooltip_calibration_result, datatype=DataType.pose)
        store_data(store, '/results/fwkbase_position', self.result.fwkbase_position_calibration_result, datatype=DataType.position3d)
        store_data(store, '/results/fwkbase_position2', self.result.fwkbase_position2_calibration_result, datatype=DataType.position3d)

        store_data(store, '/results/absolute_orientation', self.result.absolute_orientation_result, datatype=DataType.pose)
        store_data(store, '/results/jointangles_correction', self.result.jointangles_correction_result, datatype=DataType.mat33)
        store_data(store, '/results/gimbalangles_correction', self.result.gimbalangles_correction_result, datatype=DataType.mat33)

        # store computed errors
        log.info("Store computed errors.")
        for i, poserr in enumerate(self.result.position_errors):
            store_data(store, '/evaluation/position_errors/I%02d' % i, poserr)

        for i, ornerr in enumerate(self.result.orientation_errors):
            store_data(store, '/evaluation/orientation_errors/I%02d' % i, ornerr)

        log.info("Store all datasources.")
        for key, datasource in self.datasources.items():
            datasource.export_data(store, '/datasource/%s' % key)

        log.info("Store all process data.")
        for processor_name, process_data_items in self.result.process_data.items():
            for i, process_data in enumerate(process_data_items):
                base_path = '/process_data/%s/I%02d' % (processor_name, i)
                log.info("Store process data for %s iteration %d" % (processor_name, i))

                # Store intermediate results
                for key, value in process_data.results.items():
                    try:
                        datatype = guess_type(value)
                    except TypeError, e:
                        log.warn("Unable to store process data: %s - skipping" % key)
                        log.exception(e)
                        continue
                    store_data(store, '%s/results/%s' % (base_path, key), value, datatype=datatype)

                # Store attributes
                for key, value in process_data.attributes.items():
                    try:
                        datatype = guess_type(value)
                    except TypeError:
                        # XXX maybe log here
                        continue
                    store_data(store, '%s/attributes/%s' % (base_path, key), value, datatype=datatype)

                process_data.dataset.export_data(store, '%s/dataset' % base_path)


        log.info("Close HDF5Store.")
        store.close()

