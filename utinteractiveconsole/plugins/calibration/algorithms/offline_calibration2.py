__author__ = 'jack'
import logging


from math import sin, cos, acos, atan2, sqrt, fabs, radians, degrees
import numpy as np
from numpy.linalg import norm

from scipy import odr
from scipy import stats
from scipy.stats import scoreatpercentile
from scipy.stats import nanmedian
from scipy.signal import correlate
from scipy import interpolate
from scipy import optimize
from scipy import spatial
from scipy import stats

from collections import namedtuple

from atom.api import Atom, Value, Float, Int, List, observe
import time

from ubitrack.core import measurement, math, util

log = logging.getLogger(__name__)


class CalibrationProcessor(Atom):

    # the facade will be set from the controller
    facade = Value()

    # the data stream to be consumed by this processor
    data = List()

    # def prepare_stream(self, stream,
    #                    tooltip_offset=None,
    #                    absolute_orientation=None,
    #                    forward_kinematics=None):
    #     return stream


class TooltipCalibrationProcessor(CalibrationProcessor):

    max_computation_time = Float(30)

    # data extracted from stream
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
            self.sink_result_tooltip_offset = facade.instance.getApplicationPushSinkPose("calib_tooltip_out")
            self.sink_result_tooltip_offset.setCallback(self.handler_result_tooltip_offset)
            self.source_tracker_poses = facade.instance.getApplicationPushSourcePose("tt_calib_et_pose")

    def handler_result_tooltip_offset(self, m):
        self.result_tooltip_offset = m.get()
        self.result_last_timestamp = m.time()

    def run(self):
        ts = measurement.now()

        self.data_tracker_poses = [r.externaltracker_pose for r in self.data]

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

    # data extracted from stream
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

        self.data_tracker_hip_positions = [r.externaltracker_hip_position for r in self.data]
        self.data_fwk_hip_positions = [r.haptic_pose.translation() for r in self.data]

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

    # def prepare_stream(self, stream,
    #                    tooltip_offset=None,
    #                    absolute_orientation=None,
    #                    forward_kinematics=None):
    #
    #     if tooltip_offset is None:
    #         raise ValueError("TooltipOffset not supplied")
    #
    #     if forward_kinematics is None:
    #         raise ValueError("ForwardKinematics not supplied")
    #
    #
    #     stream_fields = stream[0]._fields
    #
    #     data_fieldnames = list(stream_fields)
    #     data_fieldnames.append("haptic_pose")
    #     data_fieldnames.append("externaltracker_hip_position")
    #
    #     DataSet = namedtuple('DataSet', data_fieldnames)
    #
    #     result = []
    #
    #     for record in stream:
    #         haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
    #         externaltracker_hip_position = (record.externaltracker_pose * tooltip_offset).translation()
    #
    #         values = list(record) + [haptic_pose, externaltracker_hip_position]
    #         result.append(DataSet(*values))
    #
    #     return result


class JointAngleCalibrationProcessor(CalibrationProcessor):

    # data extracted from stream
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

        self.data_tracker_hip_positions = [r.hip_reference_pose.translation() for r in self.data]
        self.data_joint_angles = [r.jointangles for r in self.data]

        return self.sink_result_jointangle_correction.get(ts).get()

    # def prepare_stream(self, stream,
    #                    tooltip_offset=None,
    #                    absolute_orientation=None,
    #                    forward_kinematics=None):
    #
    #     if tooltip_offset is None:
    #         raise ValueError("TooltipOffset not supplied")
    #
    #     if absolute_orientation is None:
    #         raise ValueError("AbsoluteOrientation not supplied")
    #
    #     if forward_kinematics is None:
    #         raise ValueError("ForwardKinematics not supplied")
    #
    #
    #     stream_fields = stream[0]._fields
    #
    #     data_fieldnames = list(stream_fields)
    #     data_fieldnames.append("haptic_pose")
    #     data_fieldnames.append("hip_reference_pose")
    #
    #     DataSet = namedtuple('DataSet', data_fieldnames)
    #
    #     absolute_orientation_inv = absolute_orientation.invert()
    #
    #     result = []
    #
    #     for record in stream:
    #         haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
    #         hip_reference_pose = (
    #         absolute_orientation_inv * record.externaltracker_pose * tooltip_offset)
    #
    #         values = list(record) + [haptic_pose, hip_reference_pose]
    #         result.append(DataSet(*values))
    #
    #     return result



class GimbalAngleCalibrationProcessor(CalibrationProcessor):

    # input data will be set from the controller
    data_joint_angle_correction = Value()

    # data extracted from stream
    data_zrefaxis = List()
    data_joint_angles = List()
    data_gimbal_angles = List()

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

        self.data_zrefaxis = [r.zrefaxis for r in self.data]
        self.data_joint_angles = [r.jointangles for r in self.data]
        self.data_gimbal_angles = [r.gimbalangles for r in self.data]

        return self.sink_result_gimbalangle_correction.get(ts).get()

    # def prepare_stream(self, stream,
    #                    tooltip_offset=None,
    #                    absolute_orientation=None,
    #                    forward_kinematics=None,
    #                    zrefaxis_calib=None):
    #
    #     if tooltip_offset is None:
    #         raise ValueError("TooltipOffset not supplied")
    #
    #     if absolute_orientation is None:
    #         raise ValueError("AbsoluteOrientation not supplied")
    #
    #     if forward_kinematics is None:
    #         raise ValueError("ForwardKinematics not supplied")
    #
    #     if zrefaxis_calib is None:
    #         raise ValueError("Z-Axis reference not supplied")
    #
    #
    #     stream_fields = stream[0]._fields
    #
    #     data_fieldnames = list(stream_fields)
    #     data_fieldnames.append("haptic_pose")
    #     data_fieldnames.append("zrefaxis")
    #
    #     DataSet = namedtuple('DataSet', data_fieldnames)
    #
    #     absolute_orientation_inv = absolute_orientation.invert()
    #
    #     result = []
    #
    #     for record in stream:
    #         haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
    #
    #         # HIP target pose in HDorigin
    #         hiptarget_rotation = math.Quaternion((absolute_orientation_inv * record.externaltracker_pose).rotation())
    #         ht_pose_no_trans = math.Pose(hiptarget_rotation, np.array([0, 0, 0]))
    #
    #         # re-orient zrefaxis_calib using hiptarget pose
    #         zrefaxis = ht_pose_no_trans * zrefaxis_calib
    #         zrefaxis = zrefaxis / np.linalg.norm(zrefaxis)
    #
    #         values = list(record) + [haptic_pose, zrefaxis]
    #         result.append(DataSet(*values))
    #
    #     return result


class ReferenceOrientationProcessor(CalibrationProcessor):


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
        if result[2] > 0:
            result *= -1.0

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

        # the center of the circle described by the center-of-mass of the tracking target
        target_position = np.asarray([d.target_position for d in self.data])

        # center info consists of: ((xz_slope, xz_intercept), (yz_slope, yz_intercept), (xc, yc, radius, residual))
        target_circle_center, target_center_info = self.find_center(target_position)
        zaxis_points.append(target_circle_center)

        # theta6 fitting helpers
        theta6_angles = np.asarray([d.gimbalangles[2] for d in self.data])
        best_residual_radius = target_center_info[2][2] / target_center_info[2][3]
        theta6_data = dict(points=target_position, radius=target_center_info[2][2], residual=target_center_info[2][3],
                           xc=target_circle_center[0], yc=target_circle_center[1])


        if use_markers:
            num_markers = len(self.data[0].target_markers)
            target_markers = np.asarray([d.target_markers for d in self.data])
            # individual markers travel on circles as well - find their centers
            for i in range(num_markers):
                m_circle_center, m_center_info = self.find_center(target_markers[:, i, :])
                zaxis_points.append(m_circle_center)

                residual_radius = m_center_info[2][2] / m_center_info[2][3]
                if residual_radius > best_residual_radius:
                    best_residual_radius = residual_radius
                    theta6_data = dict(points=target_markers[:, i, :], radius=m_center_info[2][2], residual=m_center_info[2][3],
                                       xc=m_circle_center[0], yc=m_circle_center[1])

        zaxis = self.find_zaxis(zaxis_points)

        # project zaxis back into OTtarget coordinates
        #corrected_zaxis_ot = []
        corrected_zaxis_points_ot = []

        for record in self.data:
            hiptarget_pose_inv = record.hiptarget_pose.invert()
            # un-project markers using the corrected stylus pose (5dof)
            #corrected_zaxis_ot.append(hiptarget_pose_inv * (record.device_to_stylus_5dof * zaxis))

            # un-project the found centers for debugging
            otp = []
            for i, p in enumerate(zaxis_points):
                otp.append(hiptarget_pose_inv * (record.device_to_stylus_5dof * p))
            corrected_zaxis_points_ot.append(otp)


        if False:
            corrected_zaxis_ot = np.asarray(corrected_zaxis_ot)
            # maybe a more clever approach to mean could be used here .. svd ??
            corrected_zaxis_ot_mean = corrected_zaxis_ot.mean(axis=0)


            # normalize
            corrected_zaxis_ot_mean /= np.linalg.norm(corrected_zaxis_ot_mean)
            log.info("Corrected Z-Axis in OT space: %s" % (corrected_zaxis_ot_mean,))


            # unused data:
            zref = np.asarray(corrected_zaxis_ot_mean)
            zref = zref / np.linalg.norm(zref)


        corrected_zaxis_points_ot = np.asarray(corrected_zaxis_points_ot)
        corrected_zaxis_points_ot_mean = []
        for i in range(corrected_zaxis_points_ot.shape[1]):
            corrected_zaxis_points_ot_mean.append(corrected_zaxis_points_ot[:, i, :].mean(axis=0))

        zref = np.asarray(self.find_zaxis(corrected_zaxis_points_ot_mean))
        zref = zref / np.linalg.norm(zref)


        # compute corrections for theta6 here since data is all available
        theta6_correction = self.compute_theta6_correction(theta6_data, theta6_angles)

        return zref, corrected_zaxis_points_ot_mean, theta6_correction

    # def prepare_stream(self, stream,
    #                    tooltip_offset=None,
    #                    absolute_orientation=None,
    #                    forward_kinematics=None,
    #                    forward_kinematics_5dof=None,
    #                    use_markers=True):
    #
    #     if tooltip_offset is None:
    #         raise ValueError("TooltipOffset not supplied")
    #
    #     if absolute_orientation is None:
    #         raise ValueError("AbsoluteOrientation not supplied")
    #
    #     if forward_kinematics is None:
    #         raise ValueError("ForwardKinematics not supplied")
    #     stream_fields = stream[0]._fields
    #
    #     data_fieldnames = list(stream_fields)
    #     data_fieldnames.append("haptic_pose")
    #     data_fieldnames.append("mean_marker_error")
    #     data_fieldnames.append("target_position")
    #     data_fieldnames.append("target_markers")
    #     data_fieldnames.append("device_to_stylus_5dof")
    #     data_fieldnames.append("hiptarget_pose")
    #
    #     DataSet = namedtuple('DataSet', data_fieldnames)
    #
    #     absolute_orientation_inv = absolute_orientation.invert()
    #
    #     # find marker count and verify that it is constant for the complete dataset
    #     nmarkers = 0
    #     if use_markers:
    #         nmarkers = len(stream[0].externaltracker_markers)
    #         assert (np.asarray([len(d.externaltracker_markers) for d in stream
    #                             if d.externaltracker_markers is not None]) == nmarkers).all()
    #
    #     rel_marker_positions = []
    #     skipped_markers = 0
    #
    #
    #     result = []
    #
    #     for record in stream:
    #         # fwk pose in HDorigin
    #         haptic_pose = forward_kinematics.calculate_pose(record.jointangles, record.gimbalangles)
    #
    #         # HIP target pose in HDorigin
    #         hiptarget_pose = absolute_orientation_inv * record.externaltracker_pose
    #         hiptarget_pose_inv = hiptarget_pose.invert()
    #
    #         mean_marker_error = None
    #         markers = []
    #
    #         if use_markers:
    #             if record.externaltracker_markers is not None:
    #                 # HIP target markers in HDorigin
    #                 hiptarget_markers = [absolute_orientation_inv * m for m in record.externaltracker_markers]
    #                 if not rel_marker_positions:
    #                     # initialize positions
    #                     for m in hiptarget_markers:
    #                         rel_marker_positions.append(hiptarget_pose_inv * m)
    #                         markers.append(m)
    #                 else:
    #                     # validate positions
    #                     markers = [None, ] * nmarkers
    #                     dbg_dists = []
    #                     for m in hiptarget_markers:
    #                         dist_ = []
    #                         for i, relm in enumerate(rel_marker_positions):
    #                             dist_.append((i, norm(relm - (hiptarget_pose_inv * m))))
    #                         dist_ = sorted(dist_, lambda x, y: cmp(x[1], y[1]))
    #                         markers[dist_[0][0]] = m
    #                         dbg_dists.append(dist_)
    #
    #                     mean_marker_error = np.asarray([d[0][1] for d in dbg_dists]).mean()
    #
    #                     # XXX remove print statements or improve to make useful logging output
    #                     if None in markers:
    #                         log.warn("Incomplete marker dataset received!")
    #                         # print "Markers:"
    #                         # for m in markers:
    #                         #     print m
    #                         # print "Distances:"
    #                         # for i, dist_ in enumerate(dbg_dists):
    #                         #     print "Item%d" % i
    #                         #     for i,m in dist_:
    #                         #         print i, m
    #                         skipped_markers += 1
    #                         markers = []
    #
    #         # calculate the stylus pose (5DOF) based on the calibrated correction_factors and the measured angles
    #         device_to_stylus_5dof = forward_kinematics_5dof.calculate_pose(record.jointangles, record.gimbalangles)
    #         device_to_stylus_5dof_inv = device_to_stylus_5dof.invert()
    #
    #         # multiply the inverse of the externaltracker to stylus transform with the externaltracker to stylus target
    #         target_position = (device_to_stylus_5dof_inv * hiptarget_pose).translation()
    #
    #         # back-project markers using the corrected stylus pose (5dof)
    #         if use_markers and markers:
    #             target_markers = np.asarray([device_to_stylus_5dof_inv * m for m in markers])
    #         else:
    #             target_markers = None
    #
    #         values = list(record) + [haptic_pose, mean_marker_error,
    #                                  target_position, target_markers, device_to_stylus_5dof,
    #                                  hiptarget_pose]
    #         result.append(DataSet(*values))
    #
    #     return result
