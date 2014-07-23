__author__ = 'jack'

import logging
import os
import sys

from math import sin, cos, fabs, radians
import numpy as np
from scipy import odr
from scipy import stats
from scipy.stats import scoreatpercentile
from scipy.stats import nanmedian


from ubitrack.core import util, measurement, math
from utinteractiveconsole.playback import (loadData, DSC, interpolatePoseList,
                                           interpolateVec3List, selectOnlyMatchingSamples)

from utinteractiveconsole.plugins.calibration.hapticdevice.phantom_forward_kinematics import FWKinematicPhantom

log = logging.getLogger(__name__)


def loadCalibrationFiles(root_dir):
    if isinstance(root_dir, unicode):
        root_dir = root_dir.encode(sys.getdefaultencoding())

    # XXX needs refactoring (ubitrack components, all DFGs, this code)
    fname = os.path.join(root_dir, "phantom_jointangle_correction.calib")
    if os.path.isfile(fname):
        phantom_jointangle_calib = util.readCalibMeasurementMatrix3x4(fname)
        log.info("Phantom JointAngle Calibration\n%s" % (phantom_jointangle_calib.reshape((6, 2)),))
    else:
        log.warn("Phantom JointAngle Calibration NOT FOUND")
        phantom_jointangle_calib = np.array([1.0, 0.0]*6).reshape((3, 4))

    fname = os.path.join(root_dir, "phantom_gimbalangle_correction.calib")
    if os.path.isfile(fname):
        phantom_gimbalangle_calib = util.readCalibMeasurementMatrix3x4(fname)
        log.info("Phantom GimbalAngle Calibration\n%s" % (phantom_gimbalangle_calib.reshape((6, 2)),))
    else:
        log.warn("Phantom Angle Calibration 5DOF NOT FOUND")
        phantom_gimbalangle_calib = np.array([1.0, 0.0]*6).reshape((3, 4))

    fname = os.path.join(root_dir, "absolute_orientation_calibration.calib")
    if os.path.isfile(fname):
        externaltracker_to_device = util.readCalibMeasurementPose(fname)
        log.info("OptiTrack to Device Transform\n%s" % (externaltracker_to_device,))
    else:
        log.warn("Absolute Orientation Calibration NOT FOUND")
        externaltracker_to_device = math.Pose(math.Quaternion(), np.array([0.0, 0.0, 0.0]))

    fname = os.path.join(root_dir, "tooltip_calibration.calib")
    if os.path.isfile(fname):
        tooltip_calib = util.readCalibMeasurementPosition(fname)
        log.info("Tooltip Calibration\n%s" % (tooltip_calib,))
    else:
        log.warn("Tooltip Calibration NOT FOUND")
        tooltip_calib = np.array([0.0, 0.0, 0.0])

    return dict(phantom_jointangle_calib=phantom_jointangle_calib,
                phantom_gimbalangle_calib=phantom_gimbalangle_calib,
                externaltracker_to_device=externaltracker_to_device,
                tooltip_calib=tooltip_calib,
                )




def loadCalibrationFiles2(root_dir):
    if isinstance(root_dir, unicode):
        root_dir = root_dir.encode(sys.getdefaultencoding())


    fname = os.path.join(root_dir, "phantom_jointangle_correction.calib")
    if os.path.isfile(fname):
        phantom_jointangle_calib = util.readCalibMeasurementMatrix3x3(fname)
        log.info("Phantom Joint Angle Calibration\n%s" % (phantom_jointangle_calib),)
    else:
        log.warn("Phantom Joint Angle Calibration NOT FOUND")
        phantom_jointangle_calib = np.array([0.0, 1.0, 0.0]*3).reshape((3, 3))

    phantom_gimbalangle_calib = np.array([0.0, 1.0, 0.0]*3).reshape((3, 3))

    fname = os.path.join(root_dir, "absolute_orientation_calibration.calib")
    if os.path.isfile(fname):
        externaltracker_to_device = util.readCalibMeasurementPose(fname)
        log.info("OptiTrack to Device Transform\n%s" % (externaltracker_to_device,))
    else:
        log.warn("Absolute Orientation Calibration NOT FOUND")
        externaltracker_to_device = math.Pose(math.Quaternion(), np.array([0.0, 0.0, 0.0]))

    fname = os.path.join(root_dir, "tooltip_calibration.calib")
    if os.path.isfile(fname):
        tooltip_calib = util.readCalibMeasurementPosition(fname)
        log.info("Tooltip Calibration\n%s" % (tooltip_calib,))
    else:
        log.warn("Tooltip Calibration NOT FOUND")
        tooltip_calib = np.array([0.0, 0.0, 0.0])

    return dict(phantom_jointangle_calib=phantom_jointangle_calib,
                phantom_gimbalangle_calib=phantom_gimbalangle_calib,
                externaltracker_to_device=externaltracker_to_device,
                tooltip_calib=tooltip_calib,
                )







# simple line fitting
def fit_line(v1, v2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(v1, v2)
    return slope, intercept

# fit a circle on the xy-part of the data
# circle fitting from scipy howto
def fit_circle(x, y):

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
    beta0 = [ x_m, y_m, R_m]

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


def find_center(data):
    x_data = data[:, 0]
    y_data = data[:, 1]
    z_data = data[:, 2]

    # fit a line onto the measured points
    xz_slope, xz_intercept = fit_line(x_data,z_data)

    # fit a line onto the measured points
    yz_slope, yz_intercept = fit_line(y_data,z_data)

    #log.info("distance difference between xz and yz line fitting: %f" % (xz_intercept - yz_intercept))
    log.info("xz slope: %f, yz slope: %f, xz intercept: %f, yz intercept: %f, z_mean: %f" % (xz_slope, yz_slope, xz_intercept, yz_intercept, z_data.mean()))

    # Fit the circle
    xc, yc, radius, residual = fit_circle(x_data, y_data)
    log.info("Center of circle: [%f, %f], radius %f, residual: %f" % (xc, yc, radius, residual))

    center = [xc, yc, np.array([xz_intercept,yz_intercept]).mean()]
    log.info("Center: %s" % (center,))
    # return x/z line, y/z line, x/y circle
    return center, ((xz_slope, xz_intercept), (yz_slope, yz_intercept), (xc, yc, radius, residual))


def find_zaxis(corrected_zaxis_points):
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    data = np.asarray(corrected_zaxis_points)
    datamean = data.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.
    result = np.asarray(vv[0])
    if result[2] < 0:
        result *= -1.0

    return result


class CalculateZRefAxis(object):


    def __init__(self, record_dir, calib_dir, joint_lengths, origin_offset, use_2ndorder=False):
        self.record_dir = record_dir
        self.calib_dir = calib_dir
        if use_2ndorder:
            self.calibrations = self.load_calibrations2()
            self.fwk = FWKinematicPhantom2(joint_lengths,
                                           self.calibrations["phantom_jointangle_calib"],
                                           self.calibrations["phantom_gimbalangle_calib"],
                                           origin_offset,
                                           disable_theta6=True)
        else:
            self.calibrations = self.load_calibrations()
            self.fwk = FWKinematicPhantom(joint_lengths,
                                          self.calibrations["phantom_jointangle_calib"],
                                          origin_offset,
                                          disable_theta6=True)

    def load_data(self):
        return loadData(
                self.record_dir,
                DSC('externaltracker_pose', 'record_phantom_hip_target.log',
                    util.PoseStreamReader),
                items=(DSC('haptic_pose', 'record_haptic_raw.log',
                           util.PoseStreamReader, interpolatePoseList),
                       DSC('jointangles_interp', 'record_jointangles_raw.log',
                           util.PositionStreamReader, interpolateVec3List),
                       DSC('gimbalangles_interp', 'record_gimbalangles_raw.log',
                           util.PositionStreamReader, interpolateVec3List),
                       DSC('hiptarget_markers', 'record_phantom_hip_target_markers.log',
                           util.PositionListStreamReader, selectOnlyMatchingSamples),
                       )
                )

    def load_calibrations(self):
        return loadCalibrationFiles(self.calib_dir)

    def load_calibrations2(self):
        return loadCalibrationFiles2(self.calib_dir)


    def process_data(self, data, data_slice=slice(0, -1, 1), use_markers=True):
        target_position = []
        gimbal_angles = []

        target_markers = []
        rel_marker_positions = []
        skipped_markers = 0
        avg_marker_dist = []

        nmarkers = 0
        if use_markers:
            nmarkers = len(data[0].hiptarget_markers)
            assert (np.asarray([len(d.hiptarget_markers) for d in data if d.hiptarget_markers is not None]) == nmarkers).all()

        for ds in data[data_slice]:

            # calculate the stylus pose (5DOF) based on the calibrated correction_factors and the measured angles
            device_to_stylus = self.fwk.calculate_pose(ds.jointangles_interp,
                                                       ds.gimbalangles_interp)

            if use_markers:
                markers = []
                if ds.hiptarget_markers is not None:
                    if not rel_marker_positions:
                        # initialize positions
                        for m in ds.hiptarget_markers:
                            rel_marker_positions.append(ds.externaltracker_pose.invert() * m)
                            markers.append(m)
                    else:
                        # validate positions
                        markers = [None,] * nmarkers
                        dbg_dists = []
                        for m in ds.hiptarget_markers:
                            dist_ = []
                            for i, relm in enumerate(rel_marker_positions):
                                dist_.append((i, np.linalg.norm(relm - (ds.externaltracker_pose.invert() * m))))
                            dist_ = sorted(dist_, lambda x,y: cmp(x[1], y[1]))
                            markers[dist_[0][0]] = m

                            dbg_dists.append(dist_)

                        avg_marker_dist.append(np.asarray([d[0][1] for d in dbg_dists]).mean())

                        # XXX remove print statements or improve to make useful logging output
                        if None in markers:
                            print "Markers:"
                            for m in markers:
                                print m
                            print "Distances:"
                            for i, dist_ in enumerate(dbg_dists):
                                print "Item%d" % i
                                for i,m in dist_:
                                    print i, m
                            skipped_markers += 1
                            markers = []

            # multiply the inverse of the externaltracker to stylus transform with the externaltracker to stylus target
            target_position.append((device_to_stylus.invert() * ds.externaltracker_pose).translation())

            # back-project markers using the corrected stylus pose (5dof)
            if use_markers and markers:
                target_markers.append(np.asarray([device_to_stylus.invert() * m for m in markers]))

            gimbal_angles.append(ds.gimbalangles_interp)

        gimbal_angles = np.asarray(gimbal_angles)
        target_position = np.asarray(target_position)

        target_markers = np.asarray(target_markers)
        ltt = np.linalg.norm(self.calibrations["tooltip_calib"])

        return dict(target_position=target_position,
                    gimbal_angles=gimbal_angles,
                    target_markers=target_markers,
                    avg_marker_dist=avg_marker_dist,
                    nmarkers=nmarkers,
                    ltt=ltt)

    def transform_to_ot_coords(self, all_data, data_slice, corrected_zaxis, corrected_zaxis_points):
        corrected_zaxis_ot = []
        corrected_zaxis_points_ot = []

        for ds in all_data[data_slice]:

            # calculate the stylus pose (5DOF) based on the calibrated correction_factors and the measured angles
            device_to_stylus = self.fwk.calculate_pose(ds.jointangles_interp,
                                                       ds.gimbalangles_interp)

            # un-project markers using the corrected stylus pose (5dof)
            corrected_zaxis_ot.append(ds.externaltracker_pose.invert() * (device_to_stylus * corrected_zaxis))

            # un-project the found centers for debugging
            otp = []
            for i,p in enumerate(corrected_zaxis_points):
                otp.append(ds.externaltracker_pose.invert() * (device_to_stylus * p))
            corrected_zaxis_points_ot.append(otp)

        corrected_zaxis_ot = np.asarray(corrected_zaxis_ot)
        # maybe a more clever approach to mean could be used here .. svd ??
        corrected_zaxis_ot_mean = corrected_zaxis_ot.mean(axis=0)
        # normalize
        corrected_zaxis_ot_mean /= np.linalg.norm(corrected_zaxis_ot_mean)
        log.info("Corrected Z-Axis in OT space: %s" % (corrected_zaxis_ot_mean,))

        corrected_zaxis_points_ot = np.asarray(corrected_zaxis_points_ot)
        corrected_zaxis_points_ot_mean = []
        for i in range(corrected_zaxis_points_ot.shape[1]):
            corrected_zaxis_points_ot_mean.append(corrected_zaxis_points_ot[:,i,:].mean(axis=0))

        return corrected_zaxis_ot_mean, corrected_zaxis_points_ot_mean

    def save_result(self, zaxis):
        ts = measurement.now()
        zref = np.asarray(zaxis)
        zref = zref / np.linalg.norm(zref)
        zref_m = measurement.Position(ts, zref)
        util.writeCalibMeasurementPosition(os.path.join(self.calib_dir, "zref_axis.calib"), zref_m)


    def run(self, data_slice=slice(0, -1, 1), use_markers=True):
        raw_data = self.load_data()
        data = self.process_data(raw_data, data_slice=data_slice,  use_markers=use_markers)

        # haptic interface point as origin in the corrent coordinate system
        zaxis_points = [np.array([0.0, 0.0, 0.0])]

        # the center of the circle described by the center-of-mass of the tracking target
        target_position = data["target_position"]
        target_circle_center, target_center_info = find_center(target_position)
        zaxis_points.append(target_circle_center)

        if use_markers:
            target_markers = data["target_markers"]
            # individual markers travel on circles as well - find their centers
            for i in range(data["nmarkers"]):
                m_circle_center, m_center_info = find_center(target_markers[:, i, :])
                zaxis_points.append(m_circle_center)

        zaxis = find_zaxis(zaxis_points)
        zaxis_ot, zaxis_points_ot = self.transform_to_ot_coords(raw_data, data_slice, zaxis, zaxis_points)
        self.save_result(zaxis_ot)
