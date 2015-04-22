__author__ = 'jack'

import logging
import os
import sys
import numpy as np
from math import sin, cos
from ubitrack.core import math

log = logging.getLogger(__name__)

angle_null_correction = np.array([[0.0, 1.0, 0.0, ], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

class FWKinematicScale(object):

    def __init__(self, joint_lengths, platformsensor_correction, jointangle_correction, gimbalangle_correction, disable_theta6=False):
        self.joint_lengths = joint_lengths
        self.platformsensor_correction_factors = np.array(platformsensor_correction)
        self.jointangle_correction_factors = np.array(jointangle_correction)
        self.gimbalangle_correction_factors = np.array(gimbalangle_correction)
        if disable_theta6:
            self.gimbalangle_correction_factors[2, 0] = 0.0
            self.gimbalangle_correction_factors[2, 1] = 0.0
            self.gimbalangle_correction_factors[2, 2] = 0.0

    def calculate_position(self, platform_sensors, joint_angles):
        pscf = self.platformsensor_correction_factors
        jacf = self.jointangle_correction_factors

        S1 = platform_sensors[0]
        S2 = platform_sensors[1]

        h1 = pscf[0, 1]
        j1 = pscf[0, 2]
        h2 = pscf[1, 1]
        j2 = pscf[1, 2]

        O1 = joint_angles[0]
        O2 = joint_angles[1]
        O3 = joint_angles[2]

        k1 = jacf[0, 1]
        m1 = jacf[0, 2]
        k2 = jacf[1, 1]
        m2 = jacf[1, 2]
        k3 = jacf[2, 1]
        m3 = jacf[2, 2]

        S1_ = S1 * h1 + j1
        S2_ = S2 * h2 + j2

        O1_ = O1 * k1 + m1
        O2_ = O2 * k2 + m2
        O3_ = O3 * k3 + m3

        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        # calculate translation
        trans = np.array(
            [S1_ + l1*cos(O1_)*cos(O2_) + l2*cos(O1_)*cos(O2_ + O3_),
             S2_ + l1*sin(O1_)*cos(O2_) + l2*sin(O1_)*cos(O2_ + O3_),
             -l1*sin(O2_) - l2*sin(O2_ + O3_)]
        )

        return trans

    def calculate_pose(self, platform_sensors, joint_angles, gimbal_angles):
        pscf = self.platformsensor_correction_factors
        jacf = self.jointangle_correction_factors
        gacf = self.gimbalangle_correction_factors

        S1 = platform_sensors[0]
        S2 = platform_sensors[1]

        h1 = pscf[0, 1]
        j1 = pscf[0, 2]
        h2 = pscf[1, 1]
        j2 = pscf[1, 2]

        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        O1 = joint_angles[0]
        O2 = joint_angles[1]
        O3 = joint_angles[2]
        O4 = gimbal_angles[0]
        O5 = gimbal_angles[1]
        O6 = gimbal_angles[2]

        k1 = jacf[0, 1]
        m1 = jacf[0, 2]
        k2 = jacf[1, 1]
        m2 = jacf[1, 2]
        k3 = jacf[2, 1]
        m3 = jacf[2, 2]
        k4 = gacf[0, 1]
        m4 = gacf[0, 2]
        k5 = gacf[1, 1]
        m5 = gacf[1, 2]
        k6 = gacf[2, 1]
        m6 = gacf[2, 2]

        S1_ = S1 * h1 + j1
        S2_ = S2 * h2 + j2

        O1_ = O1 * k1 + m1
        O2_ = O2 * k2 + m2
        O3_ = O3 * k3 + m3
        O4_ = O4 * k4 + m4
        O5_ = O5 * k5 + m5
        O6_ = O6 * k6 + m6

        # calculate translation
        trans = np.array(
            [S1_ + l1*cos(O1_)*cos(O2_) + l2*cos(O1_)*cos(O2_ + O3_),
             S2_ + l1*sin(O1_)*cos(O2_) + l2*sin(O1_)*cos(O2_ + O3_),
             -l1*sin(O2_) - l2*sin(O2_ + O3_)]
        )

        # calculate rotation of arm
        rot = np.array([[-(sin(O4_)*sin(O5_)*cos(O6_) + sin(O6_)*cos(O4_))*sin(O1_) + (sin(O4_)*sin(O6_)*sin(O2_ + O3_) - sin(O5_)*sin(O2_ + O3_)*cos(O4_)*cos(O6_) + cos(O5_)*cos(O6_)*cos(O2_ + O3_))*cos(O1_),
                         (sin(O4_)*sin(O5_)*cos(O6_) + sin(O6_)*cos(O4_))*cos(O1_) + (sin(O4_)*sin(O6_)*sin(O2_ + O3_) - sin(O5_)*sin(O2_ + O3_)*cos(O4_)*cos(O6_) + cos(O5_)*cos(O6_)*cos(O2_ + O3_))*sin(O1_),
                         sin(O4_)*sin(O6_)*cos(O2_ + O3_) - sin(O5_)*cos(O4_)*cos(O6_)*cos(O2_ + O3_) - sin(O2_ + O3_)*cos(O5_)*cos(O6_)],
                        [(sin(O4_)*sin(O5_)*sin(O6_) - cos(O4_)*cos(O6_))*sin(O1_) + (sin(O4_)*sin(O2_ + O3_)*cos(O6_) + sin(O5_)*sin(O6_)*sin(O2_ + O3_)*cos(O4_) - sin(O6_)*cos(O5_)*cos(O2_ + O3_))*cos(O1_),
                         -(sin(O4_)*sin(O5_)*sin(O6_) - cos(O4_)*cos(O6_))*cos(O1_) + (sin(O4_)*sin(O2_ + O3_)*cos(O6_) + sin(O5_)*sin(O6_)*sin(O2_ + O3_)*cos(O4_) - sin(O6_)*cos(O5_)*cos(O2_ + O3_))*sin(O1_),
                         sin(O4_)*cos(O6_)*cos(O2_ + O3_) + sin(O5_)*sin(O6_)*cos(O4_)*cos(O2_ + O3_) + sin(O6_)*sin(O2_ + O3_)*cos(O5_)],
                        [(sin(O5_)*cos(O2_ + O3_) + sin(O2_ + O3_)*cos(O4_)*cos(O5_))*cos(O1_) + sin(O1_)*sin(O4_)*cos(O5_),
                         (sin(O5_)*cos(O2_ + O3_) + sin(O2_ + O3_)*cos(O4_)*cos(O5_))*sin(O1_) - sin(O4_)*cos(O1_)*cos(O5_),
                         -sin(O5_)*sin(O2_ + O3_) + cos(O4_)*cos(O5_)*cos(O2_ + O3_)]])
        return math.Pose(math.Quaternion.fromMatrix(rot).normalize(), trans)



class FWKinematicVirtuose(object):

    def __init__(self, joint_lengths, jointangle_correction, gimbalangle_correction, disable_theta6=False):
        self.joint_lengths = joint_lengths
        self.jointangle_correction_factors = np.array(jointangle_correction)
        self.gimbalangle_correction_factors = np.array(gimbalangle_correction)
        if disable_theta6:
            self.gimbalangle_correction_factors[2, 0] = 0.0
            self.gimbalangle_correction_factors[2, 1] = 0.0
            self.gimbalangle_correction_factors[2, 2] = 0.0

    def calculate_position(self, joint_angles):
        jacf = self.jointangle_correction_factors

        O1 = joint_angles[0]
        O2 = joint_angles[1]
        O3 = joint_angles[2]

        k1 = jacf[0, 1]
        m1 = jacf[0, 2]
        k2 = jacf[1, 1]
        m2 = jacf[1, 2]
        k3 = jacf[2, 1]
        m3 = jacf[2, 2]

        S1_ = S1 * h1 + j1
        S2_ = S2 * h2 + j2

        O1_ = O1 * k1 + m1
        O2_ = O2 * k2 + m2
        O3_ = O3 * k3 + m3

        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        # calculate translation
        trans = np.array(
            [l1*cos(O1_)*cos(O2_) + l2*cos(O1_)*cos(O2_ + O3_),
             l1*sin(O1_)*cos(O2_) + l2*sin(O1_)*cos(O2_ + O3_),
             -l1*sin(O2_) - l2*sin(O2_ + O3_)]
        )

        return trans

    def calculate_pose(self, joint_angles, gimbal_angles):
        jacf = self.jointangle_correction_factors
        gacf = self.gimbalangle_correction_factors

        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        O1 = joint_angles[0]
        O2 = joint_angles[1]
        O3 = joint_angles[2]
        O4 = gimbal_angles[0]
        O5 = gimbal_angles[1]
        O6 = gimbal_angles[2]

        k1 = jacf[0, 1]
        m1 = jacf[0, 2]
        k2 = jacf[1, 1]
        m2 = jacf[1, 2]
        k3 = jacf[2, 1]
        m3 = jacf[2, 2]
        k4 = gacf[0, 1]
        m4 = gacf[0, 2]
        k5 = gacf[1, 1]
        m5 = gacf[1, 2]
        k6 = gacf[2, 1]
        m6 = gacf[2, 2]

        O1_ = O1 * k1 + m1
        O2_ = O2 * k2 + m2
        O3_ = O3 * k3 + m3
        O4_ = O4 * k4 + m4
        O5_ = O5 * k5 + m5
        O6_ = O6 * k6 + m6

        # calculate translation
        trans = np.array(
            [l1*cos(O1_)*cos(O2_) + l2*cos(O1_)*cos(O2_ + O3_),
             l1*sin(O1_)*cos(O2_) + l2*sin(O1_)*cos(O2_ + O3_),
             -l1*sin(O2_) - l2*sin(O2_ + O3_)]
        )

        # calculate rotation of arm
        rot = np.array([[-(sin(O4_)*sin(O5_)*cos(O6_) + sin(O6_)*cos(O4_))*sin(O1_) + (sin(O4_)*sin(O6_)*sin(O2_ + O3_) - sin(O5_)*sin(O2_ + O3_)*cos(O4_)*cos(O6_) + cos(O5_)*cos(O6_)*cos(O2_ + O3_))*cos(O1_),
                         (sin(O4_)*sin(O5_)*cos(O6_) + sin(O6_)*cos(O4_))*cos(O1_) + (sin(O4_)*sin(O6_)*sin(O2_ + O3_) - sin(O5_)*sin(O2_ + O3_)*cos(O4_)*cos(O6_) + cos(O5_)*cos(O6_)*cos(O2_ + O3_))*sin(O1_),
                         sin(O4_)*sin(O6_)*cos(O2_ + O3_) - sin(O5_)*cos(O4_)*cos(O6_)*cos(O2_ + O3_) - sin(O2_ + O3_)*cos(O5_)*cos(O6_)],
                        [(sin(O4_)*sin(O5_)*sin(O6_) - cos(O4_)*cos(O6_))*sin(O1_) + (sin(O4_)*sin(O2_ + O3_)*cos(O6_) + sin(O5_)*sin(O6_)*sin(O2_ + O3_)*cos(O4_) - sin(O6_)*cos(O5_)*cos(O2_ + O3_))*cos(O1_),
                         -(sin(O4_)*sin(O5_)*sin(O6_) - cos(O4_)*cos(O6_))*cos(O1_) + (sin(O4_)*sin(O2_ + O3_)*cos(O6_) + sin(O5_)*sin(O6_)*sin(O2_ + O3_)*cos(O4_) - sin(O6_)*cos(O5_)*cos(O2_ + O3_))*sin(O1_),
                         sin(O4_)*cos(O6_)*cos(O2_ + O3_) + sin(O5_)*sin(O6_)*cos(O4_)*cos(O2_ + O3_) + sin(O6_)*sin(O2_ + O3_)*cos(O5_)],
                        [(sin(O5_)*cos(O2_ + O3_) + sin(O2_ + O3_)*cos(O4_)*cos(O5_))*cos(O1_) + sin(O1_)*sin(O4_)*cos(O5_),
                         (sin(O5_)*cos(O2_ + O3_) + sin(O2_ + O3_)*cos(O4_)*cos(O5_))*sin(O1_) - sin(O4_)*cos(O1_)*cos(O5_),
                         -sin(O5_)*sin(O2_ + O3_) + cos(O4_)*cos(O5_)*cos(O2_ + O3_)]])
        return math.Pose(math.Quaternion.fromMatrix(rot).normalize(), trans)



def from_config(root_directory, context,
                gimbalangle_correction=None,
                jointangle_correction=None, joint_length=None, disable_theta6=None,
                *args, **kwargs):
    from ubitrack.core import util

    log.info("Create FWK Instance with: %s, %s" % (jointangle_correction, gimbalangle_correction, ))
    calib_directory = os.path.join(root_directory, context.get("calib_directory"))

    ja_correction = util.readCalibMeasurementMatrix3x3(os.path.join(calib_directory, jointangle_correction).encode(sys.getfilesystemencoding())) if jointangle_correction else angle_null_correction
    ga_correction = util.readCalibMeasurementMatrix3x3(os.path.join(calib_directory, gimbalangle_correction).encode(sys.getfilesystemencoding())) if gimbalangle_correction else angle_null_correction
    j_length = np.array([float(v.strip()) for v in joint_length.split(',')]) if joint_length else None
    d_theta6 = disable_theta6 if disable_theta6 is not None else False
    if isinstance(d_theta6, (str, unicode)):
        d_theta6 = d_theta6.strip().lower() == 'true'

    if j_length is not None and ja_correction is not None and ga_correction is not None:
        return FWKinematicVirtuose(j_length, ja_correction, ga_correction, disable_theta6=d_theta6)
    raise ValueError("Invalid configuration for FWKinematicPhantom")
