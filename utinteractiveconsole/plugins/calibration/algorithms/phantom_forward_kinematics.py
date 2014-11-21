__author__ = 'jack'

import logging
import os
import sys
import numpy as np
from math import sin, cos
from ubitrack.core import math

log = logging.getLogger(__name__)

class FWKinematicPhantom(object):

    def __init__(self, joint_lengths, jointangle_correction, gimbalangle_correction, origin_calib=None, disable_theta6=False):
        self.joint_lengths = joint_lengths
        self.jointangle_correction_factors = np.array(jointangle_correction)
        self.gimbalangle_correction_factors = np.array(gimbalangle_correction)
        if disable_theta6:
            self.gimbalangle_correction_factors[2, 0] = 0.0
            self.gimbalangle_correction_factors[2, 1] = 0.0
            self.gimbalangle_correction_factors[2, 2] = 0.0
        self.origin_calib = origin_calib if origin_calib is not None \
            else np.array([0., 0., 0.])


    def calculate_position(self, joint_angles):
        jacf = self.jointangle_correction_factors
        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        calx = self.origin_calib[0]
        caly = self.origin_calib[1]
        calz = self.origin_calib[2]

        O1 = joint_angles[0]
        O2 = joint_angles[1]
        O3 = joint_angles[2]

        k1 = jacf[0, 1]
        m1 = jacf[0, 2]
        k2 = jacf[1, 1]
        m2 = jacf[1, 2]
        k3 = jacf[2, 1]
        m3 = jacf[2, 2]

        # calculate translation
        trans = np.array(
            [calx - (l1*cos(O2*k2 + m2) + l2*sin(O3*k3 + m3))*sin(O1*k1 + m1),
             caly + l1*sin(O2*k2 + m2) - l2*cos(O3*k3 + m3) + l2,
             calz - l1 + (l1*cos(O2*k2 + m2) + l2*sin(O3*k3 + m3))*cos(O1*k1 + m1)]
        )

        return trans


    def calculate_pose(self, joint_angles, gimbal_angles):
        jacf = self.jointangle_correction_factors
        gacf = self.gimbalangle_correction_factors

        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        calx = self.origin_calib[0]
        caly = self.origin_calib[1]
        calz = self.origin_calib[2]

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

        # calculate translation
        trans = np.array(
            [calx - (l1*cos(O2*k2 + m2) + l2*sin(O3*k3 + m3))*sin(O1*k1 + m1),
             caly + l1*sin(O2*k2 + m2) - l2*cos(O3*k3 + m3) + l2,
             calz - l1 + (l1*cos(O2*k2 + m2) + l2*sin(O3*k3 + m3))*cos(O1*k1 + m1)]
        )

        # calculate rotation of arm
        rot = np.array([[-(-(-sin(O1*k1 + m1)*cos(O3*k3 + m3)*cos(O4*k4 + m4) + sin(O4*k4 + m4)*cos(O1*k1 + m1))*sin(O5*k5 + m5) + sin(O1*k1 + m1)*sin(O3*k3 + m3)*cos(O5*k5 + m5))*sin(O6*k6 + m6) + (sin(O1*k1 + m1)*sin(O4*k4 + m4)*cos(O3*k3 + m3) + cos(O1*k1 + m1)*cos(O4*k4 + m4))*cos(O6*k6 + m6),
                         (-(-sin(O1*k1 + m1)*cos(O3*k3 + m3)*cos(O4*k4 + m4) + sin(O4*k4 + m4)*cos(O1*k1 + m1))*sin(O5*k5 + m5) + sin(O1*k1 + m1)*sin(O3*k3 + m3)*cos(O5*k5 + m5))*cos(O6*k6 + m6) + (sin(O1*k1 + m1)*sin(O4*k4 + m4)*cos(O3*k3 + m3) + cos(O1*k1 + m1)*cos(O4*k4 + m4))*sin(O6*k6 + m6),
                         (-sin(O1*k1 + m1)*cos(O3*k3 + m3)*cos(O4*k4 + m4) + sin(O4*k4 + m4)*cos(O1*k1 + m1))*cos(O5*k5 + m5) + sin(O1*k1 + m1)*sin(O3*k3 + m3)*sin(O5*k5 + m5)],
                        [-(-sin(O3*k3 + m3)*sin(O5*k5 + m5)*cos(O4*k4 + m4) + cos(O3*k3 + m3)*cos(O5*k5 + m5))*sin(O6*k6 + m6) - sin(O3*k3 + m3)*sin(O4*k4 + m4)*cos(O6*k6 + m6),
                         (-sin(O3*k3 + m3)*sin(O5*k5 + m5)*cos(O4*k4 + m4) + cos(O3*k3 + m3)*cos(O5*k5 + m5))*cos(O6*k6 + m6) - sin(O3*k3 + m3)*sin(O4*k4 + m4)*sin(O6*k6 + m6),
                         sin(O3*k3 + m3)*cos(O4*k4 + m4)*cos(O5*k5 + m5) + sin(O5*k5 + m5)*cos(O3*k3 + m3)],
                        [-(-(sin(O1*k1 + m1)*sin(O4*k4 + m4) + cos(O1*k1 + m1)*cos(O3*k3 + m3)*cos(O4*k4 + m4))*sin(O5*k5 + m5) - sin(O3*k3 + m3)*cos(O1*k1 + m1)*cos(O5*k5 + m5))*sin(O6*k6 + m6) + (sin(O1*k1 + m1)*cos(O4*k4 + m4) - sin(O4*k4 + m4)*cos(O1*k1 + m1)*cos(O3*k3 + m3))*cos(O6*k6 + m6),
                         (-(sin(O1*k1 + m1)*sin(O4*k4 + m4) + cos(O1*k1 + m1)*cos(O3*k3 + m3)*cos(O4*k4 + m4))*sin(O5*k5 + m5) - sin(O3*k3 + m3)*cos(O1*k1 + m1)*cos(O5*k5 + m5))*cos(O6*k6 + m6) + (sin(O1*k1 + m1)*cos(O4*k4 + m4) - sin(O4*k4 + m4)*cos(O1*k1 + m1)*cos(O3*k3 + m3))*sin(O6*k6 + m6),
                         (sin(O1*k1 + m1)*sin(O4*k4 + m4) + cos(O1*k1 + m1)*cos(O3*k3 + m3)*cos(O4*k4 + m4))*cos(O5*k5 + m5) - sin(O3*k3 + m3)*sin(O5*k5 + m5)*cos(O1*k1 + m1)]])
        return math.Pose(math.Quaternion.fromMatrix(rot).normalize(), trans)



class FWKinematicPhantom2(object):

    def __init__(self, joint_lengths, jointangle_correction, gimbalangle_correction, origin_calib=None, disable_theta6=False):
        self.joint_lengths = joint_lengths
        self.jointangle_correction_factors = jointangle_correction
        self.gimbalangle_correction_factors = gimbalangle_correction
        if disable_theta6:
            self.gimbalangle_correction_factors[2, 0] = 0.0
            self.gimbalangle_correction_factors[2, 1] = 0.0
            self.gimbalangle_correction_factors[2, 2] = 0.0
        self.origin_calib = origin_calib if origin_calib is not None \
            else np.array([0., 0., 0.])

    def calculate_position(self, joint_angles):
        jacf = self.jointangle_correction_factors
        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        calx = self.origin_calib[0]
        caly = self.origin_calib[1]
        calz = self.origin_calib[2]

        O1_ = joint_angles[0]
        O2_ = joint_angles[1]
        O3_ = joint_angles[2]

        j1 = jacf[0, 0]
        k1 = jacf[0, 1]
        m1 = jacf[0, 2]
        j2 = jacf[1, 0]
        k2 = jacf[1, 1]
        m2 = jacf[1, 2]
        j3 = jacf[2, 0]
        k3 = jacf[2, 1]
        m3 = jacf[2, 2]

        O1 = j1*O1_**2 + O1_ * k1 + m1
        O2 = j2*O2_**2 + O2_ * k2 + m2
        O3 = j3*O3_**2 + O3_ * k3 + m3

        sO1, sO2, sO3 = sin(O1), sin(O2), sin(O3)
        cO1, cO2, cO3 = cos(O1), cos(O2), cos(O3)


        # calculate translation
        return np.array(
            [calx - sO1*(cO2*l1 + l2*sO3),
             -cO3*l2 + caly + l1*sO2 + l2,
             cO1*(cO2*l1 + l2*sO3) + calz - l1]
        )

    def calculate_pose(self, joint_angles, gimbal_angles):
        jacf = self.jointangle_correction_factors
        gacf = self.gimbalangle_correction_factors
        l1 = self.joint_lengths[0]
        l2 = self.joint_lengths[1]

        calx = self.origin_calib[0]
        caly = self.origin_calib[1]
        calz = self.origin_calib[2]

        O1_ = joint_angles[0]
        O2_ = joint_angles[1]
        O3_ = joint_angles[2]
        O4_ = gimbal_angles[0]
        O5_ = gimbal_angles[1]
        O6_ = gimbal_angles[2]

        j1 = jacf[0, 0]
        k1 = jacf[0, 1]
        m1 = jacf[0, 2]
        j2 = jacf[1, 0]
        k2 = jacf[1, 1]
        m2 = jacf[1, 2]
        j3 = jacf[2, 0]
        k3 = jacf[2, 1]
        m3 = jacf[2, 2]
        j4 = gacf[0, 0]
        k4 = gacf[0, 1]
        m4 = gacf[0, 2]
        j5 = gacf[1, 0]
        k5 = gacf[1, 1]
        m5 = gacf[1, 2]
        j6 = gacf[2, 0]
        k6 = gacf[2, 1]
        m6 = gacf[2, 2]

        O1 = j1*O1_**2 + O1_ * k1 + m1
        O2 = j2*O2_**2 + O2_ * k2 + m2
        O3 = j3*O3_**2 + O3_ * k3 + m3
        O4 = j4*O4_**2 + O4_ * k4 + m4
        O5 = j5*O5_**2 + O5_ * k5 + m5
        O6 = j6*O6_**2 + O6_ * k6 + m6

        sO1, sO2, sO3 = sin(O1), sin(O2), sin(O3)
        sO4, sO5, sO6 = sin(O4), sin(O5), sin(O6)

        cO1, cO2, cO3 = cos(O1), cos(O2), cos(O3)
        cO4, cO5, cO6 = cos(O4), cos(O5), cos(O6)


        # calculate translation
        trans = np.array(
            [calx - sO1*(cO2*l1 + l2*sO3),
             -cO3*l2 + caly + l1*sO2 + l2,
             cO1*(cO2*l1 + l2*sO3) + calz - l1]
        )

        # calculate rotation of arm
        rot = np.array([[cO6*(cO1*cO4 + cO3*sO1*sO4) + sO6*(-cO5*sO1*sO3 - sO5*(-cO1*sO4 + cO3*cO4*sO1)),
                         cO6*(cO5*sO1*sO3 + sO5*(-cO1*sO4 + cO3*cO4*sO1)) + sO6*(cO1*cO4 + cO3*sO1*sO4),
                         cO5*(cO1*sO4 - cO3*cO4*sO1) + sO1*sO3*sO5],
                        [-cO6*sO3*sO4 + sO6*(-cO3*cO5 + cO4*sO3*sO5),
                         cO6*(cO3*cO5 - cO4*sO3*sO5) - sO3*sO4*sO6,
                         cO3*sO5 + cO4*cO5*sO3],
                        [cO6*(-cO1*cO3*sO4 + cO4*sO1) + sO6*(cO1*cO5*sO3 - sO5*(-cO1*cO3*cO4 - sO1*sO4)),
                         cO6*(-cO1*cO5*sO3 + sO5*(-cO1*cO3*cO4 - sO1*sO4)) + sO6*(-cO1*cO3*sO4 + cO4*sO1),
                         -cO1*sO3*sO5 + cO5*(cO1*cO3*cO4 + sO1*sO4)]])
        return math.Pose(math.Quaternion.fromMatrix(rot).normalize(), trans)




def from_config(root_directory, context,
                gimbalangle_correction=None, origin_offset=None,
                jointangle_correction=None, joint_length=None, disable_theta6=None,
                *args, **kwargs):
    from ubitrack.core import util

    log.info("Create FWK Instance with: %s, %s" % (jointangle_correction, gimbalangle_correction, ))
    calib_directory = os.path.join(root_directory, context.get("calib_directory"))

    ja_correction = util.readCalibMeasurementMatrix3x3(os.path.join(calib_directory, jointangle_correction).encode(sys.getfilesystemencoding())) if jointangle_correction else None
    ga_correction = util.readCalibMeasurementMatrix3x3(os.path.join(calib_directory, gimbalangle_correction).encode(sys.getfilesystemencoding())) if gimbalangle_correction else None
    j_length = np.array([float(v.strip()) for v in joint_length.split(',')]) if joint_length else None
    o_offset = np.array([float(v.strip()) for v in origin_offset.split(',')]) if origin_offset else np.array([0., 0., 0.])
    d_theta6 = disable_theta6.strip().lower() == 'true' if disable_theta6 else False

    print j_length, ja_correction, ga_correction, o_offset, d_theta6

    if j_length is not None and ja_correction is not None and ga_correction is not None:
        return FWKinematicPhantom(j_length, ja_correction, ga_correction, origin_calib=o_offset, disable_theta6=d_theta6)
    raise ValueError("Invalid configuration for FWKinematicPhantom")
