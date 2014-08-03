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
from ubitrack.facade import facade
from utinteractiveconsole.playback import (loadData, DSC, interpolatePoseList,
                                           interpolateVec3List, selectOnlyMatchingSamples)

from .phantom_forward_kinematics import FWKinematicPhantom, FWKinematicPhantom2

log = logging.getLogger(__name__)


def loadCalibrationFiles(root_dir):
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


class WorkspaceCalibrationRefinement(object):

    def __init__(self, record_dir, calib_dir, joint_lengths, origin_offset, use_2ndorder=False):
        self.record_dir = record_dir
        self.calib_dir = calib_dir
        self.calibrations = self.load_calibrations()
        self.joint_lengths = joint_lengths
        self.origin_offset = origin_offset
        self.use_2ndorder = use_2ndorder
        self.facade = facade.AdvancedFacade()

    def setup_facade(self, dfg_filename):
        self.facade.loadDataflow(dfg_filename, True)
        self.facade.startDataflow()


    def get_fwk(self, jointangle_calib=None, gimbalangle_calib=None, disable_theta6=False):
        if jointangle_calib is None:
            jointangle_calib = self.calibrations["phantom_jointangle_calib"]
        if gimbalangle_calib is None:
            gimbalangle_calib = self.calibrations["phantom_gimbalangle_calib"]

        if use_2ndorder:
            return FWKinematicPhantom2(self.joint_lengths,
                                           jointangle_calib,
                                           gimbalangle_calib,
                                           self.origin_offset,
                                           disable_theta6=disable_theta6)
        else:
            return FWKinematicPhantom(self.joint_lengths,
                                          jointangle_calib,
                                          gimbalangle_calib,
                                          self.origin_offset,
                                          disable_theta6=disable_theta6)




    def compute_et_hip_positions(self):
        pass
