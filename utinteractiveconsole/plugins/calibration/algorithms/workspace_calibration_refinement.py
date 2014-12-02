__author__ = 'jack'
from utinteractiveconsole.util import deprecate_module
deprecate_module()

import logging
import time
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
                                           interpolateVec3List, selectOnlyMatchingSamples,
                                           selectNearestNeighbour)

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

    def __init__(self, record_dir, calib_dir, joint_lengths, origin_offset,
                 use_2ndorder=False, components_path="/usr/local/lib/ubitrack"):
        self.record_dir = record_dir
        self.calib_dir = calib_dir
        self.calibrations = self.load_calibrations()
        self.joint_lengths = joint_lengths
        self.origin_offset = origin_offset
        self.use_2ndorder = use_2ndorder
        self.facade = facade.AdvancedFacade(components_path)

        # pushed data outputs
        self.calib_absolute_orientation_out = None

        # pulled data inputs
        self.angle_calib_hip_positions = []
        self.angle_calib_jointangles = []



    def setup_facade(self, dfg_filename):
        self.facade.loadDataflow(dfg_filename, True)

        # Push sinks
        self.cmp_caoo = self.facade.getApplicationPushSinkPose("calib_absolute_orientation_out")
        self.cmp_caoo.setCallback(self.consume_calib_absolite_orientation_out)

        # Pull sinks
        self.cmp_phc = self.facade.getApplicationPullSinkMatrix3x3("calib_phantom_jointangle_correction")

        # Push sources
        self.cmp_acfp = self.facade.getApplicationPushSourcePositionList("ao_calib_fwk_positions")
        self.cmp_acep = self.facade.getApplicationPushSourcePositionList("ao_calib_et_positions")

        # Pull sources
        self.cmp_chp = self.facade.getApplicationPullSourcePositionList("angle_calib_hip_positions")
        self.cmp_chp.setCallback(self.produce_angle_calib_hip_positions)

        self.cmp_cja = self.facade.getApplicationPullSourcePositionList("angle_calib_jointangles")
        self.cmp_cja.setCallback(self.produce_angle_calib_jointangles)

        self.facade.startDataflow()


    # Ubitrack callbacks
    def consume_calib_absolite_orientation_out(self, m):
        print "consume_calib_absolute_orientation_out", m
        self.calib_absolute_orientation_out = m.get()

    def produce_angle_calib_hip_positions(self, ts):
        print "produce_angle_calib_hip_positions", ts
        pl = math.PositionList.fromList(self.angle_calib_hip_positions)
        return measurement.PositionList(ts, pl)

    def produce_angle_calib_jointangles(self, ts):
        print "produce_angle_calib_jointangles", ts
        pl = math.PositionList.fromList(self.angle_calib_jointangles)
        return measurement.PositionList(ts, pl)


    def get_fwk(self, jointangle_calib=None, gimbalangle_calib=None, disable_theta6=False):
        if jointangle_calib is None:
            jointangle_calib = self.calibrations["phantom_jointangle_calib"]
        if gimbalangle_calib is None:
            gimbalangle_calib = self.calibrations["phantom_gimbalangle_calib"]

        if self.use_2ndorder:
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

    def load_data(self):
        return loadData(
                self.record_dir,
                DSC('externaltracker_pose', 'optitrack_pose.log',
                    util.PoseStreamReader),
                items=(DSC('jointangles', 'haptic_joint_angles.log',
                           util.PositionStreamReader, interpolateVec3List),
                       )
                )

    def load_data_gated(self):
        return loadData(
                self.record_dir,
                DSC('jointangles', 'haptic_joint_angles_gated.log',
                    util.PositionStreamReader),
                items=(DSC('externaltracker_position', 'optitrack_hip_position_gated.log',
                           util.PositionStreamReader, interpolateVec3List),
                       )
                )


    def load_calibrations(self):
        return loadCalibrationFiles(self.calib_dir)


    def run(self):
        print "TBD"