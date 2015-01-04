__author__ = 'jack'


from atom.api import Atom, List, Str, Int, Enum, Bool, Typed, Value, Coerced

import os
import logging

from ubitrack.core import util, math, measurement
import numpy as np
from .streamfile import UBITRACK_DATATYPES

log = logging.getLogger(__name__)


def read_calibfile(filename, dtype, is_array=False):
    reader = None
    if dtype == "distance":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
    elif dtype == "position2d":
        if is_array:
            reader = util.readCalibMeasurementPositionList2
        else:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
    elif dtype == "position3d":
        if is_array:
            reader = util.readCalibMeasurementPositionList
        else:
            reader = util.readCalibMeasurementPosition
    elif dtype == "quat":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            reader = util.readCalibMeasurementRotation
    elif dtype == "pose":
        if is_array:
            reader = util.readCalibMeasurementPoseList
        else:
            reader = util.readCalibMeasurementPose
    elif dtype == "mat33":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            reader = util.readCalibMeasurementMatrix3x3
    elif dtype == "mat34":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            reader = util.readCalibMeasurementMatrix3x4
    elif dtype == "mat44":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            reader = util.readCalibMeasurementMatrix4x4

    if reader is None:
        raise ValueError("Unknown datatype: %s" % dtype)

    return reader(filename)
