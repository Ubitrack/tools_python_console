__author__ = 'jack'

from ubitrack.core import math
import numpy as np


def position3d_to_pose(pos):
    return math.Pose(math.Quaternion(), pos)


def vector_from_config(root_directory, context, value=None):
    if value is not None:
        return np.fromstring(value, sep=',')