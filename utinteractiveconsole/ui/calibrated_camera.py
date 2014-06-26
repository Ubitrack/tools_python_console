__author__ = 'jack'
import numpy as np
from atom.api import Typed, Int, observe
from enaml.core.declarative import d_
from enaml_opengl.camera import Camera

from ubitrack.core import calibration


class CalibratedCamera(Camera):

    camera_width = d_(Int(640))
    camera_height = d_(Int(480))

    camera_intrinsics = Typed(np.ndarray)


    @observe("camera_intrinsics")
    def _update_intrinsics(self, change):
        if self.camera_intrinsics is not None:
            self.projection_matrix = calibration.projectionMatrix3x3ToOpenGL(0., self.camera_width, 0., self.camera_height, self.near, self.far, self.camera_intrinsics)
            # self.trigger_update()
