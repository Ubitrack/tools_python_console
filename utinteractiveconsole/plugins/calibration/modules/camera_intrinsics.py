__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.camera_intrinsics import CameraIntrinsicsCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class CameraIntrinsicsCalibrationController(CalibrationController):

    def setupController(self):
        print "setup camera intrinsics controller to connect the button to the dfg"

    def captureImage(self):
        print "Capture an Image from the Camera and do something with it.."


class CameraIntrinsicsCalibrationModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_name(self):
        return "Camera Intrinsics"

    def get_dependencies(self):
        return []

    def get_widget_class(self):
        return CameraIntrinsicsCalibrationPanel

    def get_controller_class(self):
        return CameraIntrinsicsCalibrationController