__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.camera_handeye import CameraHandEyeCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class CameraHandEyeCalibrationController(CalibrationController):

    def setupController(self):
        print "setup camera handeye controller to connect the button to the dfg"

    def capturePose(self):
        print "Capture a pose and do something with it.."


class CameraHandEyeCalibrationModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_name(self):
        return "Camera HandEye"

    def get_dependencies(self):
        return ['camera_intrinsics',]

    def get_widget_class(self):
        return CameraHandEyeCalibrationPanel

    def get_controller_class(self):
        return CameraHandEyeCalibrationController