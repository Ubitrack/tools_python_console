__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.hapticgimbal_calibration import HapticGimbalCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class HapticGimbalCalibrationController(CalibrationController):
    pass

class HapticGimbalCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_name(self):
        return "Haptic Gimbal Calibration"

    def get_dependencies(self):
        return ["hapticgimbal_init_calibration",]

    def get_widget_class(self):
        return HapticGimbalCalibrationPanel

    def get_controller_class(self):
        return HapticGimbalCalibrationController