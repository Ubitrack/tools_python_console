__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.hapticdevice_calibration import HapticDeviceCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class HapticDeviceCalibrationController(CalibrationController):
    pass

class HapticDeviceCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_name(self):
        return "Haptic Device Calibration"

    def get_dependencies(self):
        return ["calibration_start", ]

    def get_widget_class(self):
        return HapticDeviceCalibrationPanel

    def get_controller_class(self):
        return HapticDeviceCalibrationController