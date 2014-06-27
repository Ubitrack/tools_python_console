__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.hapticworkspace_init_calibration import HapticWorkspaceInitCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class HapticWorkspaceInitCalibrationController(CalibrationController):
    pass

class HapticWorkspaceInitCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_name(self):
        return "Haptic Workspace Initial Calibration"

    def get_dependencies(self):
        return ["timedelay_estimation_calibration",]

    def get_widget_class(self):
        return HapticWorkspaceInitCalibrationPanel

    def get_controller_class(self):
        return HapticWorkspaceInitCalibrationController