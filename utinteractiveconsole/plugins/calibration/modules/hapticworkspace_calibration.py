__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.hapticworkspace_calibration import HapticWorkspaceCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class HapticWorkspaceCalibrationController(CalibrationController):
    pass

class HapticWorkspaceCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_widget_class(self):
        return HapticWorkspaceCalibrationPanel

    def get_controller_class(self):
        return HapticWorkspaceCalibrationController