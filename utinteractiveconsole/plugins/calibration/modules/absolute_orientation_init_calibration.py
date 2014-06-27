__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.absolute_orientation_init_calibration import AbsoluteOrientationInitCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class AbsoluteOrientationInitCalibrationController(CalibrationController):
    pass

class AbsoluteOrientationInitCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_name(self):
        return "Absolute Orientation Initial Calibration"

    def get_dependencies(self):
        return ["tooltip_calibration", ]

    def get_widget_class(self):
        return AbsoluteOrientationInitCalibrationPanel

    def get_controller_class(self):
        return AbsoluteOrientationInitCalibrationController