__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.absolute_orientation_calibration import AbsoluteOrientationCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class AbsoluteOrientationCalibrationController(CalibrationController):
    pass

class AbsoluteOrientationCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_name(self):
        return "Absolute Orientation Calibration"

    def get_dependencies(self):
        return ["hapticworkspace_init_calibration", ]

    def get_widget_class(self):
        return AbsoluteOrientationCalibrationPanel

    def get_controller_class(self):
        return AbsoluteOrientationCalibrationController