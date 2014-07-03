__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.tooltip_calibration import TooltipCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class TooltipCalibrationController(CalibrationController):
    pass

class TooltipCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_widget_class(self):
        return TooltipCalibrationPanel

    def get_controller_class(self):
        return TooltipCalibrationController