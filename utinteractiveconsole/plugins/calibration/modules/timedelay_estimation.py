__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.timedelay_estimation import TimeDelayEstimationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class TimeDelayEstimationController(CalibrationController):
    pass

class TimeDelayEstimationModule(ModuleBase):

    def get_category(self):
        return "Synchronization"

    def get_widget_class(self):
        return TimeDelayEstimationPanel

    def get_controller_class(self):
        return TimeDelayEstimationController