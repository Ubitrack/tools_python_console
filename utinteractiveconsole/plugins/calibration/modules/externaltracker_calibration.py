__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.externaltracker_calibration import ExternalTrackerCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class ExternalTrackerCalibrationController(CalibrationController):
    pass

class ExternalTrackerCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_name(self):
        return "External Tracker Calibration"

    def get_dependencies(self):
        return ["calibration_start", ]

    def get_widget_class(self):
        return ExternalTrackerCalibrationPanel

    def get_controller_class(self):
        return ExternalTrackerCalibrationController