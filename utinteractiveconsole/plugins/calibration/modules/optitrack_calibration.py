__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.optitrack_calibration import OptitrackCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class OptitrackCalibrationController(CalibrationController):
    pass

class OptitrackCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_name(self):
        return "External Tracker Calibration"

    def get_dependencies(self):
        return ["calibration_start", ]

    def get_widget_class(self):
        return OptitrackCalibrationPanel

    def get_controller_class(self):
        return OptitrackCalibrationController