__author__ = 'MVL'
import os
import yaml
import logging
import enaml
with enaml.imports():
    from .views.calibration_start import CalibrationStartPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

log = logging.getLogger(__name__)

class CalibrationStartController(CalibrationController):
    save_results = False

class CalibrationStartModule(ModuleBase):

    def get_category(self):
        return "Initialization"

    def get_name(self):
        return "Calibration Start"

    def get_dependencies(self):
        return []

    def get_widget_class(self):
        return CalibrationStartPanel

    def get_controller_class(self):
        return CalibrationStartController