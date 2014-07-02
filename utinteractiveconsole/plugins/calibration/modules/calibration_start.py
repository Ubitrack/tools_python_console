__author__ = 'MVL'
import os
import sys
import yaml
import logging
from atom.api import Str, Dict

import enaml
with enaml.imports():
    from .views.calibration_start import CalibrationStartPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

log = logging.getLogger(__name__)

class CalibrationStartController(CalibrationController):
    save_results = False

    ubitrack_config = Dict()

    domain_name = Str()
    setup_name = Str()
    user_name = Str()
    platform_name = Str()

    def _default_ubitrack_config(self):
        cfg = self.wizard_state.context.get('config')
        if cfg is not None and cfg.has_section('ubitrack'):
            return dict(cfg.items('ubitrack'))
        return {}

    def _default_domain_name(self):
        return self.ubitrack_config.get("domain", "ubitrack.local")

    def _default_setup_name(self):
        return self.ubitrack_config.get("setup", "default")

    def _default_user_name(self):
        return self.ubitrack_config.get("user", "default")

    def _default_platform_name(self):
        return self.ubitrack_config.get("platform", sys.platform)



class CalibrationStartModule(ModuleBase):

    def get_category(self):
        return "Generic"

    def get_name(self):
        return "Calibration Start"

    def get_dependencies(self):
        return []

    def get_widget_class(self):
        return CalibrationStartPanel

    def get_controller_class(self):
        return CalibrationStartController