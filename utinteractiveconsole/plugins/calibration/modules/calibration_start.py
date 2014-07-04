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
    show_facade_controls = False

    domain_name = Str()
    setup_name = Str()
    user_name = Str()
    platform_name = Str()


    def _default_domain_name(self):
        cfg = self.wizard_state.context.get('config')
        if cfg is not None and cfg.has_option("calibration_wizard", "domain"):
            return cfg.get("calibration_wizard", "domain")
        return "ubitrack.local"

    def _default_setup_name(self):
        cfg = self.wizard_state.context.get('config')
        if cfg is not None and cfg.has_option("calibration_wizard", "setup"):
            return cfg.get("calibration_wizard", "setup")
        return "default"

    def _default_user_name(self):
        cfg = self.wizard_state.context.get('config')
        if cfg is not None and cfg.has_option("calibration_wizard", "user"):
            return cfg.get("calibration_wizard", "user")
        return "default"

    def _default_platform_name(self):
        cfg = self.wizard_state.context.get('config')
        if cfg is not None and cfg.has_option("calibration_wizard", "platform"):
            return cfg.get("calibration_wizard", "platform")
        return sys.platform

    def setupController(self, active_widgets=None):
        super(CalibrationStartController, self).setupController(active_widgets=active_widgets)
        self.wizard_state.calibration_domain_name = self.domain_name
        self.wizard_state.calibration_setup_name = self.setup_name
        self.wizard_state.calibration_user_name = self.user_name
        self.wizard_state.calibration_platform_name = self.platform_name

    def teardownController(self, active_widgets=None):
        if self.wizard_state.calibration_existing_delete_files:
            for m in self.wizard_state.module_manager.modules.values():
                if m.is_enabled():
                    cfs = m.get_calib_files()
                    for cf in cfs:
                        if os.path.isfile(cf):
                            log.info("Deleting calibration file: %s" % cf)
                            os.unlink(cf)

        if self.preview_controller is not None:
            self.preview_controller.setupPreview()


class CalibrationStartModule(ModuleBase):

    def get_category(self):
        return "Generic"

    def get_dependencies(self):
        return []

    def get_widget_class(self):
        return CalibrationStartPanel

    def get_controller_class(self):
        return CalibrationStartController