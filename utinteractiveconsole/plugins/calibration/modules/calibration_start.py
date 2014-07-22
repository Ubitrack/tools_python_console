__author__ = 'MVL'
import os
import shutil
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
            cfg = self.context.get("config")
            sname = "%s.initialize_files" % (self.config_ns,)
            init_files = {}
            if cfg.has_section(sname):
                init_files = dict(cfg.items(sname))

            for m in self.wizard_state.module_manager.modules.values():
                if m.is_enabled():
                    cfs = m.get_calib_files()
                    for cf in cfs:
                        bb_cf = os.path.basename(cf)
                        if bb_cf in init_files:
                            if os.path.isfile(init_files[bb_cf]):
                                log.info("Set calibration file to default: %s" % bb_cf)
                                shutil.copyfile(init_files[bb_cf], cf)
                            else:
                                log.warn("Missing default calibration file: %s" % init_files[bb_cf])
                        elif os.path.isfile(cf):
                            log.info("Deleting calibration file: %s" % cf)
                            os.unlink(cf)


class CalibrationStartModule(ModuleBase):

    def get_category(self):
        return "Generic"

    def get_dependencies(self):
        return []

    def get_widget_class(self):
        return CalibrationStartPanel

    def get_controller_class(self):
        return CalibrationStartController