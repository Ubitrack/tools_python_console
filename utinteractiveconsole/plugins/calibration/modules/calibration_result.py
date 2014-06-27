__author__ = 'MVL'
import os
import yaml
import logging
import enaml
with enaml.imports():
    from .views.calibration_result import CalibrationResultPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

log = logging.getLogger(__name__)

class CalibrationResultController(CalibrationController):

    def saveResults(self, root_dir):
        wizard_state = self.wizard_state
        extra_files = []
        if "report_filename" in self.config:
            calib_info = dict(
                setup=wizard_state.calibration_setup,
                operator=wizard_state.calibration_operator,
                datetime=wizard_state.calibration_datetime,
                comments=wizard_state.calibration_comments,
                dataok=wizard_state.calibration_dataok,
                results=[t.to_dict() for t in self.wizard_state.tasks],
            )
            fname = os.path.join(self.data_dir, self.config["report_filename"])
            yaml.dump(calib_info, open(fname, "w"), default_flow_style=False)
            extra_files.append(fname)
            log.info("Saved calibration report: %s" % fname)
        else:
            log.error("Missing report_filename in section [vharcalib.module.calibration_result]")
        super(CalibrationResultController, self).saveResults(root_dir, extra_files=extra_files)

class CalibrationResultModule(ModuleBase):

    def get_category(self):
        return "Summary"

    def get_name(self):
        return "Calibration Result"

    def get_dependencies(self):
        return ["hapticgimbal_calibration",]

    def get_widget_class(self):
        return CalibrationResultPanel

    def get_controller_class(self):
        return CalibrationResultController