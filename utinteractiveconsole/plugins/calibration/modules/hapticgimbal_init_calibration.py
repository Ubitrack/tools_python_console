__author__ = 'MVL'
import os
import logging

log = logging.getLogger(__name__)

import enaml
with enaml.imports():
    from .views.hapticgimbal_init_calibration import HapticGimbalInitCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

try:
    from vharcalib.gimbal_initialization import CalculateZRefAxis
except ImportError:
    def CalculateZRefAxis(data_dir, calib_dir, joint_lengths, use_2ndorder=False):
        raise NotImplementedError("needs to be implemented !!!")

class HapticGimbalInitCalibrationController(CalibrationController):

    def calculate_zaxis(self):
            cfg = self.context.get("config")
            root_dir = None
            calib_dir = None
            data_dir = None
            use_2ndorder = False
            joint_lengths = None
            if cfg.has_section("vharcalib"):
                vc_cfg = dict(cfg.items("vharcalib"))
                root_dir = vc_cfg["rootdir"]
                calib_dir = os.path.join(root_dir, vc_cfg["datadir"])
                if "use_2ndorder" in vc_cfg:
                    use_2ndorder = cfg.getboolean("vharcalib", "use_2ndorder")
            else:
                log.error("Missing section: [vharcalib] in config")

            if "recorddir" in self.config:
                data_dir = os.path.join(calib_dir, self.config["recorddir"])
            else:
                log.error("Missing recorddir entry in module config [vharcalib.module.hapticgimbal_init_calibration]")

            if cfg.has_section("vharcalib.devices.phantom"):
                vcph_cfg = dict(cfg.items("vharcalib.devices.phantom"))
                joint_lengths = map(float, [vcph_cfg["joint_length1"], vcph_cfg["joint_length2"]])
            else:
                log.error("Missing section: [vharcalib.devices.phantom] in config")


            if os.path.isdir(calib_dir) and os.path.isdir(data_dir):
                cza = CalculateZRefAxis(data_dir, calib_dir, joint_lengths, use_2ndorder=use_2ndorder)
                cza.run()
            else:
                log.warn("Calibration or recorder directories not found.")


class HapticGimbalInitCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_name(self):
        return "Haptic Gimbal Initial Calibration"

    def get_dependencies(self):
        return ["hapticworkspace_calibration",]

    def get_widget_class(self):
        return HapticGimbalInitCalibrationPanel

    def get_controller_class(self):
        return HapticGimbalInitCalibrationController