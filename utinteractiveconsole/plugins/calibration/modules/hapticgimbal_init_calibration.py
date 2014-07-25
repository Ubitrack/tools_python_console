__author__ = 'MVL'
import os
import logging
import numpy as np

log = logging.getLogger(__name__)

from atom.api import Event, Bool, Typed, observe
from enaml.qt import QtCore
from enaml.application import deferred_call

import enaml
with enaml.imports():
    from .views.hapticgimbal_init_calibration import HapticGimbalInitCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController
from utinteractiveconsole.plugins.calibration.hapticdevice.gimbal_initialization import CalculateZRefAxis


class BackgroundCalculationThread(QtCore.QThread):

    def __init__(self, task, ctrl):
        log.info("Init Background Calculation")
        super(BackgroundCalculationThread, self).__init__()
        self.task = task
        self.ctrl = ctrl

    def run(self):
        log.info("BackgroundCalculationThread.run()")
        deferred_call(self.set_is_working, True)
        try:
            self.task.run()
        except Exception, e:
            log.error("Error in BackgroundCalculationThread:")
            log.exception(e)
        finally:
            deferred_call(self.set_is_working, False)

    def set_is_working(self, v):
        self.ctrl.is_working = v


class HapticGimbalInitCalibrationController(CalibrationController):

    bgThread = Typed(BackgroundCalculationThread)

    is_working = Bool(False)

    def calculate_zaxis(self):
        wiz_cfg = self.wizard_state.config
        gbl_cfg = self.context.get("config")

        # file paths for reading config and recorded data
        calib_dir = os.path.expanduser(wiz_cfg.get("calibdir").strip())
        record_dir = os.path.expanduser(self.config.get("recorddir").strip())
        use_2ndorder = False
        # eventually enable 2nd-order again using a config entry .. but it's not done now..

        # XXX this should also be determined using a calibration step (component already exists in ubitrack)
        joint_lengths = None
        origin_offset = np.array([0.0, 0.0, 0.0])
        try:
            haptidevice_name = wiz_cfg.get("haptic_device").strip()
            hd_cfg = dict(gbl_cfg.items("ubitrack.devices.%s" % haptidevice_name))
            joint_lengths = np.array([float(hd_cfg["joint_length1"]), float(hd_cfg["joint_length2"]), ])
            origin_offset = np.array([float(hd_cfg["origin_offset_x"]),
                                      float(hd_cfg["origin_offset_y"]),
                                      float(hd_cfg["origin_offset_z"]),])
        except Exception, e:
            log.error("Error reading Haptic device configuration. Make sure, the configuration file is correct.")
            log.exception(e)

        config_ok = True
        if not os.path.isdir(calib_dir):
            log.error("Calibration directory not found: %s" % calib_dir)
            config_ok = False

        if not os.path.isdir(record_dir):
            log.error("Record directory not found: %s" % record_dir)
            config_ok = False


        if config_ok:
            cza = CalculateZRefAxis(record_dir, calib_dir, joint_lengths, origin_offset, use_2ndorder=use_2ndorder)
            self.bgThread = BackgroundCalculationThread(cza, self)
            self.bgThread.start()
        else:
            log.warn("Calibration or recorder directories not found.")


class HapticGimbalInitCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_widget_class(self):
        return HapticGimbalInitCalibrationPanel

    def get_controller_class(self):
        return HapticGimbalInitCalibrationController