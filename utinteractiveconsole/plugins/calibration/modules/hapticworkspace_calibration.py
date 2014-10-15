__author__ = 'MVL'
import os
import logging
import numpy as np

log = logging.getLogger(__name__)

from atom.api import Event, Bool, Typed, Value, observe, Int, Float
from enaml.qt import QtCore
from enaml.application import deferred_call

import enaml
with enaml.imports():
    from .views.hapticworkspace_calibration import HapticWorkspaceCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController


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



class HapticWorkspaceCalibrationController(CalibrationController):

    bgThread = Typed(BackgroundCalculationThread)
    is_working = Bool(False)


    is_ready = Bool(False)

    result_count = Int(128)
    errors = Typed(np.ndarray)
    max_error = Float(0.0)
    initial_error = Float(-1)

    last_result = Value(None)

    results_txt = Value()
    progress_bar = Value()

    def setupController(self, active_widgets=None):
        super(HapticWorkspaceCalibrationController, self).setupController(active_widgets=active_widgets)
        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')
            self.progress_bar = w.find('progress_bar')

        if self.autocomplete_maxerror_str != "":
            self.max_error = float(self.autocomplete_maxerror_str)

        # needs to match the SRG !!
        self.sync_source = 'calib_phantom_jointangle_correction'
        self.required_sinks = ['calib_phantom_jointangle_correction',]

        # setup a errors buffer
        self.errors = np.array([np.nan] * self.result_count, dtype=np.double)

        if self.facade is not None:
            self.facade.observe("is_loaded", self.connector_setup)

    def connector_setup(self, change):
        if change['value'] and self.verify_connector():
            self.connector.setup(self.facade.instance)
            self.connector.observe(self.sync_source, self.handle_data)
            self.is_ready = True

    def handle_data(self, c):
        if self.connector.calib_phantom_jointangle_correction is not None:
            cf = self.connector.calib_phantom_jointangle_correction.get()

            # do a bounds check:
            results_within_bounds = True
            k_cf = cf[:,1]
            m_cf = cf[:,2]
            # factors should be between 0.8 and 1.2 (phantom omni/premiums sensible defaults, should be configurable)
            if not (np.all(k_cf > 0.8) and np.all(k_cf < 1.2)):
                results_within_bounds = False
            # offsets should be between -0.2 and 0.2 (phantom omni/premiums sensible defaults, should be configurable)
            if not (np.all(m_cf > -0.2) and np.all(m_cf < 0.2)):
                results_within_bounds = False

            if results_within_bounds:
                self.results_txt.text = "Result:\n%s" % str(cf)
            else:
                self.results_txt.text = "WARNING: Results out of bounds!!\nPlease restart the calibration step.\nResult:\n%s" % str(cf)

            if self.last_result is not None:
                error = np.max(np.abs(cf.flatten() - self.last_result.flatten()))
                self.errors[0] = error
                # implements simple ringbuffer
                self.errors = np.roll(self.errors, 1)

                if self.initial_error == -1:
                    self.initial_error = error

            self.last_result = cf

            # update progress bar
            if self.initial_error != -1:
                p = error/(self.initial_error - self.max_error)
                pv = int(np.sqrt(1 - max(0, min(p, 1)))*100)
                if pv > self.progress_bar.value:
                    self.progress_bar.value = pv

            # check if the minimum of self.result_count results have been received
            if not np.isnan(np.sum(self.errors)):

                if np.all(self.errors < self.max_error):
                    log.info("Joint-Angle Correction: Results are satisfactory (<%s) min: %s max: %s" %
                             (self.max_error, np.min(self.errors), np.max(self.errors)))
                    self.result_ok = True
                    self.progress_bar.value = 100
                    if self.autocomplete_enable:
                        self.stopCalibration()







    def refine_workspace_calibration(self):
        wiz_cfg = self.wizard_state.config
        gbl_cfg = self.context.get("config")

        components_path = gbl_cfg.get("ubitrack", "components_path")

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
            raise "Do Your Homework Properly !!!"
            wcr = WorkspaceCalibrationRefinement(record_dir, calib_dir, joint_lengths, origin_offset,
                                                 use_2ndorder=use_2ndorder, components_path=components_path)
            self.bgThread = BackgroundCalculationThread(wcr, self)
            self.bgThread.start()
        else:
            log.warn("Calibration or recorder directories not found.")





class HapticWorkspaceCalibrationModule(ModuleBase):

    def get_category(self):
        return "Devices"

    def get_widget_class(self):
        return HapticWorkspaceCalibrationPanel

    def get_controller_class(self):
        return HapticWorkspaceCalibrationController