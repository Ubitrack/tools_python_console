__author__ = 'MVL'
import numpy as np
from numpy.linalg import norm
import logging

from atom.api import Bool, Value, Typed, Int, Float
import enaml
with enaml.imports():
    from .views.tooltip_calibration import TooltipCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

log = logging.getLogger(__name__)

class TooltipCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    result_count = Int(128)
    errors = Typed(np.ndarray)
    max_error = Float(0.0)
    initial_error = Float(-1)

    last_result = Value(None)

    results_txt = Value()
    progress_bar = Value()

    def setupController(self, active_widgets=None):
        super(TooltipCalibrationController, self).setupController(active_widgets=active_widgets)
        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')
            self.progress_bar = w.find('progress_bar')

        if self.autocomplete_maxerror_str != "":
            self.max_error = float(self.autocomplete_maxerror_str)

        # needs to match the SRG !!
        self.sync_source = 'calib_tooltip'
        self.required_sinks = ['calib_tooltip',]

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
        if self.connector.calib_tooltip is not None:
            tc = self.connector.calib_tooltip.get()
            self.results_txt.text = "Result:\n%s" % str(tc)

            if self.last_result is not None:
                error = norm(tc - self.last_result)
                self.errors[0] = error
                # implements simple ringbuffer
                self.errors = np.roll(self.errors, 1)

                if self.initial_error == -1:
                    self.initial_error = error

            self.last_result = tc

            # update progress bar
            if self.initial_error != -1:
                p = error/(self.initial_error - self.max_error)
                pv = int(np.sqrt(1 - max(0, min(p, 1)))*100)
                if pv > self.progress_bar.value:
                    self.progress_bar.value = pv

            # check if the minimum of self.result_count results have been received
            if not np.isnan(np.sum(self.errors)):

                if np.all(self.errors < self.max_error):
                    log.info("Tooltip Calibration: Results are satisfactory (<%s) min: %s max: %s" %
                             (self.max_error, np.min(self.errors), np.max(self.errors)))
                    self.result_ok = True
                    self.progress_bar.value = 100
                    if self.autocomplete_enable:
                        self.stopCalibration()









class TooltipCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_widget_class(self):
        return TooltipCalibrationPanel

    def get_controller_class(self):
        return TooltipCalibrationController