__author__ = 'MVL'
import numpy as np
from numpy.linalg import norm
import logging

from atom.api import Bool, Value, Typed, Int
import enaml
with enaml.imports():
    from .views.tooltip_calibration import TooltipCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

log = logging.getLogger(__name__)

class TooltipCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    result_ok = Bool(False)
    result_count = Int(128)
    results_buffer = Typed(np.ndarray)

    results_txt = Value()

    def setupController(self, active_widgets=None):
        super(TooltipCalibrationController, self).setupController(active_widgets=active_widgets)
        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = 'calib_tooltip'
        self.required_sinks = ['calib_tooltip',]

        # setup a resuls buffer
        self.results_buffer = np.array([np.nan] * self.result_count * 3, dtype=np.double)
        self.results_buffer.resize((self.result_count, 3))

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
            self.results_buffer[0] = tc
            # implement simple ringbuffer
            self.results_buffer = np.roll(self.results_buffer, 3)

            # check if the minimum of self.result_count results have been received
            if not np.isnan(np.sum(self.results_buffer)) and self.autocomplete_maxerror != "":
                max_error = float(self.autocomplete_maxerror)
                buffer = self.results_buffer
                errors = np.array([norm(buffer[i]-buffer[i+1]) for i in range(self.result_count-1)])
                if np.all(errors < max_error):
                    log.info("Tooltip Calibration: Results are satisfactory (<%s)" % self.autocomplete_maxerror)
                    self.result_ok = True
                    if self.autocomplete_enable:
                        self.stopCalibration()






class TooltipCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_widget_class(self):
        return TooltipCalibrationPanel

    def get_controller_class(self):
        return TooltipCalibrationController