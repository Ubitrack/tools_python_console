__author__ = 'MVL'
import numpy as np
from numpy.linalg import norm
import logging

from atom.api import Bool, Value, Typed, Int, Float

import enaml
with enaml.imports():
    from .views.absolute_orientation_calibration import AbsoluteOrientationCalibrationPanel

from ubitrack.core import math
from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

log = logging.getLogger(__name__)


class AbsoluteOrientationCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    result_count = Int(128)
    errors_translation = Typed(np.ndarray)
    errors_rotation = Typed(np.ndarray)

    max_error_translation = Float(0.0)
    max_error_rotation = Float(0.0)

    initial_error_translation = Float(-1)
    initial_error_rotation = Float(-1)

    last_result = Value(None)

    results_txt = Value()
    progress_bar = Value()

    def setupController(self, active_widgets=None):
        super(AbsoluteOrientationCalibrationController, self).setupController(active_widgets=active_widgets)
        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')
            self.progress_bar = w.find('progress_bar')

        if self.autocomplete_maxerror_str != "":
            translation, rotation = [s.strip() for s in self.autocomplete_maxerror_str.split(",")]
            self.max_error_translation = float(translation)
            self.max_error_rotation = float(rotation)

        # needs to match the SRG !!
        self.sync_source = 'calib_absolute_orientation'
        self.required_sinks = ['calib_absolute_orientation',]

        # setup a errors buffer
        self.errors_translation = np.array([np.nan] * self.result_count, dtype=np.double)
        self.errors_rotation = np.array([np.nan] * self.result_count, dtype=np.double)

        if self.facade is not None:
            self.facade.observe("is_loaded", self.connector_setup)

    def connector_setup(self, change):
        if change['value'] and self.verify_connector():
            self.connector.setup(self.facade.instance)
            self.connector.observe(self.sync_source, self.handle_data)
            self.is_ready = True

    def handle_data(self, c):
        if self.connector.calib_absolute_orientation is not None:
            ao = self.connector.calib_absolute_orientation.get()
            self.results_txt.text = "Result:\n%s" % str(ao)

            if self.last_result is not None:
                t_error = norm(ao.translation() - self.last_result.translation())
                self.errors_translation[0] = t_error
                # implement simple ringbuffer
                self.errors_translation = np.roll(self.errors_translation, 1)

                if self.initial_error_translation == -1:
                    self.initial_error_translation = t_error

                r_error = abs(math.Quaternion(ao.rotation().inverted() * self.last_result.rotation()).angle())
                self.errors_rotation[0] = r_error
                # implement simple ringbuffer
                self.errors_rotation = np.roll(self.errors_rotation, 1)

                if self.initial_error_rotation == -1:
                    self.initial_error_rotation = r_error

            self.last_result = ao

            # update progress bar
            if self.initial_error_translation != -1 and self.initial_error_translation != -1:
                t_p = t_error/(self.initial_error_translation - self.max_error_translation)
                r_p = r_error/(self.initial_error_rotation - self.max_error_rotation)
                pv = int(np.sqrt(1 - max(0, min(max(t_p, r_p), 1)))*100)
                if pv > self.progress_bar.value:
                    self.progress_bar.value = pv

            # check if the minimum of self.result_count results have been received
            if not np.isnan(np.sum(self.errors_translation)) and not np.isnan(np.sum(self.errors_rotation)):
                if np.all(self.errors_translation < self.max_error_translation) and \
                        np.all(self.errors_rotation < self.max_error_rotation):
                    log.info("Absolute Orientation: Results are satisfactory for translation (<%s) min: %s max: %s and rotation (<%s) min: %s max %s" %
                             (self.max_error_translation, np.min(self.errors_translation), np.max(self.errors_translation),
                              self.max_error_rotation, np.min(self.errors_rotation), np.max(self.errors_rotation)))
                    self.result_ok = True
                    self.progress_bar.value = 100
                    if self.autocomplete_enable:
                        self.stopCalibration()





class AbsoluteOrientationCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_widget_class(self):
        return AbsoluteOrientationCalibrationPanel

    def get_controller_class(self):
        return AbsoluteOrientationCalibrationController