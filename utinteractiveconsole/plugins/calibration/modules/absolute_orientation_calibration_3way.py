__author__ = 'MVL'
import numpy as np
from numpy.linalg import norm
import logging

from atom.api import Bool, Value, Typed, Int, Float

import enaml
with enaml.imports():
    from .views.absolute_orientation_calibration_3way import AbsoluteOrientationCalibrationPanel

from ubitrack.core import math
from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

log = logging.getLogger(__name__)


class AbsoluteOrientationCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)
    results_txt = Value()

    def setupController(self, active_widgets=None):
        super(AbsoluteOrientationCalibrationController, self).setupController(active_widgets=active_widgets)
        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')
            #self.progress_bar = w.find('progress_bar')

        # XXX TODO Make interactive ..
        # needs to match the SRG !!
        #self.sync_source = 'calib_absolute_orientation1'
        #self.required_sinks = ['calib_absolute_orientation1','calib_absolute_orientation2','calib_absolute_orientation3']

        #if self.facade is not None:
        #    self.facade.observe("is_loaded", self.connector_setup)

    def connector_setup(self, change):
        if change['value'] and self.verify_connector():
            self.connector.setup(self.facade.instance)
            #self.connector.observe(self.sync_source, self.handle_data)
            #self.is_ready = True

    def handle_data(self, c):
        results = []
        if self.connector.calib_absolute_orientation1 is not None:
            ao = self.connector.calib_absolute_orientation1.get()
            results.append("Result 1:\n%s" % str(ao))
        if self.connector.calib_absolute_orientation2 is not None:
            ao = self.connector.calib_absolute_orientation2.get()
            results.append("Result 2:\n%s" % str(ao))
        if self.connector.calib_absolute_orientation3 is not None:
            ao = self.connector.calib_absolute_orientation3.get()
            results.append("Result 3:\n%s" % str(ao))

        self.results_txt.text = "\n".join(results)


class AbsoluteOrientationCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_widget_class(self):
        return AbsoluteOrientationCalibrationPanel

    def get_controller_class(self):
        return AbsoluteOrientationCalibrationController