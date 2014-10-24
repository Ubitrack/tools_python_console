__author__ = 'MVL'
import numpy as np

import enaml
with enaml.imports():
    from .views.tracker_handeye import TrackerHandEyeCalibrationPanel

from atom.api import Bool, Value, Enum

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

import logging
log = logging.getLogger(__name__)

class TrackerHandEyeCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)
    results_txt = Value()

    def setupController(self, active_widgets=None):
        super(TrackerHandEyeCalibrationController, self).setupController(active_widgets=active_widgets)

        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = 'calib_handeye'
        self.required_sinks = ['calib_handeye',]

        if self.facade is not None:
            self.facade.observe("is_loaded", self.connector_setup)

    def teardownController(self, active_widgets=None):
        if self.connector is not None:
            self.connector.unobserve(self.sync_source, self.handle_data)
        if self.facade is not None:
            self.facade.unobserve("is_loaded", self.connector_setup)


    def connector_setup(self, change):
        if change['value'] and self.verify_connector():
            self.connector.setup(self.facade.instance)
            self.connector.observe(self.sync_source, self.handle_data)
            self.is_ready = True

    def handle_data(self, c):
        conn = self.connector
        if conn.calib_handeye is not None:
            ao = conn.calib_handeye.get()
            self.results_txt.text = "Result:\n%s" % str(ao)

    def handle_keypress(self, key):
        if not self.is_ready:
            return
        if key == 32:
            self.capturePoseHE()

    def capturePoseHE(self):
        if self.connector is not None:
            # use space a default trigger
            log.info("Capture Pose Hand-Eye")
            self.connector.capture_pose(" ")



class TrackerHandEyeCalibrationModule(ModuleBase):

    def get_category(self):
        return "Tracker"

    def get_widget_class(self):
        return TrackerHandEyeCalibrationPanel

    def get_controller_class(self):
        return TrackerHandEyeCalibrationController