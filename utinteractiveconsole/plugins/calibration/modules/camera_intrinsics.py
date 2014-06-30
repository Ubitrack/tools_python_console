__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.camera_intrinsics import CameraIntrinsicsCalibrationPanel

from atom.api import Bool, Value

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

class CameraIntrinsicsCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    bgtexture = Value()

    def setupController(self, active_widgets=None):

        if active_widgets is not None:
            self.bgtexture = active_widgets[0].bgtexture

        # needs to match the SRG !!
        self.sync_source = "debug_image"

        if self.facade is not None:
            self.facade.observe("is_loaded", self.connector_setup)


    def connector_setup(self, change):

        def store_image(c):
            self.bgtexture.image_in(c['value'])

        if change['value'] == True:
            self.connector.setup(self.facade.instance)
            self.connector.observe("debug_image", store_image)
            self.is_ready = True


    def captureImage(self):
        if self.connector is not None:
            # use space a default trigger
            print "trigger"
            self.connector.trigger(" ")


class CameraIntrinsicsCalibrationModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_name(self):
        return "Camera Intrinsics"

    def get_dependencies(self):
        return []

    def get_widget_class(self):
        return CameraIntrinsicsCalibrationPanel

    def get_controller_class(self):
        return CameraIntrinsicsCalibrationController