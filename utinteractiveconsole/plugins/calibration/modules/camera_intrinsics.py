__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.camera_intrinsics import CameraIntrinsicsCalibrationPanel

from atom.api import Bool, Value, Enum

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

import logging
log = logging.getLogger(__name__)

class CameraIntrinsicsCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    bgtexture = Value()

    results_txt = Value()

    image_selector = Enum("calibration", "verification")

    def setupController(self, active_widgets=None):
        super(CameraIntrinsicsCalibrationController, self).setupController(active_widgets=active_widgets)
        if active_widgets is not None:
            w = active_widgets[0]
            self.bgtexture = w.find('bgtexture')
            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = 'corner_image'
        self.required_sinks = ['corner_image', 'camera_image', 'camera_intrinsics', 'camera_distortion',]

        if self.facade is not None:
            self.facade.observe("is_loaded", self.connector_setup)

    def teardownController(self, active_widgets=None):
        if self.connector is not None:
            self.connector.unobserve(self.sync_source, self.handle_data)
        if self.facade is not None:
            self.facade.unobserve("is_loaded", self.connector_setup)


    def connector_setup(self, change):
        if change['value'] == True and self.verify_connector() == True:
            self.connector.setup(self.facade.instance)
            self.connector.observe(self.sync_source, self.handle_data)
            self.is_ready = True

    def handle_data(self, c):

        if self.image_selector == "calibration":
            self.bgtexture.image_in(c['value'])
        else:
            self.bgtexture.image_in(self.connector.camera_image)

            # the intrinsics do not need to be applied unless we want to show
            # debug overlays = still we need to restart the dataflow at the moment ..

        results = []
        if self.connector.camera_intrinsics is not None:
            results.append(self.connector.camera_intrinsics)
        if self.connector.camera_distortion is not None:
            results.append(self.connector.camera_distortion)

        if results:
            self.state.result.value = [str(i) for i in results]
            self.results_txt.text = "Results:\n%s" % "\n\n".join([str(i) for i in results])



    def captureImage(self):
        if self.connector is not None:
            # use space a default trigger
            log.info("Capture Image")
            self.connector.capture_image(" ")


class CameraIntrinsicsCalibrationModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_widget_class(self):
        return CameraIntrinsicsCalibrationPanel

    def get_controller_class(self):
        return CameraIntrinsicsCalibrationController