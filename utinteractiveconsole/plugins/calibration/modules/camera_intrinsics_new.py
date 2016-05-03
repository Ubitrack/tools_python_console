__author__ = 'MVL'

import enaml
with enaml.imports():
    from .views.camera_intrinsics_new import CameraIntrinsicsCalibrationNewPanel

from atom.api import Bool, Value, Enum

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

import logging
log = logging.getLogger(__name__)

class CameraIntrinsicsCalibrationNewController(LiveCalibrationController):

    is_ready = Bool(False)

    bgtexture_video = Value()
    bgtexture_grid = Value()

    results_txt = Value()

    image_selector = Enum("calibration", "verification")

    def setupController(self, active_widgets=None):
        super(CameraIntrinsicsCalibrationNewController, self).setupController(active_widgets=active_widgets)
        if active_widgets is not None:
            w = active_widgets[0]
            self.bgtexture_video = w.find('bgtexture_video')
            self.bgtexture_grid = w.find('bgtexture_grid')
            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = 'distorted_image'
        self.required_sinks = ['distorted_image', 'corner_image', 'undistorted_image', 'camera_intrinsics', 'camera_distortion',]

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
            self.bgtexture_video.image_in(c['value'])
        else:
            self.bgtexture_video.image_in(self.connector.undistorted_image)

        if self.connector.corner_image is not None:
            self.bgtexture_grid.image_in(self.connector.corner_image)

        # the intrinsics do not need to be applied unless we want to show
        # debug overlays = still we need to restart the dataflow at the moment ..

        # there should be a data structure for the camera intrinsics
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

    def useMeasurement(self):
        if self.connector is not None:
            # use space a default trigger
            log.info("Use Measurement")
            self.connector.capture_image("c")


class CameraIntrinsicsCalibrationNewModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_widget_class(self):
        return CameraIntrinsicsCalibrationNewPanel

    def get_controller_class(self):
        return CameraIntrinsicsCalibrationNewController
