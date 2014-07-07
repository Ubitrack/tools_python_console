__author__ = 'MVL'
import numpy as np

import enaml
with enaml.imports():
    from .views.camera_handeye import CameraHandEyeCalibrationPanel

from atom.api import Bool, Value

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

import logging
log = logging.getLogger(__name__)

class CameraHandEyeCalibrationController(CalibrationController):

    is_ready = Bool(False)

    bgtexture = Value()
    camera = Value()
    renderer = Value()

    origin_marker = Value()
    origin_tracker = Value()
    verification_alignment = Value()

    results_txt = Value()

    def setupController(self, active_widgets=None):
        super(CameraHandEyeCalibrationController, self).setupController(active_widgets=active_widgets)

        if active_widgets is not None:
            w = active_widgets[0]
            self.bgtexture = w.find('bgtexture')
            self.camera = w.find('camera')
            self.renderer = w.find('renderer')

            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = "debug_image"

        if self.facade is not None:
            self.facade.observe("is_loaded", self.connector_setup)

    def teardownController(self, active_widgets=None):
        if self.connector is not None:
            self.connector.unobserve("debug_image", self.handle_data)
        if self.facade is not None:
            self.facade.unobserve("is_loaded", self.connector_setup)


    def connector_setup(self, change):
        # XXX Add SRG Verification to Controllers !!!
        if change['value'] == True:
            self.connector.setup(self.facade.instance)
            self.connector.observe("debug_image", self.handle_data)
            self.is_ready = True

    def handle_data(self, c):
        conn = self.connector

        # set debug image texture for glview
        self.renderer.enable_trigger(False)
        if conn.camera_intrinsics is not None:
            self.camera.camera_intrinsics = conn.camera_intrinsics.get()
        self.renderer.enable_trigger(True)
        self.bgtexture.image_in(c['value'])

        if self.preview_controller is not None:
            pc = self.preview_controller
            # set marker tracking for glview1
            pc.renderer.enable_trigger(False)

            # could be optimized to fetch only once ...
            if conn.camera_resolution is not None:
                pc.camera.camera_width, pc.camera.camera_height = conn.camera_resolution.get().astype(np.int)

            if conn.camera_intrinsics is not None:
                pc.camera.camera_intrinsics = conn.camera_intrinsics.get()

            if conn.origin_marker is not None:
                pc.origin_marker.transform = conn.origin_marker.get().toMatrix()

            if conn.origin_tracker is not None:
                pc.origin_tracker.visible = True
                pc.origin_tracker.transform = conn.origin_tracker.get().toMatrix()

            if conn.verification_alignment is not None:
                pc.verification_alignment.visible = True
                pc.verification_alignment.transform = conn.verification_alignment.get().toMatrix()

            pc.renderer.enable_trigger(True)
            if conn.camera_image is not None:
                pc.bgtexture.image_in(conn.camera_image)


        results = []
        if conn.tracker_camera_transform is not None:
            results.append(conn.tracker_camera_transform)

        if conn.tracker_marker_transform is not None:
            results.append(conn.tracker_marker_transform)

        if results:
            self.state.result.value = [str(i) for i in results]
            self.results_txt.text = "Results:\n%s" % "\n\n".join([str(i) for i in results])

    def handle_keypress(self, key):
        if not self.is_ready:
            return
        if key == 32:
            self.capturePoseHE()
        elif key == 65:
            self.capturePoseAlign()

    def capturePoseHE(self):
        if self.connector is not None:
            # use space a default trigger
            log.info("Capture Pose Hand-Eye")
            self.connector.capture_pose(" ")

    def capturePoseAlign(self):
        if self.connector is not None:
            # use space a default trigger
            log.info("Capture Pose Align")
            self.connector.capture_pose("a")



class CameraHandEyeCalibrationModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_widget_class(self):
        return CameraHandEyeCalibrationPanel

    def get_controller_class(self):
        return CameraHandEyeCalibrationController