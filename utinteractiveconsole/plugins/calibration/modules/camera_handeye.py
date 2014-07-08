__author__ = 'MVL'
import numpy as np

import enaml
with enaml.imports():
    from .views.camera_handeye import CameraHandEyeCalibrationPanel

from atom.api import Bool, Value, Enum

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

import logging
log = logging.getLogger(__name__)

class CameraHandEyeCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    bgtexture = Value()
    camera = Value()
    renderer = Value()

    origin_marker = Value()
    origin_tracker = Value()
    verification_alignment = Value()

    results_txt = Value()

    image_selector = Enum("calibration", "verification")

    def setupController(self, active_widgets=None):
        super(CameraHandEyeCalibrationController, self).setupController(active_widgets=active_widgets)

        if active_widgets is not None:
            w = active_widgets[0]
            self.bgtexture = w.find('bgtexture')
            self.camera = w.find('camera')
            self.renderer = w.find('renderer')
            self.origin_marker = w.find('origin_marker')
            self.origin_tracker = w.find('origin_tracker')
            self.verification_alignment = w.find('verification_alignment')

            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = 'debug_image'
        self.required_sinks = ['debug_image', 'camera_image', 'camera_resolution', 'camera_intrinsics',
                               'origin_marker', 'origin_tracker', 'verification_alignment']

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
        conn = self.connector

        if self.image_selector == "calibration":
            # set debug image texture for glview
            self.renderer.enable_trigger(False)

            if conn.camera_intrinsics is not None:
                self.camera.camera_intrinsics = conn.camera_intrinsics.get()
            self.origin_marker.visible = False
            self.origin_tracker.visible = False
            self.verification_alignment.visible = False

            self.renderer.enable_trigger(True)
            self.bgtexture.image_in(c['value'])
        else:
            # set marker tracking for glview1
            self.renderer.enable_trigger(False)

            # could be optimized to fetch only once ...
            if conn.camera_resolution is not None:
                self.camera.camera_width, self.camera.camera_height = conn.camera_resolution.get().astype(np.int)

            if conn.camera_intrinsics is not None:
                self.camera.camera_intrinsics = conn.camera_intrinsics.get()

            if conn.origin_marker is not None:
                self.origin_marker.visible = True
                self.origin_marker.transform = conn.origin_marker.get().toMatrix()

            if conn.origin_tracker is not None:
                self.origin_tracker.visible = True
                self.origin_tracker.transform = conn.origin_tracker.get().toMatrix()

            if conn.verification_alignment is not None:
                self.verification_alignment.visible = True
                self.verification_alignment.transform = conn.verification_alignment.get().toMatrix()

            self.renderer.enable_trigger(True)
            if conn.camera_image is not None:
                self.bgtexture.image_in(conn.camera_image)


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