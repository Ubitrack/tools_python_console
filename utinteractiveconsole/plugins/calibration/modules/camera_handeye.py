__author__ = 'MVL'
import numpy as np

import enaml
with enaml.imports():
    from .views.camera_handeye import CameraHandEyeCalibrationPanel

from atom.api import Bool, Value

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

import logging
log = logging.getLogger(__name__)

class CameraHandEyeCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    bgtexture = Value()
    camera = Value()
    renderer = Value()

    bgtexture1 = Value()
    camera1 = Value()
    renderer1 = Value()
    verification_marker = Value()

    results_txt = Value()

    def setupController(self, active_widgets=None):

        if active_widgets is not None:
            w = active_widgets[0]
            self.bgtexture = w.find('bgtexture')
            self.camera = w.find('camera')
            self.renderer = w.find('renderer')

            self.bgtexture1 = w.find('bgtexture1')
            self.camera1 = w.find('camera1')
            self.renderer1 = w.find('renderer1')
            self.verification_marker = w.find('verification_marker')

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

        # set marker tracking for glview1
        self.renderer1.enable_trigger(False)

        if conn.camera_resolution is not None:
            self.camera1.camera_width, self.camera1.camera_height = conn.camera_resolution.get().astype(np.int)

        if conn.camera_intrinsics is not None:
            self.camera1.camera_intrinsics = conn.camera_intrinsics.get()

        if conn.camera_pose is not None:
            self.camera1.modelview_matrix = conn.camera_pose.get().toMatrix()

        if conn.marker_pose_verification is not None:
            self.verification_marker.transform = conn.marker_pose_verification.get().toMatrix()

        self.renderer1.enable_trigger(True)
        if conn.camera_image is not None:
            self.bgtexture1.image_in(conn.camera_image)


        results = []
        if conn.tracker2camera_transform is not None:
            results.append(conn.tracker2camera_transform)

        if results:
            self.results_txt.text = "Results:\n%s" % "\n\n".join([str(i) for i in results])



    def capturePose(self):
        if self.connector is not None:
            # use space a default trigger
            log.info("Capture Pose")
            self.connector.capture_pose(" ")


class CameraHandEyeCalibrationModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_name(self):
        return "Camera HandEye"

    def get_dependencies(self):
        return ['camera_intrinsics',]

    def get_widget_class(self):
        return CameraHandEyeCalibrationPanel

    def get_controller_class(self):
        return CameraHandEyeCalibrationController