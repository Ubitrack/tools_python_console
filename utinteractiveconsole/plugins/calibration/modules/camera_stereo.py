__author__ = 'MVL'
import numpy as np

import enaml
with enaml.imports():
    from .views.camera_stereo import CameraStereoCalibrationPanel

from atom.api import Bool, Value, Enum

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import LiveCalibrationController

import logging
log = logging.getLogger(__name__)

class CameraStereoCalibrationController(LiveCalibrationController):

    is_ready = Bool(False)

    bgtexture_left = Value()
    camera_left = Value()
    renderer_left = Value()

    bgtexture_right = Value()
    camera_right = Value()
    renderer_right = Value()

    results_txt = Value()

    # image_selector = Enum("calibration", "verification")

    def setupController(self, active_widgets=None):
        super(CameraStereoCalibrationController, self).setupController(active_widgets=active_widgets)

        if active_widgets is not None:
            w = active_widgets[0]
            self.bgtexture_left = w.find('bgtexture_left')
            self.camera_left = w.find('camera_left')
            self.renderer_left = w.find('renderer_left')

            self.bgtexture_right = w.find('bgtexture_right')
            self.camera_right = w.find('camera_right')
            self.renderer_right = w.find('renderer_right')

            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = 'corner_image_left'
        self.required_sinks = ['corner_image_left', 'corner_image_right',
                               # 'camera_image_left', 'camera_image_right',
                               'camera_instrinsics_left', 'camera_intrinsics_right',
                               'camera_resolution_left', 'camera_resolution_right',
                               'stereo_left_right_transform',
                               ]

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

        # camera resolution/intrinsics only needed for overlays ..
        if conn.camera_resolution_left is not None:
            self.camera_left.camera_width, self.camera_left.camera_height = conn.camera_resolution_left.get().astype(np.int)

        if conn.camera_resolution_right is not None:
            self.camera_right.camera_width, self.camera_right.camera_height = conn.camera_resolution_right.get().astype(np.int)

        # set debug image texture for glview_left
        self.renderer_left.enable_trigger(False)
        if conn.camera_intrinsics_left is not None:
            self.camera_left.camera_intrinsics = conn.camera_intrinsics_left.get()
        self.renderer_left.enable_trigger(True)
        self.bgtexture_left.image_in(c['value'])

        # set debug image texture for glview_right
        self.renderer_right.enable_trigger(False)
        if conn.camera_intrinsics_right is not None:
            self.camera_right.camera_intrinsics = conn.camera_intrinsics_right.get()
        self.renderer_right.enable_trigger(True)

        if conn.corner_image_right is not None:
            self.bgtexture_right.image_in(conn.corner_image_right)

        results = []
        if conn.stereo_left_right_transform is not None:
            results.append(conn.stereo_left_right_transform)

        if results:
            self.state.result.value = [str(i) for i in results]
            self.results_txt.text = "Results:\n%s" % "\n\n".join([str(i) for i in results])

    def handle_keypress(self, key):
        if not self.is_ready:
            return
        if key == 32:
            self.captureImage()

    def captureImage(self):
        if self.connector is not None:
            # use space a default trigger
            log.info("Capture Image")
            self.connector.capture_image(" ")




class CameraStereoCalibrationModule(ModuleBase):

    def get_category(self):
        return "Camera"

    def get_widget_class(self):
        return CameraStereoCalibrationPanel

    def get_controller_class(self):
        return CameraStereoCalibrationController