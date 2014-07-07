__author__ = 'MVL'
import numpy as np
from atom.api import Bool, Value
import enaml
with enaml.imports():
    from .views.tooltip_calibration import TooltipCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class TooltipCalibrationController(CalibrationController):

    is_ready = Bool(False)

    results_txt = Value()

    def setupController(self, active_widgets=None):
        super(TooltipCalibrationController, self).setupController(active_widgets=active_widgets)

        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')

        # needs to match the SRG !!
        self.sync_source = "camera_image"

        if self.facade.master is not None:
            if self.facade.master.is_loaded:
                self.connector_setup(dict(value=True))
            else:
                self.facade.master.observe("is_loaded", self.connector_setup)

    def teardownController(self, active_widgets=None):
        if self.connector is not None:
            self.connector.unobserve("camera_image", self.handle_data)


    def connector_setup(self, change):
        # XXX Add SRG Verification to Controllers !!!
        if change['value'] == True:
            self.connector.setup(self.facade.master.instance)
            self.connector.observe("camera_image", self.handle_data)
            self.is_ready = True

        if self.facade.master is not None:
            self.facade.master.unobserve("is_loaded", self.connector_setup)

    def handle_data(self, c):
        conn = self.connector

        if self.preview_controller is not None:
            pc = self.preview_controller
            # set debug image texture for glview
            pc.renderer.enable_trigger(False)

            # could be optimized to fetch only once ...
            if conn.camera_resolution is not None:
                pc.camera.camera_width, pc.camera.camera_height = conn.camera_resolution.get().astype(np.int)

            if conn.camera_intrinsics is not None:
                pc.camera.camera_intrinsics = conn.camera_intrinsics.get()

            pc.renderer.enable_trigger(True)
            pc.bgtexture.image_in(c['value'])

            # if conn.origin_marker is not None:
            #     pc.origin_marker.transform = conn.origin_marker.get().toMatrix()
            #
            # if conn.origin_tracker is not None:
            #     pc.origin_tracker.visible = True
            #     pc.origin_tracker.transform = conn.origin_tracker.get().toMatrix()
            #
            # if conn.verification_alignment is not None:
            #     pc.verification_alignment.visible = True
            #     pc.verification_alignment.transform = conn.verification_alignment.get().toMatrix()


        results = []
        # if conn.tracker_camera_transform is not None:
        #     results.append(conn.tracker_camera_transform)
        #
        # if conn.tracker_marker_transform is not None:
        #     results.append(conn.tracker_marker_transform)

        if results:
            self.state.result.value = [str(i) for i in results]
            self.results_txt.text = "Results:\n%s" % "\n\n".join([str(i) for i in results])

    def handle_keypress(self, key):
        pass

class TooltipCalibrationModule(ModuleBase):

    def get_category(self):
        return "Co-location"

    def get_widget_class(self):
        return TooltipCalibrationPanel

    def get_controller_class(self):
        return TooltipCalibrationController