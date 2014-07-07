__author__ = 'MVL'
import numpy as np
from atom.api import Bool, Value
import enaml
with enaml.imports():
    from .views.gui_test import GuiTestPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

class GuiTestController(CalibrationController):

    is_ready = Bool(False)

    results_txt = Value()

    def setupController(self, active_widgets=None):
        super(GuiTestController, self).setupController(active_widgets=active_widgets)

        if active_widgets is not None:
            w = active_widgets[0]
            self.results_txt = w.find('results_txt')

    def connector_setup(self, change):
        # XXX Add SRG Verification to Controllers !!!
        if change['value'] == True:
            self.connector.setup(self.facade.instance)
            self.connector.observe("camera_image", self.handle_data)
            self.is_ready = True

        if self.facade is not None:
            self.facade.unobserve("is_loaded", self.connector_setup)

    def handle_data(self, c):
        conn = self.connector
        print "handle_data", c

        if self.preview_controller is not None:
            pass
            # pc = self.preview_controller
            # # set debug image texture for glview
            # pc.renderer.enable_trigger(False)
            #
            # # could be optimized to fetch only once ...
            # if conn.camera_resolution is not None:
            #     pc.camera.camera_width, pc.camera.camera_height = conn.camera_resolution.get().astype(np.int)
            #
            # if conn.camera_intrinsics is not None:
            #     pc.camera.camera_intrinsics = conn.camera_intrinsics.get()
            #
            # pc.renderer.enable_trigger(True)
            # pc.bgtexture.image_in(c['value'])


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

class GuiTestModule(ModuleBase):

    def get_category(self):
        return "Testing"

    def get_widget_class(self):
        return GuiTestPanel

    def get_controller_class(self):
        return GuiTestController