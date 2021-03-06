__author__ = 'MVL'
import numpy as np
import os
from atom.api import Value, Typed
import enaml
from enaml.layout.api import InsertItem
from enaml_opengl.geometry import Size

from utinteractiveconsole.plugins.calibration.controller import PreviewControllerBase, PreviewControllerFactory
from ubitrack.core import util, math, measurement

import logging
log = logging.getLogger(__name__)

class TimeDelayEstimationPreviewFactory(PreviewControllerFactory):

    def create(self, workspace, name, widget_parent):
        return TimeDelayEstimationPreview(parent=self.parent,
                                          context=self.context,
                                          config_ns=self.config_ns,
                                          widget_parent=widget_parent,
                                          widget_name=name,
                                          workspace=workspace)
class TimeDelayEstimationPreview(PreviewControllerBase):

    camera = Value()
    renderer = Value()

    bgtexture = Value()
    origin_marker = Value()
    target_marker = Value()
    tooltip_marker = Value()

    def initialize(self):
        super(TimeDelayEstimationPreview, self).initialize()
        log.info("Setup LivePreview")
        with enaml.imports():
            from .views.time_delay_estimation import TimeDelayEstimationPreviewContent
            from utinteractiveconsole.plugins.calibration.views.live_preview import LivePreview

        self.content = TimeDelayEstimationPreviewContent(parent=self.widget_parent, controller=self)
        self.content.initialize()

        # create and show preview
        self.parent.preview = LivePreview(name="%s_preview" % self.widget_name,
                                          title="Time Delay Estimation Preview",
                                          controller=self,
                                          state=self.parent.current_state,
                                          renderer=self.content.renderer)
        # add to layout
        parent = self.workspace.content.find("wizard_dockarea")
        self.parent.preview.set_parent(self.widget_parent)
        op = InsertItem(item=self.parent.preview.name, target=self.widget_name, position='right')
        parent.update_layout(op)

        self.camera = self.content.camera
        self.renderer = self.content.renderer
        self.bgtexture = self.content.scene.find("preview_bgtexture")
        self.origin_marker = self.content.scene.find("origin_marker")
        self.target_marker = self.content.scene.find("target_marker")
        self.tooltip_marker = self.content.scene.find("tooltip_marker")

        if self.facade is not None:
            # XXX needs some improvement .. is_running does not directly relate to startDataflow ..
            if self.facade.is_loaded:
                self.connector_setup(dict(value=True))
            else:
                self.facade.observe("is_loaded", self.connector_setup)
                self.facade.startDataflow()

    def teardown(self):
        log.info("Teardown LivePreview")
        if self.facade is not None:
            self.facade.stopDataflow()
            self.facade.clearDataflow()

        if self.connector is not None:
            self.connector.unobserve(self.sync_source, self.handle_data)
            self.connector.teardown(self.facade.instance)

        super(TimeDelayEstimationPreview, self).teardown()

    def connector_setup(self, change):
        # XXX Add SRG Verification to Controllers !!!
        if change['value'] == True:
            if self.connector is not None:
                self.connector.setup(self.facade.instance,
                                     update_ignore_ports=["cam2et_tracker_markers", "cam2et_tracker_target",
                                                          "cam2et_origin", "cam2oh_hip_target",
                                                          "cam2et_hip_target", "cam2fwk_hip_target"])
                self.connector.observe(self.sync_source, self.handle_data)

        if self.facade is not None:
            self.facade.unobserve("is_loaded", self.connector_setup)

    def handle_data(self, c):
        conn = self.connector

        # set debug image texture for glview
        self.renderer.enable_trigger(False)

        # could be optimized to fetch only once ...
        if conn.camera_resolution is not None:
            self.camera.camera_width, self.camera.camera_height = conn.camera_resolution.get().astype(np.int)

        if conn.camera_intrinsics is not None:
            self.camera.camera_intrinsics = conn.camera_intrinsics.get()

        if self.target_marker.visible and conn.cam2et_tracker_target is not None:
            self.target_marker.transform = conn.cam2et_tracker_target.get().toMatrix()

        if self.tooltip_marker.visible and conn.cam2et_hip_target is not None:
            self.tooltip_marker.transform = conn.cam2et_hip_target.get().toMatrix()

        self.renderer.enable_trigger(True)
        self.bgtexture.image_in(c['value'])



    def on_module_after_load(self, change):
        log.info("LivePreview: setup for module %s" % change['value'])
        current_module = change['value']
        if current_module == "tooltip_calibration":
            if self.connector is not None:
                uip = self.connector.update_ignore_ports
                if "cam2et_tracker_target" in uip:
                    uip.pop(uip.index("cam2et_tracker_target"))
                if "cam2et_hip_target" in uip:
                    uip.pop(uip.index("cam2et_hip_target"))

                self.target_marker.visible = True
                self.tooltip_marker.visible = True

    def on_module_before_unload(self, change):
        log.info("LivePreview: teardown for module %s" % change['value'])
        current_module = change['value']
        if current_module == "tooltip_calibration":
            if self.connector is not None:
                uip = self.connector.update_ignore_ports
                if "cam2et_tracker_target" not in uip:
                    uip.append("cam2et_tracker_target")
                if "cam2et_hip_target" not in uip:
                    uip.append("cam2et_hip_target")

                self.target_marker.visible = False
                self.tooltip_marker.visible = False
