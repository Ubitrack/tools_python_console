__author__ = 'MVL'
from atom.api import Value, Typed
import enaml
from enaml.layout.api import InsertItem
from enaml_opengl.geometry import Size

from utinteractiveconsole.plugins.calibration.controller import PreviewControllerBase, PreviewControllerFactory

import logging
log = logging.getLogger(__name__)


class GuiTestPreviewFactory(PreviewControllerFactory):

    def create(self, workspace, name, widget_parent):
        return GuiTestPreview(parent=self.parent,
                              context=self.context,
                              widget_parent=widget_parent,
                              widget_name=name,
                              workspace=workspace)

class GuiTestPreview(PreviewControllerBase):

    camera = Value()
    renderer = Value()

    bgtexture = Value()
    # origin_marker = Value()
    # origin_tracker = Value()
    # verification_alignment = Value()

    def setupPreview(self):
        log.info("Setup LivePreview")
        with enaml.imports():
            from .views.gui_test import GuiTestPreviewContent
            from utinteractiveconsole.plugins.calibration.views.live_preview import LivePreview

        self.content = GuiTestPreviewContent(parent=self.widget_parent, controller=self)
        self.content.initialize()

        # create and show preview
        self.parent.preview = LivePreview(name="%s_preview" % self.widget_name,
                                          title="Gui Test Preview",
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
        # self.origin_marker = self.content.scene.find("origin_marker")
        # self.origin_tracker = self.content.scene.find("origin_tracker")
        # self.verification_alignment = self.content.scene.find("verification_alignment")

        cfg = self.parent.current_state.config
        facade = self.parent.current_state.facade
        if 'master_dfg_basedir' in cfg and 'master_dfg_filename' in cfg:
            facade.setupMaster(cfg['master_dfg_basedir'],
                               cfg['master_dfg_filename'])
            facade.master.startDataflow()

    def teardownPreview(self):
        log.info("Teardown LivePreview")
        facade = self.parent.current_state.facade
        if facade.master is not None:
            facade.master.stopDataflow()

        self.camera = None
        self.renderer = None
        self.bgtexture = None
        # self.origin_marker = None
        # self.origin_tracker = None
        # self.verification_alignment = None

    def moduleSetupPreview(self, controller):
        log.info("LivePreview: setup for module %s" % controller.module_name)

    def moduleTeardownPreview(self, controller):
        log.info("LivePreview: setup for module %s" % controller.module_name)
