__author__ = 'MVL'
from utinteractiveconsole.plugins.calibration.controller import PreviewControllerBase, PreviewControllerFactory

import enaml
import logging
log = logging.getLogger(__name__)

class PhantomColocationPreviewFactory(PreviewControllerFactory):

    def create(self):
        return PhantomColocationPreview(parent=self.parent, context=self.context)


class PhantomColocationPreview(PreviewControllerBase):


    def setupPreview(self):
        log.info("Setup LivePreview")

    def teardownPreview(self):
        log.info("Teardown LivePreview")

    def moduleSetupPreview(self, controller):
        log.info("LivePreview: setup for module %s" % controller.module_name)

    def moduleTeardownPreview(self, controller):
        log.info("LivePreview: setup for module %s" % controller.module_name)
