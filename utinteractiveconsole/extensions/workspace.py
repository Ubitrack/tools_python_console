__author__ = 'jack'
from atom.api import Subclass, Unicode, Typed

from enaml.widgets.api import Container
from enaml.workbench.api import PluginManifest
from enaml.workbench.ui.api import Workspace

from utinteractiveconsole.app import AppState
from .base import ExtensionBase


class ExtensionWorkspace(Workspace):
    """ A custom Workspace class for extensions.

    This workspace class will instantiate the content and register an
    additional plugin with the workbench when it is started. The extra
    plugin can be used to add addtional functionality to the workbench
    window while this workspace is active. The plugin is unregistered
    when the workspace is stopped.

    """
    #: The enamldef'd Container to create when the workbench is started.
    content_def = Subclass(Container)

    #: The enamldef'd PluginManifest to register on start.
    manifest_def = Subclass(PluginManifest)

    #: Storage for the plugin manifest's id.
    _manifest_id = Unicode()

    # global state for the utic plugins
    appstate = Typed(AppState)
    utic_plugin = Typed(ExtensionBase)

    def start(self):
        """ Start the workspace instance.

        This method will create the container content and register the
        provided plugin with the workbench.

        """
        self.content = self.content_def(appstate=self.appstate, utic_plugin=self.utic_plugin)
        manifest = self.manifest_def(appstate=self.appstate, utic_plugin=self.utic_plugin)
        self._manifest_id = manifest.id
        self.workbench.register(manifest)

    def stop(self):
        """ Stop the workspace instance.

        This method will unregister the workspace's plugin that was
        registered on start.

        """
        # XXX check for running processes / unsaved changes here ?
        self.workbench.unregister(self._manifest_id)

