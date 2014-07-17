__author__ = 'mvl'
import stevedore
import logging

import enaml
from atom.api import Atom, Value, Typed, List, Dict, ForwardTyped, Event
from enaml.workbench.api import Extension
from enaml.workbench.ui.autostart import Autostart

from .guilogging import Syslog

log = logging.getLogger(__name__)


class ExtensionManager(Atom):
    appstate = ForwardTyped(lambda: AppState)

    extensions = Dict()
    extension_manager = Typed(stevedore.extension.ExtensionManager)
    extensions_actionitems = List()
    extensions_workspaceplugins = List()

    def _default_extension_manager(self):
        return stevedore.extension.ExtensionManager(
                        namespace='utinteractiveconsole.extension',
                        invoke_on_load=True,
                        invoke_args=(self.appstate.context,),
                    )


    def updateCmdlineParser(self, parser):
        self.extension_manager.map(lambda e: e.obj.update_optparse(parser))


    def initExtensions(self):
        log.info("Init Extensions")
        # load all extensions

        def register(ext, mgr):
            log.info("Register plugin: %s" % ext.name)
            return (ext.name, ext.obj.register(mgr))

        results = self.extension_manager.map(register, self)

        for name, result in results:
            if result.widget is not None:
                result.widget.connect(result.widget, QtCore.SIGNAL('extensionChanged()'), self.updateExtensionInfo)

    def updateExtensionInfo(self):

        for cat_name, cat in self.extensions.items():
            for name, ext in cat.items():
                print cat_name, name, ext.get_name()
                print ext.get_ports()


    def registerExtension(self, name, inst, category="default", action_items=None, workspace_plugins=None, add_globals=None):
        cat = self.extensions.setdefault(category, {})
        if name in cat:
            raise ValueError("Extension with name: %s already registered in category %s." % (name, category))

        cat[name] = inst
        if action_items is not None:
            self.extensions_actionitems.append((name, category, action_items))
        if workspace_plugins is not None:
            self.extensions_workspaceplugins.append((name, category, workspace_plugins))
        if add_globals is not None:
            for k,v in add_globals.items():
                self.appstate.context[k] = v


    def generateWorkspaceActionItems(self, plugin_ext):
        result = []
        for name, category, action_items in self.extensions_actionitems:
            for item in action_items:
                try:
                    item.set_parent(plugin_ext)
                    result.append(item)
                except Exception, e:
                    log.error("error while adding menu action: %s" % item.path)
                    log.exception(e)
        return result

    def generateWorkspaceExtensions(self, plugin_manifest):
        result = []
        for name, category, plugins in self.extensions_workspaceplugins:
            for plugin in plugins:
                try:
                    log.info("Workspace Plugin: %s" % plugin.id)
                    plugin.set_parent(plugin_manifest)
                    result.append(plugin)
                except Exception, e:
                    log.error("error while adding plugin extension: %s" % plugin.id)
                    log.exception(e)

        # autostart = self.get_autostart_config()
        # if autostart is not None:
        #     def autostart_factory(workbench):
        #         log.info("Autostart workspace: %r" % autostart)
        #         return Autostart(plugin_id="utic.%s" % autostart)
        #
        #     ext = Extension(parent=plugin_manifest,
        #                     id='autostart',
        #                     point="enaml.workbench.ui.autostart",
        #                     factory=autostart_factory)
        #     result.append(ext)
        return result

    def get_autostart_config(self):
        ctx = self.appstate.context
        # preference to command line
        opt = ctx.get('options')
        if opt.autostart_workspace is not None:
            return opt.autostart_workspace
        # use config file
        cfg = ctx.get('config')
        if cfg and cfg.has_option('extensions', 'autostart'):
            return cfg.get('extensions', 'autostart')
        # no autostart
        return None

class AppState(Atom):
    context = Dict()
    extensions = Typed(ExtensionManager)
    current_workspace = Value()

    args = Value()
    options = Value()

    syslog = Typed(Syslog)

    workspace_started = Event()
    workspace_stopped = Event()


