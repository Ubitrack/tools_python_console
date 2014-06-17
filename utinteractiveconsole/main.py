__author__ = 'mvl'
import os, sys
import StringIO
import ConfigParser
from optparse import OptionParser
from stevedore import extension
import logging

import enaml
from atom.api import Atom, Value, Typed, List, Dict, ForwardTyped
from enaml.workbench.ui.api import UIWorkbench
from enaml.workbench.ui.api import ActionItem
from enaml.workbench.api import Extension

from ubitrack.core import util
from utinteractiveconsole.guilogging import Syslog, ConsoleWindowLogHandler


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Extensions(Atom):
    appstate = ForwardTyped(lambda: AppState)

    extensions = Dict()
    extension_manager = Value()
    extensions_actionitems = List()
    extensions_workspaceplugins = List()

    def _default_extension_manager(self):
        return extension.ExtensionManager(
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
            for info in action_items:
                try:
                    result.append(ActionItem(parent=plugin_ext, **info))
                except Exception, e:
                    log.error("error while adding menu action: %s" % info['path'])
                    log.exception(e)
        return result

    def generateWorkspaceExtensions(self, plugin_manifest):
        result = []
        for name, category, plugins in self.extensions_workspaceplugins:
            for info in plugins:
                try:
                    result.append(Extension(parent=plugin_manifest, **info))
                except Exception, e:
                    log.error("error while adding plugin extension: %s" % info['path'])
                    log.exception(e)
        return result




class AppState(Atom):
    context = Dict()
    extensions = Typed(Extensions)

    args = Value()
    options = Value()

    syslog = Typed(Syslog)




def main():

    parser = OptionParser()

    parser.add_option("-l", "--logconfig",
                  action="store", dest="logconfig", default="/etc/mvl/log4cpp.conf",
                  help="log4cpp config file")

    parser.add_option("-L", "--show-logwindow",
                  action="store_true", dest="show_logwindow", default=False,
                  help="Show logging window in gui")

    parser.add_option("-C", "--configfile",
                  action="store", dest="configfile", default="~/utic.conf",
                  help="Interactive console config file")


    syslog = Syslog()

    appstate = AppState(context=dict(),
                        syslog=syslog)
    extensions = Extensions(appstate=appstate)

    extensions.updateCmdlineParser(parser)
    appstate.extensions=extensions
    appstate.context['extensions'] = extensions

    (options, args) = parser.parse_args()

    if options.show_logwindow:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(syslog.handler)


    appstate.args = args
    appstate.options = options

    appstate.context['args'] = args
    appstate.context['options'] = options

    cfgfiles = []
    if (os.path.isfile(os.environ.get("UTIC_CONFIG_FILE", "/etc/mvl/utic.conf"))):
        cfgfiles.append(os.environ["UTIC_CONFIG_FILE"])

    if (os.path.isfile(options.configfile)):
        cfgfiles.append(options.configfile)

    config = ConfigParser.ConfigParser()
    try:
        config.read(cfgfiles)
        appstate.context['config'] = config
    except Exception, e:
        log.error("Error parsing config file(s): %s" % (cfgfiles,))
        log.exception(e)


    if len(args) < 1:
        filename = None
    else:
        filename = args[0]
    appstate.context['filename'] = filename

    with enaml.imports():
        from utinteractiveconsole.ui.manifest import ApplicationManifest

    workbench = UIWorkbench()

    util.initLogging(options.logconfig)
    extensions.initExtensions()
    # XXX use weakref here !!
    appstate.context['appstate'] = appstate

    manifest = ApplicationManifest(appstate=appstate, extension_mgr=extensions)
    print manifest.children[2].objects

    workbench.register(manifest)
    workbench.run()


if __name__ == '__main__':
    main()
