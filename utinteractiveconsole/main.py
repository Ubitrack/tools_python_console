__author__ = 'mvl'
import os, sys
import ConfigParser

import enaml
from enaml.qt import QtCore, QtGui
from enaml.qt.qt_application import QtApplication

from atom.api import Atom, Float, Value, Typed, List, Dict, Unicode, ForwardTyped
from IPython.lib import guisupport

from ubitrack.core import util

from optparse import OptionParser
from stevedore import extension
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class Extensions(Atom):
    appstate = ForwardTyped(lambda: AppState)

    extensions = Dict()
    extension_manager = Value()
    extensions_menuitems = List()

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


    def registerExtension(self, name, inst, category="default", menu_items=None, add_globals=None):
        cat = self.extensions.setdefault(category, {})
        if name in cat:
            raise ValueError("Extension with name: %s already registered in category %s." % (name, category))

        cat[name] = inst
        if menu_items is not None:
            self.extensions_menuitems.append((name, category, menu_items))
        if add_globals is not None:
            for k,v in add_globals.items():
                self.appstate.context[k] = v

    def addToolbarItems(self, toolbar):
        for name, category, menu_items in self.extensions_menuitems:
            for info in menu_items:
                try:
                    #menu_id = info["menu_id"]
                    #menu = self.menus.setdefault(menu_id, None)
                    #if menu is None:
                    #    menu = self.menus[menu_id] = self.menubar.addMenu(info.get("menu_title", menu_id.capitalize()))
                    toolbar.children.append(info["menu_action"])
                except Exception, e:
                    log.error("error while adding menu action")
                    log.exception(e)




class AppState(Atom):
    context = Dict()
    extensions = Typed(Extensions)

    args = Value()
    options = Value()




def main():

    parser = OptionParser()

    parser.add_option("-l", "--logconfig",
                  action="store", dest="logconfig", default="log4cpp.conf",
                  help="log4cpp config file")

    parser.add_option("-C", "--configfile",
                  action="store", dest="configfile", default="~/.utic.conf",
                  help="Interactive console config file")

    appstate = AppState(context=dict())
    extensions = Extensions(appstate=appstate)

    extensions.updateCmdlineParser(parser)
    appstate.extensions=extensions
    appstate.context['extensions'] = extensions

    (options, args) = parser.parse_args()

    appstate.args = args
    appstate.options = options

    appstate.context['args'] = args
    appstate.context['options'] = options

    cfgfiles = []
    if (os.path.isfile(options.configfile)):
        cfgfiles.append(options.configfile)

    if (os.path.isfile(os.environ.get("UTIC_CONFIG_FILE", ""))):
        cfgfiles.append(os.environ["UTIC_CONFIG_FILE"])

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

    QtGui.QApplication.setGraphicsSystem('raster')
    app = QtApplication()
    appstate.context['app'] = app

    util.initLogging(options.logconfig)

    with enaml.imports():
        from utinteractiveconsole.ui.views.main_window import Main

    extensions.initExtensions()

    appstate.context['appstate'] = appstate

    win = Main(appstate=appstate)
    appstate.context['main_window'] = win

    extensions.addToolbarItems(win.tbar)

    win.show()
    guisupport.start_event_loop_qt4(app._qapp)


if __name__ == '__main__':
    main()
