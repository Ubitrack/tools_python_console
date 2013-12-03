__author__ = 'mvl'
import os, sys

from enaml.qt import QtCore, QtGui
from enaml.qt.qt_application import QtApplication

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from ubitrack.core import util
from ubitrack.facade import facade

from optparse import OptionParser
from stevedore import extension
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class UTInteractiveConsoleWindow(QtGui.QMainWindow):

    def __init__(self, app, df, options, args):
        super(UTInteractiveConsoleWindow, self).__init__()
        self.app = app
        self.df = df
        self.args = args
        self.options = options

        self.extension_manager = None
        self.extensions = {}
        self.extensions_menuitems = []
        self.extensions_globals = []

        self.initUI()
        self.initREPL()
        self.initExtensions()


    def initUI(self):
        log.info("Init UI")
        self.cw = QtGui.QWidget()
        self.setCentralWidget(self.cw)
        self.layout = QtGui.QGridLayout()
        self.cw.setLayout(self.layout)

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.QApplication.quit)

        self.statusBar()

        self.menus = {}
        self.menubar = self.menuBar()
        self.menus["file"] = fileMenu = self.menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        #self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Ubitrack Interactive Console')

        self.context = dict(win=self, df=self.df, app=self.app, layout=self.layout,
                            args=self.args, options=self.options, extensions=self.extensions)


    def initREPL(self):
        log.info("Init REPL")
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push(self.context)

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
            QtGui.QApplication.quit()

        self.control = RichIPythonWidget()
        self.control.kernel_manager = self.kernel_manager
        self.control.kernel_client = self.kernel_client
        self.control.exit_requested.connect(stop)
        self.layout.addWidget(self.control, 2, 0, 1, 2)


    def initExtensions(self):
        log.info("Init Extensions")
        # load all extensions

        self.extension_manager = extension.ExtensionManager(
            namespace='utinteractiveconsole.extension',
            invoke_on_load=True,
            invoke_args=(self.context,),
        )

        def register(ext, mgr):
            log.info("Register plugin: %s" % ext.name)
            return (ext.name, ext.obj.register(mgr))

        results = self.extension_manager.map(register, self)

        for name, result in results:
            if result.widget is not None:
                self.connect(result.widget, QtCore.SIGNAL('extensionChanged()'), self.updateExtensionInfo)

        for name, category, menu_items in self.extensions_menuitems:
            for info in menu_items:
                try:
                    menu_id = info["menu_id"]
                    menu = self.menus.setdefault(menu_id, None)
                    if menu is None:
                        menu = self.menus[menu_id] = self.menubar.addMenu(info.get("menu_title", menu_id.capitalize()))
                    menu.addAction(info["menu_action"])
                except Exception, e:
                    log.error("error while adding menu action")
                    log.exception(e)


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
            self.extensions_globals.append(add_globals)




def main():

    default_componenents_path = "/usr/local/lib/ubitrack"
    if sys.platform.startswith("win32") and "UBITRACK_PATH" in os.environ:
        default_componenents_path = os.path.join(os.environ["UBITRACK_PATH"], "ubitrack")

    parser = OptionParser()
    parser.add_option("-a", "--autostart",
                  action="store_false", dest="autostart", default=True,
                  help="automatically start dataflow?")

    parser.add_option("-l", "--logconfig",
                  action="store", dest="logconfig", default="log4cpp.conf",
                  help="log4cpp config file")

    parser.add_option("-c", "--components_path",
                  action="store", dest="components_path", default=default_componenents_path,
                  help="path to UbiTrack components")

    (options, args) = parser.parse_args()

    if len(args) < 1:
        #print "utInteractiveConsole <filename>"
        #sys.exit(0)
        filename = None
    else:
        filename = args[0]

    # not optimal, since it mixes ipython/pyqtgraph/enaml wrappers %-/
    #app = guisupport.get_app_qt4()
    app = QtApplication()

    util.initLogging(options.logconfig)
    df = facade.AdvancedFacade(options.components_path)

    win = UTInteractiveConsoleWindow(app, df, options, args)

    if filename is not None:
        df.loadDataflow(filename, True)
        if options.autostart:
            df.startDataflow()


    win.show()

    guisupport.start_event_loop_qt4(app._qapp)


if __name__ == '__main__':
    main()
