__author__ = 'mvl'
import os, sys

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from pyqtgraph.Qt import QtGui

from ubitrack.core import util
from ubitrack.facade import facade

from optparse import OptionParser



class UTInteractiveConsoleWindow(QtGui.QMainWindow):

    def __init__(self, app, options, args):
        super(UTInteractiveConsoleWindow, self).__init__()
        self.app = app
        self.args = args
        self.options = options

        self.initUI()
        self.initREPL()
        self.initExtensions()

    def initUI(self):
        self.cw = QtGui.QWidget()
        self.setCentralWidget(self.cw)
        self.layout = QtGui.QGridLayout()
        self.cw.setLayout(self.layout)

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        #self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Ubitrack Interactive Console')


    def initREPL(self):
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push(dict(win=self, app=self.app, layout=self.layout, args=self.args, options=self.options))

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
            self.app.exit()

        self.control = RichIPythonWidget()
        self.control.kernel_manager = self.kernel_manager
        self.control.kernel_client = self.kernel_client
        self.control.exit_requested.connect(stop)
        self.layout.addWidget(self.control, 2, 0, 1, 2)

    def initExtensions(self):
        pass

def main():
    parser = OptionParser()
    parser.add_option("-a", "--autostart",
                  action="store_false", dest="autostart", default=True,
                  help="automatically start dataflow?")

    parser.add_option("-l", "--logconfig",
                  action="store", dest="logconfig", default="log4cpp.conf",
                  help="log4cpp config file")

    parser.add_option("-c", "--components_path",
                  action="store", dest="components_path", default="/usr/local/lib/ubitrack",
                  help="path to UbiTrack components")

    (options, args) = parser.parse_args()

    if len(args) < 1:
        #print "utInteractiveConsole <filename>"
        #sys.exit(0)
        filename = None
    else:
        filename = args[0]

    app = guisupport.get_app_qt4()

    win = UTInteractiveConsoleWindow(app, options, args)

    util.initLogging(options.logconfig)
    df = facade.AdvancedFacade(options.components_path)

    if filename is not None:
        df.loadDataflow(filename, True)
        if options.autostart:
            df.startDataflow()


    win.show()

    guisupport.start_event_loop_qt4(app)


if __name__ == '__main__':
    main()
