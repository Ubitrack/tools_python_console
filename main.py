__author__ = 'mvl'
import os, sys

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from pyqtgraph.flowchart import Flowchart
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


from ubitrack.core import math, measurement, util
from ubitrack.facade import facade

from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("-a", "--autostart",
                  action="store_false", dest="autostart", default=True,
                  help="automatically start dataflow?")

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print "utInteractiveConsole <filename>"
        sys.exit(0)

    filename = args[0]

    app = guisupport.get_app_qt4()

    win = QtGui.QMainWindow()
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    util.initLogging("log4cpp.conf")
    df = facade.AdvancedFacade("/usr/local/lib/ubitrack")

    df.loadDataflow(filename, True)
    if options.autostart:
        df.startDataflow()

    # Create an in-process kernel
    # >>> print_process_id()
    # will print the same process ID as the main process
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()
    kernel = kernel_manager.kernel
    kernel.gui = 'qt4'
    kernel.shell.push(dict(win=win, layout=layout, df=df, filename=filename, args=args, options=options))

    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    def stop():
        kernel_client.stop_channels()
        kernel_manager.shutdown_kernel()
        app.exit()

    control = RichIPythonWidget()
    control.kernel_manager = kernel_manager
    control.kernel_client = kernel_client
    control.exit_requested.connect(stop)

    #control.show()

    layout.addWidget(control, 2, 0, 1, 2)

    win.show()


    guisupport.start_event_loop_qt4(app)


if __name__ == '__main__':
    main()
