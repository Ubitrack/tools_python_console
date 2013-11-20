__author__ = 'mvl'
import os

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from pyqtgraph.flowchart import Flowchart
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

def main():

    app = guisupport.get_app_qt4()

    win = QtGui.QMainWindow()
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    pw1 = pg.PlotWidget()
    layout.addWidget(pw1, 0, 0)

    data = np.random.normal(size=1000)
    data[200:300] += 1
    data += np.sin(np.linspace(0, 100, 1000))

    #fc.setInput(dataIn=data)

    #pw1Node = fc.createNode('PlotWidget', pos=(0, -150))
    #pw1Node.setPlot(pw1)

    #pw2Node = fc.createNode('PlotWidget', pos=(150, -150))
    #pw2Node.setPlot(pw2)

    #fNode = fc.createNode('GaussianFilter', pos=(0, 0))
    #fNode.ctrls['sigma'].setValue(5)
    #fc.connectTerminals(fc['dataIn'], fNode['In'])
    #fc.connectTerminals(fc['dataIn'], pw1Node['In'])
    #fc.connectTerminals(fNode['Out'], pw2Node['In'])
    #fc.connectTerminals(fNode['Out'], fc['dataOut'])


    # Create an in-process kernel
    # >>> print_process_id()
    # will print the same process ID as the main process
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()
    kernel = kernel_manager.kernel
    kernel.gui = 'qt4'
    kernel.shell.push(dict(win=win, layout=layout))

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
