__author__ = 'mvl'
import os

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from pyqtgraph.flowchart import Flowchart
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


from ubitrack.core import math, measurement, util
from ubitrack.facade import facade

class ProcessData(object):
    def __init__(self, hp, hpc, tp, pw, buffer_size=1024):
        self.hp = hp
        self.hpc = hpc
        self.tp = tp
        self.pw = pw

        self.data =  np.zeros((buffer_size, 3))
        self.plot_handler0 = pw.plot(pen=(255,0,0), y=self.data[:,0])
        self.plot_handler1 = pw.plot(pen=(0,255,0), y=self.data[:,1])
        self.plot_handler2 = pw.plot(pen=(0,0,255), y=self.data[:,2])

    def tick(self, *args, **kw):
        ts = measurement.now()
        hp = self.hp.get(ts).get().translation()
        hpc = self.hpc.get(ts).get().translation()
        tp = self.tp.get(ts).get().translation()

        self.data = np.concatenate((self.data[:1,:], self.data[:-1,:]), 0)
        self.data[0,0] = np.linalg.norm(tp - hp)
        self.data[0,1] = np.linalg.norm(hp - hpc)
        self.data[0,2] = np.linalg.norm(tp - hpc)

        self.plot_handler0.setData(self.data[:,0])
        self.plot_handler1.setData(self.data[:,1])
        self.plot_handler2.setData(self.data[:,2])

def main():

    app = guisupport.get_app_qt4()

    win = QtGui.QMainWindow()
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    pw1 = pg.PlotWidget()
    layout.addWidget(pw1, 0, 0)


    util.initLogging("log4cpp.conf")
    df = facade.AdvancedFacade("/usr/local/lib/ubitrack")

    df.loadDataflow("/home/mvl/vhar_calibration/calib_linux/haptic_workspace_accuracy_test.dfg", True)


    ps_haptic_pose = df.getApplicationPullSinkPose("haptic_pose")
    ps_haptic_pose_calib = df.getApplicationPullSinkPose("haptic_pose_calib")
    ps_tracker_pose = df.getApplicationPullSinkPose("tracker_pose")

    pd = ProcessData(ps_haptic_pose, ps_haptic_pose_calib, ps_tracker_pose, pw1)
    df.startDataflow()

    # Create an in-process kernel
    # >>> print_process_id()
    # will print the same process ID as the main process
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()
    kernel = kernel_manager.kernel
    kernel.gui = 'qt4'
    kernel.shell.push(dict(win=win, layout=layout, pw1=pw1, df=df,
                           ps_haptic_pose=ps_haptic_pose,
                           ps_haptic_pose_calib=ps_haptic_pose_calib,
                           ps_tracker_pose=ps_tracker_pose))

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


    updater = QtCore.QTimer(win)
    updater.timeout.connect(pd.tick)
    updater.setInterval(33)
    updater.start()


    win.show()


    guisupport.start_event_loop_qt4(app)


if __name__ == '__main__':
    main()
