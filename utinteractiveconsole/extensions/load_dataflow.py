__author__ = 'jack'
import os, sys
from pyqtgraph.Qt import QtCore, QtGui
from utinteractiveconsole.extensions import ExtensionBase


class LoadDataflowWidget(QtGui.QWidget):

    def __init__(self, extension=None, parent=None):
        super(LoadDataflowWidget, self).__init__(parent)
        self.extension = extension
        self.df = self.extension.context.get("df")
        self.initUI()

        self.is_loaded = False

    def initUI(self):
        grid = QtGui.QGridLayout()

        self.start_button = QtGui.QPushButton("Start")
        grid.addWidget(self.start_button, 0, 0)
        self.connect(self.start_button, QtCore.SIGNAL('clicked()'), self.handle_start_dataflow)

        self.stop_button = QtGui.QPushButton("Stop")
        grid.addWidget(self.stop_button, 0, 1)
        self.connect(self.stop_button, QtCore.SIGNAL('clicked()'), self.handle_stop_dataflow)
        self.stop_button.setEnabled(False)

        self.close_button = QtGui.QPushButton("Close")
        grid.addWidget(self.close_button, 0, 3)
        self.connect(self.close_button, QtCore.SIGNAL('clicked()'), self.handle_unload_dataflow)

        self.setLayout(grid)
        self.setWindowTitle('Dataflow')


    def showDialog(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open Dataflow Graph', './')
        self.load_dataflow(fname)

    def handle_load_dataflow(self):
        if not self.is_loaded:
            self.showDialog()
        else:
            reply = QtGui.QMessageBox.question(self, 'Dataflow already loaded.',
                "Do you want to replace the current instance?", QtGui.QMessageBox.Yes |
                QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                self.handle_unload_dataflow()
                self.showDialog()

    def handle_start_dataflow(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if self.df is not None:
            self.df.startDataflow()

    def handle_stop_dataflow(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.df is not None:
            self.df.stopDataflow()

    def handle_unload_dataflow(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.df is not None:
            self.df.stopDataflow()
            self.df.clearDataflow()

        self.is_loaded = False
        self.emit(QtCore.SIGNAL('extensionChanged()'))
        self.hide()

    def load_dataflow(self, fname):
        name = fname[0].encode(sys.getdefaultencoding())
        if self.df is not None:
            self.df.loadDataflow(name, True)
            self.setWindowTitle('Dataflow: %s' % os.path.basename(name))
            self.is_loaded = True
            self.emit(QtCore.SIGNAL('extensionChanged()'))
            self.show()


class LoadDataflow(ExtensionBase):

    def register(self, mgr):
        win = self.context.get("win")
        if self.widget is None:
            self.widget = LoadDataflowWidget(win)

        action = QtGui.QAction('&Load Dataflow', win)
        action.setShortcut('Ctrl+L')
        action.setStatusTip('Load Dataflow')
        action.triggered.connect(self.widget.handle_load_dataflow)

        menu_item = dict(menu_id="file", menu_title="&File", menu_action=action)
        mgr.registerExtension("load_dataflow", self, category="file", menu_items=[menu_item,])

        return self

    def get_name(self):
        return "Dataflow"

    def get_ports(self):
        return []

    def get_port(self, name):
        raise ValueError("tbd")

