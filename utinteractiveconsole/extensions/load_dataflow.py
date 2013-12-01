__author__ = 'jack'
import os, sys
from pyqtgraph.Qt import QtCore, QtGui
from lxml import etree

from utinteractiveconsole.extensions import (ExtensionBase, PortInfo, InPort,
                                             OutPort, PORT_TYPE_SOURCE, PORT_TYPE_SINK,
                                             PORT_MODE_PUSH, PORT_MODE_PULL)
from ubitrack.dataflow import graph
from ubitrack.core import util

import logging
log = logging.getLogger(__name__)

class LoadDataflowWidget(QtGui.QWidget):

    def __init__(self, extension=None, parent=None):
        super(LoadDataflowWidget, self).__init__(parent)
        self.extension = extension
        self.df = self.extension.context.get("df")
        self.initUI()

        self.is_loaded = False
        self.dataflow_graph = None
        self.all_patterns = {}

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
        self.dataflow_graph = None
        self.emit(QtCore.SIGNAL('extensionChanged()'))
        self.hide()

    def load_dataflow(self, fname):
        name = fname[0].encode(sys.getdefaultencoding())
        self.parseUTQL(name)
        if self.df is not None:
            self.df.loadDataflow(name, True)
            self.setWindowTitle('Dataflow: %s' % os.path.basename(name))
            self.is_loaded = True
            self.emit(QtCore.SIGNAL('extensionChanged()'))
            self.show()

    def parseUTQL(self, name):
        try:
            self.dataflow_graph = graph.readUTQLDocument(util.streambuf(open(name, "r"), 1024))
        except Exception, e:
            log.exception(e)
            self.dataflow_graph = None

        if self.dataflow_graph is not None:
            dfg = self.dataflow_graph
            self.all_patterns = {}
            for k,pat in dfg.SubgraphById.items():
                config = {}
                xml = pat.DataflowConfiguration.getXML()
                try:
                    doc = etree.XML(xml)
                    config["class"] = doc.xpath("/root/DataflowConfiguration/UbitrackLib")[0].attrib["class"]
                    config["attrs"] = dict((e.attrib["name"], e.attrib["value"]) for e in doc.xpath("/root/DataflowConfiguration/Attribute"))
                except Exception, e:
                    log.exception(e)
                self.all_patterns[k] = config


    def get_ports(self):
        if not self.is_loaded:
            return []

        ports = []
        for k, cfg in self.all_patterns.items():
            mode = None
            port_type = None
            queued = False
            type_name = None

            if "class" in cfg:
                if cfg["class"].startswith("ApplicationPushSink"):
                    port_type = PORT_TYPE_SINK
                    mode = PORT_MODE_PUSH
                    queued = True
                    type_name = cfg["class"][19:]
                elif cfg["class"].startswith("ApplicationPullSink"):
                    port_type = PORT_TYPE_SINK
                    mode = PORT_MODE_PULL
                    type_name = cfg["class"][19:]
                elif cfg["class"].startswith("ApplicationPushSource"):
                    port_type = PORT_TYPE_SOURCE
                    mode = PORT_MODE_PUSH
                    type_name = cfg["class"][21:]
                elif cfg["class"].startswith("ApplicationPullSource"):
                    port_type = PORT_TYPE_SOURCE
                    mode = PORT_MODE_PULL
                    type_name = cfg["class"][21:]

            if port_type is not None and mode is not None:
                ports.append(PortInfo(k, port_type, mode, type_name, queued))
        return ports



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
        return self.widget.get_ports()

    def get_port(self, name):
        raise ValueError("tbd")

