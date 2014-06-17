__author__ = 'jack'
import os, sys
import logging

import enaml
from enaml.qt import QtCore, QtGui
from lxml import etree

from utinteractiveconsole.extensions import ExtensionBase, ExtensionWorkspace
from utinteractiveconsole.uthelpers import PortInfo, PORT_MODE_PULL, PORT_MODE_PUSH, PORT_TYPE_SINK, PORT_TYPE_SOURCE

from ubitrack.dataflow import graph
from ubitrack.core import util

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

    def update_optparse(self, parser):
        print "init parser options"
        default_componenents_path = "/usr/local/lib/ubitrack"
        if sys.platform.startswith("win32") and "UBITRACK_PATH" in os.environ:
            default_componenents_path = os.path.join(os.environ["UBITRACK_PATH"], "ubitrack")

        parser.add_option("-a", "--autostart",
                      action="store_false", dest="autostart", default=True,
                      help="automatically start dataflow?")

        parser.add_option("-f", "--facade",
                      action="store_true", dest="facade", default=False,
                      help="automatically load facade?")

        parser.add_option("-c", "--components_path",
                      action="store", dest="components_path", default=default_componenents_path,
                      help="path to UbiTrack components")

    def register(self, mgr):

        name = "load_dataflow"
        category = "util"
        #win = self.context.get("win")
        #if self.widget is None:
        #    self.widget = LoadDataflowWidget(win)

        def plugin_factory(workbench):

            with enaml.imports():
                from utinteractiveconsole.extensions.views.load_dataflow import LoadDataflowMain, LoadDataflowManifest

            space = ExtensionWorkspace(appstate=mgr.appstate, utic_plugin=self)
            space.window_title = 'Load Dataflow'
            space.content_def = LoadDataflowMain
            space.manifest_def = LoadDataflowManifest
            return space

        plugin = dict(id=name,
                         point="enaml.workbench.ui.workspaces",
                         factory=plugin_factory)


        action = dict(path="/workspace/load_dataflow",
                      label="Load Dataflow",
                      shortcut= "Ctrl+L",
                      group="spaces",
                      command="enaml.workbench.ui.select_workspace",
                      parameters= {'workspace': "utic.%s" % name, }
                      )

        mgr.registerExtension(name, self, category=category, action_items=[action,], workspace_plugins=[plugin,])
        return self

    def get_name(self):
        return "Dataflow"

    def get_ports(self):
        return self.widget.get_ports()

    def get_port(self, name):
        raise ValueError("tbd")

