__author__ = 'jack'
import os
import sys
import logging

import enaml
from enaml.qt import QtCore, QtGui
from enaml.workbench.ui.api import ActionItem
from enaml.workbench.api import Extension
from ubitrack.dataflow import graph
from ubitrack.core import util

from utinteractiveconsole.extensions import ExtensionBase
from utinteractiveconsole.workspace import ExtensionWorkspace
from utinteractiveconsole.uthelpers import PortInfo, PORT_MODE_PULL, PORT_MODE_PUSH, PORT_TYPE_SINK, PORT_TYPE_SOURCE

log = logging.getLogger(__name__)



class RemoteControl(ExtensionBase):

    def update_optparse(self, parser):
        parser.add_option("-R", "--enable-remote-control",
                      action="store_true", dest="enable_remote_control", default=False,
                      help="Enable remote control via HTTP/Rest server")


    def register(self, mgr):
        self.network_initialized = False
        self.current_site = None
        self.current_port = None

        appstate = self.context.get("appstate")
        if appstate is not None and appstate.options.enable_remote_control:
            appstate.observe("workspace_started", self.handle_workspace_started)
            appstate.observe("workspace_stopped", self.handle_workspace_stopped)

            name = "remote_control"
            category = "util"
            mgr.registerExtension(name, self, category=category, action_items=[], workspace_plugins=[],
                                  add_globals=dict(module_remote_control=self))
        return self

    def handle_workspace_started(self, change):
        workspace = change['value']
        if workspace._manifest_id == 'uticmain.calibration_wizard':
            log.info("Initialize Remote Control")
            if self.initialize_networking():
                from .rest_server import reactor, Site, RemoteControlAPIServer
                self.current_site = Site(RemoteControlAPIServer(self.context, workspace))
                self.current_port = reactor.listenTCP(8080, self.current_site)


    def handle_workspace_stopped(self, change):
        workspace = change['value']
        if workspace._manifest_id == 'uticmain.calibration_wizard':
            log.info("Teardown Remote Control")
            if self.current_port is not None:
                self.current_port.stopListening()
            self.current_port = None
            self.current_site = None


    def initialize_networking(self):
        if not self.network_initialized:
            try:
                from utinteractiveconsole import qt4reactor
                # install twisted qt4 reactor
                qt4reactor.install()
                self.network_initialized = True
                log.info("Initialized Networking support")
            except ImportError, e:
                log.info("Error while initializing networking components")
                log.error(e)
                self.network_initialized = False

        return self.network_initialized


    def get_name(self):
        return "RemoteControl"

    def get_ports(self):
        return []

    def get_port(self, name):
        raise ValueError("tbd")

