__author__ = 'jack'
import os, sys

from enaml.qt import QtCore
from collections import namedtuple

from lxml import etree
import new

from atom.catom import Member
from atom.api import (List, Dict, Str, Long, Atom, Value, Event, Bool, observe)

# import all ubitrack modules to setup boost-python inline converters
from ubitrack.core import math, calibration, util, measurement
from ubitrack.facade import facade
from ubitrack.dataflow import graph
from ubitrack.vision import vision
from ubitrack.visualization import visualization

from .subprocess import SubProcessManager


import logging
log = logging.getLogger(__name__)


PortInfo = namedtuple("PortInfo", ["name", "port_type", "mode", "data_type", "queued"])

PORT_TYPE_SOURCE = 0
PORT_TYPE_SINK = 1

PORT_MODE_PUSH = 0
PORT_MODE_PULL = 1

class NoValueException(ValueError):
    pass


class PushSinkAdapter(QtCore.QObject):

    def __init__(self, sink):
        super(PushSinkAdapter, self).__init__()

        self.sink = sink
        self._time = None
        self._value = None


    def cb_handler(self, m):
        try:
            self._time = m.time()
            self._value = m
            self.emit(QtCore.SIGNAL('dataReady(PyQt_PyObject)'), self._time)
        except Exception, e:
            log.exception(e)

    def connect(self, handler):
        self.sink.setCallback(self.cb_handler)
        return super(PushSinkAdapter, self).connect(self, QtCore.SIGNAL('dataReady(PyQt_PyObject)'), handler, QtCore.Qt.QueuedConnection)

    def disconnect(self, handler):
        self.sink.setCallback(None)
        return super(PushSinkAdapter, self).disconnect(self, QtCore.SIGNAL('dataReady(PyQt_PyObject)'), handler)

    def get(self, ts=None):
        if (ts is not None and ts != self._time) or self._value is None:
            raise NoValueException
        return self._value

    @property
    def value(self):
        return self._value


    @property
    def time(self):
        return self._time





class PullSinkAdapter(QtCore.QObject):

    def __init__(self, sink):
        super(PullSinkAdapter, self).__init__()
        self.sink = sink


    def get(self, ts=None):
        if ts is None:
            raise NoValueException
        try:
            return self.sink.get(ts)
        except Exception, e:
            #log ?
            raise NoValueException


class PushSourceAdapter(QtCore.QObject):

    converters = {
        (str, "Button"): lambda ts, v: measurement.Button(ts, math.ScalarInt(ord(v))),
        (int, "Button"): lambda ts, v: measurement.Button(ts, math.ScalarInt(v)),
        # more converters to come
    }


    def __init__(self, source, datatype):
        super(PushSourceAdapter, self).__init__()
        self.source = source
        self.datatype = datatype

    def send(self, v):
        ts = measurement.now()
        vtyp = type(v)
        ckey = (vtyp, self.datatype)
        if ckey in self.converters:
            m = self.converters[ckey](ts, v)
            self.source.send(m)
        else:
            log.warn("No converter found for datatype: %s" % self.datatype)


class UbitrackConnectorBase(Atom):

    sync_source = Str()
    adapters = Dict()
    current_timestamp = Long()

    def setup(self, facade):
        if not hasattr(self, "ports"):
            raise TypeError("UbitrackConnector is not set up correctly: ports attribute is missing.")

        log.info("Setup Ubitrack Connector with %d ports and sync source: %s" % (len(self.ports), self.sync_source))
        adapters = {}
        for pi in self.ports:
            if pi.port_type == PORT_TYPE_SINK:
                if pi.mode == PORT_MODE_PUSH:
                    accessor_name = "getApplicationPushSink%s" % pi.data_type
                    if hasattr(facade, accessor_name):
                        accessor = getattr(facade, accessor_name)(pi.name)
                        adapters[pi.name] = PushSinkAdapter(accessor)
                        if pi.name == self.sync_source:
                            log.info("Connect %s as sync source" % pi.name)
                            adapters[pi.name].connect(self.handleSinkData)
                    else:
                        log.warn("Missing accessor %s from facade, cannot connect PushSink %s" % (accessor_name, pi.name))
                else:
                    accessor_name = "getApplicationPullSink%s" % pi.data_type
                    if hasattr(facade, accessor_name):
                        accessor = getattr(facade, accessor_name)(pi.name)
                        adapters[pi.name] = PullSinkAdapter(accessor)
                    else:
                        log.warn("Missing accessor %s from facade, cannot connect PullSink %s" % (accessor_name, pi.name))
            else:
                if pi.mode == PORT_MODE_PUSH:
                    accessor_name = "getApplicationPushSource%s" % pi.data_type
                    if hasattr(facade, accessor_name):
                        accessor = getattr(facade, accessor_name)(pi.name)
                        adapters[pi.name] = PushSourceAdapter(accessor, pi.data_type)
                        self.observe(pi.name, self.handleSourceData)
                    else:
                        log.warn("Missing accessor %s from facade, cannot connect PushSource %s" % (accessor_name, pi.name))

        self.adapters = adapters

    def teardown(self, facade):
        log.info("Teardown Ubitrack Connector")
        # XXX implement connector teardown



    def handleSinkData(self, ts):
        for pi in self.ports:
            if pi.port_type == PORT_TYPE_SINK:
                try:
                    val = self.adapters[pi.name].get(ts)
                    setattr(self, pi.name, val)
                except NoValueException, e:
                    log.debug("No value from sink: %s for timestamp: %d" % (pi.name, ts))
                except Exception, e:
                    log.error("Error while retrieving a value from sink: %s for timestamp: %d" % (pi.name, ts))
                    log.exception(e)
        self.current_timestamp = ts


    def handleSourceData(self, change):
        port_name = change["name"]
        port_value = change["value"]
        if port_name in self.adapters:
            adapter = self.adapters[port_name]
            adapter.send(port_value)



def ubitrack_connector_class(dfg_filename):
    all_patterns = {}

    try:
        dfg = graph.readUTQLDocument(util.streambuf(open(dfg_filename, "r"), 1024))
    except Exception, e:
        log.exception(e)
        dfg = None

    if dfg is not None:
        for k, pat in dfg.SubgraphById.items():
            config = dict()
            config["class"] = pat.Name
            xml = pat.DataflowConfiguration.getXML()
            try:
                doc = etree.XML(xml)
                # alternative way to find class: doc.xpath("/root/DataflowConfiguration/UbitrackLib")[0].attrib["class"]
                config["attrs"] = dict((e.attrib["name"], e.attrib["value"]) for e in doc.xpath("/root/DataflowConfiguration/Attribute"))
            except Exception, e:
                log.exception(e)
            all_patterns[k] = config

    attrs = {}
    ports = []

    for k, cfg in all_patterns.items():
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
            if port_type == PORT_TYPE_SOURCE and mode == PORT_MODE_PUSH:
                attrs[k] = Event()
            else:
                attrs[k] = Value()

    attrs["ports"] = List(default=ports)

    return new.classobj("UbitrackConnector", (UbitrackConnectorBase,), attrs)






class UbitrackFacadeBase(Atom):
    context = Member()
    components_path = Str()
    instance = Member()
    dfg_basedir = Str()
    dfg_filename = Str()

    is_loaded = Bool()
    is_running = Bool()

    def start(self):
        pass

    def stop(self):
        pass

    def _default_components_path(self):
        cpath = None
        if self.context is not None and self.context.get("config") is not None:
            cfg = self.context.get("config")
            if cfg.has_section("ubitrack"):
                ut_cfg = dict(cfg.items("ubitrack"))
                cpath = ut_cfg.get("components_path")
        if cpath is None:
            cpath = os.environ.get("UBITRACK_COMPONENTS_PATH", None)
            if cpath is None:
                log.warn("Missing UBITRACK_COMPONENTS_PATH environment variable")
        return cpath

    def _default_dfg_basedir(self):
        basedir = None
        if self.context is not None and self.context.get("config") is not None:
            cfg = self.context.get("config")
            root_dir = None
            srg_dir = None
            if cfg.has_section("vharcalib"):
                vc_cfg = dict(cfg.items("vharcalib"))
                root_dir = vc_cfg["rootdir"]
                srg_dir = vc_cfg["srgdir"]
            basedir = os.path.join(*filter(None, [root_dir, srg_dir]))
        else:
            osname = None
            if sys.platform.startswith("win"):
                osname = "win"
            elif sys.platform.startswith("linux"):
                osname = "linux"
            elif sys.platform.startswith("darwin"):
                osname = "mac"
            if osname is None:
                raise ValueError("Unkown Operating System: %s " % sys.platform)

            # XXX this does not work anymore ...
            basedir = os.path.join(os.path.dirname(__file__), "srgs", osname)
        if basedir is None or not os.path.isdir(basedir):
            log.error("Invalid Basedir for CalibrationModules: %s" % basedir)
        return basedir


    @observe("dfg_filename")
    def _handle_dfg_change(self, change):
        if self.is_loaded:
            fname = change["value"]
            self.stopDataflow()
            self.loadDataflow(fname)

    def loadDataflow(self, fname, force_reload=False):
        if self.is_loaded and not force_reload:
            return

        if fname is not None:
            if not os.path.isfile(fname):
                fname = os.path.join(self.dfg_basedir, fname)
                if not os.path.isfile(fname):
                    log.error("Invalid DFG filename: %s" % fname)
                    return
            log.info("Load DFG: %s" % fname)
            self.instance.loadDataflow(fname, True)
            self.is_loaded = True

    def startDataflow(self):
        if not self.is_loaded:
            if self.dfg_filename:
                self.loadDataflow(self.dfg_filename)

        if not self.is_running:
            log.info("Start Dataflow")
            self.instance.startDataflow()
            self.is_running = True

    def stopDataflow(self):
        if not self.is_loaded:
            return

        if self.is_running:
            log.info("Stop Dataflow")
            self.instance.stopDataflow()
            self.is_running = False

    def clearDataflow(self):
        self.instance.clearDataflow()
        self.is_loaded = False


    def cleanup(self):
        self.stopDataflow()
        self.clearDataflow()
        self.instance.killEverything()
        self.instance = self._default_instance()



class UbitrackFacade(UbitrackFacadeBase):


    def _default_instance(self):
        log.info("Create UbiTrack facade")
        return facade.AdvancedFacade(self.components_path)


class UbitrackSubProcessFacade(UbitrackFacadeBase):

    def start(self):
        self.instance.start()

    def stop(self):
        self.instance.stop()

    def restart(self, autostart=True):
        self.instance.restart(autostart=autostart)

    def get_messages(self, timeout=0):
        return self.instance.get_messages(timeout=timeout)

    def is_alive(self):
        return self.instance.is_alive()

    def _default_instance(self):
        log.info("Create UbiTrack SubProcess facade")
        return SubProcessManager("calibration_wizard_slave", components_path=self.components_path)

