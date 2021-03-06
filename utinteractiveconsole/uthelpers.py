__author__ = 'jack'
import os, sys

from enaml.qt import QtCore
from collections import namedtuple

from lxml import etree
import new

from atom.catom import Member
from atom.api import (List, Dict, Str, Long, Typed, Atom, Value, Event, Bool, observe)

# import all ubitrack modules to setup boost-python inline converters
from ubitrack.core import math, calibration, util, measurement
from ubitrack.facade import facade
from ubitrack.dataflow import graph
try:
    from ubitrack.vision import vision
except ImportError, e:
    print "Problem importing utvision"

try:
    from ubitrack.visualization import visualization
except ImportError, e:
    print "Problem importing utvisualization"

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

    dataReady = QtCore.Signal(['PyQt_PyObject'], name='dataReady')

    def __init__(self, sink):
        super(PushSinkAdapter, self).__init__()

        self.sink = sink
        self._time = None
        self._value = None


    def cb_handler(self, m):
        try:
            self._time = m.time()
            self._value = m
            self.dataReady.emit(self._time)
        except Exception, e:
            log.exception(e)

    def connect(self, handler):
        self.sink.setCallback(self.cb_handler)
        return self.dataReady.connect(handler, QtCore.Qt.QueuedConnection)

    def disconnect(self, handler):
        self.sink.setCallback(None)
        return self.dataReady.disconnect(handler)

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



class SourceAdapterBase(QtCore.QObject):

    converters = {
        (str, "Button"): lambda ts, v: measurement.Button(ts, math.ScalarInt(ord(v))),
        (int, "Button"): lambda ts, v: measurement.Button(ts, math.ScalarInt(v)),
        # more converters to come
    }


    def __init__(self, source, datatype):
        super(SourceAdapterBase, self).__init__()
        self.source = source
        self.datatype = datatype



class PushSourceAdapter(SourceAdapterBase):

    def send(self, v):
        ts = measurement.now()
        vtyp = type(v)
        ckey = (vtyp, self.datatype)
        if ckey in self.converters:
            m = self.converters[ckey](ts, v)
            self.source.send(m)
        else:
            log.warn("No converter found for datatype: %s" % self.datatype)


class PullSourceAdapter(SourceAdapterBase):

    def __init__(self, source, datatype):
        super(SourceAdapterBase, self).__init__(source, datatype)
        self._time = None
        self._value = None

    def cb_handler(self, ts):
        if self._value is None:
            raise NoValueException("Value for: %s has not been set yet." % self.datatype)
        vtyp = type(self._value)
        ckey = (vtyp, self.datatype)
        # XXX self._time vs ts .. what is correct behaviour ?
        # do we need to be able to configure this ?
        if ckey in self.converters:
            m = self.converters[ckey](ts, self._value)
            return m
        else:
            log.warn("No converter found for datatype: %s" % self.datatype)

    def connect(self, handler=None):
        handler = handler if handler is not None else self.cb_handler
        self.source.setCallback(handler)

    def disconnect(self, handler=None):
        self.sink.setCallback(None)

    def send(self, value):
        self._time = measurement.now()
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def time(self):
        return self._time



class UbitrackConnectorBase(Atom):

    sync_source = Str()
    adapters = Dict()
    current_timestamp = Long()

    update_ignore_ports = List()

    def setup(self, facade, update_ignore_ports=None):
        if not hasattr(self, "ports"):
            raise TypeError("UbitrackConnector is not set up correctly: ports attribute is missing.")

        if update_ignore_ports is not None:
            self.update_ignore_ports = update_ignore_ports

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

                elif pi.mode == PORT_MODE_PULL:
                    accessor_name = "getApplicationPullSource%s" % pi.data_type
                    if hasattr(facade, accessor_name):
                        accessor = getattr(facade, accessor_name)(pi.name)
                        adapters[pi.name] = PullSourceAdapter(accessor, pi.data_type)
                        # use default pull implementation for now.
                        adapters[pi.name].connect()
                        self.observe(pi.name, self.handleSourceData)
                    else:
                        log.warn("Missing accessor %s from facade, cannot connect PushSource %s" % (accessor_name, pi.name))

        self.adapters = adapters

    def teardown(self, facade):
        log.info("Teardown Ubitrack Connector")
        for pi in self.ports:
            if pi.name not in self.adapters:
                continue
            if pi.port_type == PORT_TYPE_SINK:
                if pi.mode == PORT_MODE_PUSH:
                    self.adapters[pi.name].disconnect(self.handleSinkData)
            else:
                if pi.mode == PORT_MODE_PULL:
                    self.adapters[pi.name].disconnect(self.handleSourceData)

        self.adapters = {}


    def handleSinkData(self, ts):
        try:
            uip = self.update_ignore_ports
        except Exception, e:
            log.error("Problem with observer setup in uthelpers connector (self: %s)" % (self,))
            log.exception(e)
            return

        for pi in self.ports:
            if pi.name in uip:
                continue
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

    # needed to mitigiate a strange error that weakreferences are not supported
    attrs['__slots__'] = ('__weakref__',)
    attrs["ports"] = List()
    attrs["_default_ports"] = lambda s: ports

    return new.classobj("UbitrackConnector", (UbitrackConnectorBase,), attrs)






class UbitrackFacadeBase(Atom):
    context = Member()
    components_path = Str()
    instance = Member()
    dfg_filename = Str()

    is_loaded = Bool()
    is_running = Bool()

    def start(self):
        pass

    def stop(self):
        pass

    def restart(self, autostart=True):
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

    @observe("dfg_filename")
    def _handle_dfg_change(self, change):
        if self.is_loaded:
            fname = change["value"]
            self.stopDataflow()
            self.loadDataflow(fname)

    def loadDataflow(self, fname, force_reload=False):
        if self.is_loaded and not force_reload:
            return

        if fname in [None, '']:
            log.warn("loadDataflow called without filename")
            return

        # XXX expand user required here ?
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

    def get_messages(self, timeout=0):
        return []

    def is_alive(self):
        return True


class UbitrackFacade(UbitrackFacadeBase):


    def _default_instance(self):
        log.info("Create InProcess UbiTrack facade")
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
        log.info("Create SubProcess UbiTrack facade")
        return SubProcessManager("calibration_wizard_slave", components_path=self.components_path)

