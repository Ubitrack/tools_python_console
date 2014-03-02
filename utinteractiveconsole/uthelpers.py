__author__ = 'jack'
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from collections import namedtuple

from lxml import etree
import new

from atom.api import (Bool, List, Dict, observe, set_default, Unicode, Str, Enum, Int, Long, Atom, Value, Typed, Event)

from ubitrack.core import math, calibration, util, measurement
from ubitrack.dataflow import graph
from ubitrack.vision import vision
from ubitrack.visualization import visualization

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




# Recorder Files helper

DSC_ = namedtuple("DSC", ['fieldname', 'filename', 'reader', 'interpolator', "tsoffset", "syncgroup"])
def DSC(fieldname, filename, reader, interpolator=None, tsoffset=0.0, syncgroup=None):
    return DSC_(fieldname, filename, reader, interpolator, tsoffset, syncgroup)


def loadData(root_dir, reference, items=None):
    data_fieldnames = []
    data_items = []

    # load reference dataset
    ref_data = reference.reader(fd(os.path.join(root_dir, reference.filename))).values()

    ref_timestamps = np.asarray([p.time() for p in ref_data])

    # check timestamps
    if not np.all(np.diff(ref_timestamps) > 0):
        log.warn("Reference Timestamps are not ascending")

    data_fieldnames.append("timestamp")
    data_items.append(ref_timestamps)

    data_fieldnames.append(reference.fieldname)
    data_items.append([p.get() for p in ref_data])



    # load additional datasets
    start_indexes = [0, ]
    for item in items:
        data_fieldnames.append(item.fieldname)
        records = item.reader(fd(os.path.join(root_dir, item.filename))).values()
        if item.interpolator is not None:
            r_data, sidx = item.interpolator(ref_timestamps, records, item.tsoffset)
            start_indexes.append(sidx)
        else:
            # XXX should verify timestamps or number of samples ...
            r_data = [p.get() for p in records]

        data_items.append(r_data)

    start_index = np.max(np.asarray(start_indexes))
    stop_index = np.min(np.asarray([len(v) for v in data_items]))

    # create result dataset with timely aligned measurements
    DataSet = namedtuple('DataSet', data_fieldnames)

    all_data = []
    for i in range(start_index, stop_index):
        all_data.append(DataSet(*[e[i] for e in data_items]))

    return all_data




DataSetItem = namedtuple("DataSetItem", ["name", "values", "timestamps", "interval", "interpolator", "tsoffset", "syncgroup"])

def loadRawData(root_dir, items=None):
    data = {}
    sync_groups = {}

    # load additional datasets
    for item in items:
        if item.syncgroup:
            sync_groups.setdefault(item.syncgroup, []).append(item.fieldname)

        records = item.reader(fd(os.path.join(root_dir, item.filename))).values()

        ts_data = [p.time() for p in records]
        data[item.fieldname] = DataSetItem(item.fieldname, [p.get() for p in records], np.asarray(ts_data),
                                            int(np.diff(np.asarray(ts_data)).mean() / MS_DIVIDER),
                                            item.interpolator, item.tsoffset, item.syncgroup,
                                            )

    return data, sync_groups




def selectNearestNeighbour(dts, src_ts, src_data):

    idx = (np.abs(src_ts-dts)).argmin()
    if dts < src_ts[idx]:
        if idx == 0:
            ts1 = src_ts[idx]
            idx1 = idx
        else:
            ts1 = src_ts[idx-1]
            idx1 = idx-1
        ts2 = src_ts[idx]
        idx2 = idx
    else:
        ts1 = src_ts[idx]
        idx1 = idx
        if not idx+1 < len(src_ts):
            ts2 = src_ts[idx]
            idx2 = idx
        else:
            ts2 = src_ts[idx+1]
            idx2 = idx+1

    ediff = ts2 - ts1
    tdiff = dts - ts1

    if ediff != 0:
        h = float(tdiff) / float(ediff)
    else:
        h = 1.0

    if h < 0.5:
        return src_data[idx1]
    else:
        return src_data[idx2]



def interpolatePose(dts, src_ts, src_poses):

    idx = (np.abs(src_ts-dts)).argmin()
    if dts < src_ts[idx]:
        if idx == 0:
            ts1 = src_ts[idx]
            idx1 = idx
        else:
            ts1 = src_ts[idx-1]
            idx1 = idx-1
        ts2 = src_ts[idx]
        idx2 = idx
    else:
        ts1 = src_ts[idx]
        idx1 = idx
        if not idx+1 < len(src_ts):
            ts2 = src_ts[idx]
            idx2 = idx
        else:
            ts2 = src_ts[idx+1]
            idx2 = idx+1

    ediff = ts2 - ts1
    tdiff = dts - ts1

    if ediff != 0:
        h = float(tdiff) / float(ediff)
    else:
        h = 1.0

    return math.linearInterpolatePose(src_poses[idx1], src_poses[idx2], h)


def interpolateVec3(dts, src_ts, src_vecs):
    idx = (np.abs(src_ts-dts)).argmin()
    if dts < src_ts[idx]:
        if idx == 0:
            ts1 = src_ts[idx]
            idx1 = idx
        else:
            ts1 = src_ts[idx-1]
            idx1 = idx-1
        ts2 = src_ts[idx]
        idx2 = idx
    else:
        ts1 = src_ts[idx]
        idx1 = idx
        if not idx+1 < len(src_ts):
            ts2 = src_ts[idx]
            idx2 = idx
        else:
            ts2 = src_ts[idx+1]
            idx2 = idx+1

    ediff = ts2 - ts1
    tdiff = dts - ts1

    if ediff != 0:
        h = float(tdiff) / float(ediff)
    else:
        h = 1.0

    return math.linearInterpolateVector3(src_vecs[idx1], src_vecs[idx2], h)




def selectOnlyMatchingSamples(dest_ts, data, tsoffset=0.0):

    samples = []
    start_idx = 0

    src_ts = np.asarray([p.time() for p in data]) + tsoffset
    src_samples = np.asarray([p.get() for p in data])

    for dts in dest_ts:
        idx = (np.abs(src_ts-dts)).argmin()
        if src_ts[idx] == dts:
            samples.append(src_samples[idx])
        else:
            samples.append(None)
    return samples, start_idx



def interpolatePoseList(dest_ts, data, tsoffset=0.0):
    poses_intp = []
    start_idx = 0

    src_ts = np.asarray([p.time() for p in data]) + tsoffset
    src_poses = np.asarray([p.get() for p in data])

    # Linear Interpolation from UbiTrack component
    for dts in dest_ts:
        idx = (np.abs(src_ts-dts)).argmin()
        if idx == 0:
            poses_intp.append(None)
            start_idx += 1
            continue

        if dts < src_ts[idx]:
            ts1 = src_ts[idx-1]
            idx1 = idx-1
            ts2 = src_ts[idx]
            idx2 = idx
        else:
            ts1 = src_ts[idx]
            idx1 = idx
            ts2 = src_ts[idx+1]
            idx2 = idx+1

        ediff = ts2 - ts1
        tdiff = dts - ts1
        if ediff != 0:
            h = float(tdiff) / float(ediff)
        else:
            h = 1.0
        p = math.linearInterpolatePose(src_poses[idx1], src_poses[idx2], h)
        poses_intp.append(p)

    return poses_intp, start_idx

def interpolateVec3List(dest_ts, data, tsoffset=0.0):
    vecs_intp = []
    start_idx = 0

    src_ts = np.asarray([p.time() for p in data]) + tsoffset
    src_vecs = np.asarray([p.get() for p in data])

    len_src_ts = len(src_ts)

    # Linear Interpolation from UbiTrack component
    for dts in dest_ts:
        idx = (np.abs(src_ts-dts)).argmin()
        if idx == 0:
            vecs_intp.append(None)
            start_idx += 1
            continue

        if dts < src_ts[idx]:
            ts1 = src_ts[idx-1]
            idx1 = idx-1
            ts2 = src_ts[idx]
            idx2 = idx
        else:
            ts1 = src_ts[idx]
            idx1 = idx
            if not idx+1 < len_src_ts:
                ts2 = src_ts[idx]
                idx2 = idx
            else:
                ts2 = src_ts[idx+1]
                idx2 = idx+1

        ediff = ts2 - ts1
        tdiff = dts - ts1
        if ediff != 0:
            h = float(tdiff) / float(ediff)
        else:
            h = 1.0
        v = math.linearInterpolateVector3(src_vecs[idx1], src_vecs[idx2], h)
        vecs_intp.append(v)

    return vecs_intp, start_idx

