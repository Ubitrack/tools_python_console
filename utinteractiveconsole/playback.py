__author__ = 'jack'
from enaml.qt import QtCore
import numpy as np
import os
from collections import namedtuple

from ubitrack.core import math, util, measurement

import logging
log = logging.getLogger(__name__)




# Recorder Files helper

def fd(fname):
    return util.streambuf(open(fname, "r"), 1024)


DSC_ = namedtuple("DSC", ['fieldname', 'filename', 'reader', 'interpolator', "tsoffset", "syncgroup"])
def DSC(fieldname, filename, reader, interpolator=None, tsoffset=0.0, syncgroup=None):
    return DSC_(fieldname, filename, reader, interpolator, tsoffset, syncgroup)


def loadData(root_dir, reference, items=None):
    data_fieldnames = []
    data_items = []

    # XXX strange bug happening on linux :((
    # all the try again ... sections are attributed to this problem ..
    # clean up after the problem has been found ...
    # test available in ubitrack_python: test_utUtil.py

    ref_data = None
    ref_timestamps = None

    try:
        # load reference dataset
        ref_data = reference.reader(fd(os.path.join(root_dir, reference.filename)))
    except Exception, e:
        log.warn("error loading data, trying again ...")
        ref_data = reference.reader(fd(os.path.join(root_dir, reference.filename)))

    if ref_data is not None:
        try:
            ref_timestamps = np.asarray([p.time() for p in ref_data.values()])
        except Exception, e:
            log.warn("error reading timestamps from data, trying again ...")
            ref_timestamps = np.asarray([p.time() for p in ref_data.values()])



    # check timestamps
    if not np.all(np.diff(ref_timestamps) > 0):
        log.warn("Reference Timestamps are not ascending")

    data_fieldnames.append("timestamp")
    data_items.append(ref_timestamps)

    data_fieldnames.append(reference.fieldname)
    data_items.append([p.get() for p in ref_data.values()])



    # load additional datasets
    start_indexes = [0, ]
    for item in items:
        log.info("Loading data from recording file: %s" % item.filename)
        data_fieldnames.append(item.fieldname)
        records = None
        try:
            records_stream = item.reader(fd(os.path.join(root_dir, item.filename)))
        except Exception, e:
            log.warn("error reading stream for %s, trying again ..." % item.filename)
            records_stream = item.reader(fd(os.path.join(root_dir, item.filename)))

        try:
            records = [p for p in records_stream.values()]
        except Exception, e:
            log.warn("error reading items from stream, trying again ..." % item.filename)
            records = [p for p in records_stream.values()]

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



def selectNearestNeighbour(dest_ts, data, tsoffset=0.0):

    samples = []
    start_idx = 0

    src_ts = np.asarray([p.time() for p in data]) + tsoffset
    src_data = np.asarray([p.get() for p in data])

    for dts in dest_ts:

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
            samples.append(src_data[idx1])
        else:
            samples.append(src_data[idx2])

    return samples, start_idx


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

    len_src_ts = len(src_ts)

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

