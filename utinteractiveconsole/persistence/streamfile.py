__author__ = 'jack'

from atom.api import Atom, List, Str, Int, Enum, Bool, Typed, Value, Coerced

import os
import logging

from ubitrack.core import util, math, measurement
import numpy as np

log = logging.getLogger(__name__)

UBITRACK_DATATYPES = ['distance', 'position2d', 'position3d', 'quat', 'pose', 'mat33', 'mat34', 'mat44']
MS_DIVIDER = 1000000.0

def get_streamreader_for_datatype(dtype, is_array=False):
    if dtype == "distance":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
    elif dtype == "position2d":
        if is_array:
            return util.PositionList2StreamReader
        else:
            return util.Position2DStreamReader
    elif dtype == "position3d":
        if is_array:
            return util.PositionListStreamReader
        else:
            return util.PositionStreamReader
    elif dtype == "quat":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            return util.RotationStreamReader
    elif dtype == "pose":
        if is_array:
            return util.PoseListStreamReader
        else:
            return util.PoseStreamReader
    elif dtype == "mat33":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            return util.Matrix3x3StreamReader
    elif dtype == "mat34":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            return util.Matrix3x4StreamReader
    elif dtype == "mat44":
        if is_array:
            raise NotImplemented("No streamreader available for: %s%s" % (dtype, "-list" if is_array else ""))
        else:
            return util.Matrix4x4StreamReader
    else:
        raise ValueError("Unknown datatype: %s" % dtype)


class StreamFileSpec(Atom):
    fieldname = Str()
    filename = Str()
    datatype = Enum(*UBITRACK_DATATYPES)
    is_array = Bool(False)

    def getStreamFile(self):
        if not os.path.isfile(self.filename):
            raise ValueError("Invalid filename: %s" % self.filename)

        reader = get_streamreader_for_datatype(self.datatype, self.is_array)
        return StreamFile(spec=self, filename=self.filename, reader=reader)


class StreamFile(Atom):

    spec = Typed(StreamFileSpec)
    filename = Str()
    reader = Value()

    raw_data = Value()
    timestamps = Coerced(np.ndarray)
    count = Int()
    interval = Int()
    values = List()

    interpolator = Value(None)

    @property
    def fieldname(self):
        return self.spec.fieldname

    @property
    def datatype(self):
        return self.spec.datatype

    @property
    def is_array(self):
        return self.spec.is_array

    def _default_raw_data(self):
        fdesc = open(self.filename, "r")
        fdesc.seek(0)
        buf = util.streambuf(fdesc, 1024)
        log.info("Reading StreamFile: %s" % self.spec.filename)
        data = self.reader(buf).values()
        fdesc.close()
        return data

    def _default_timestamps(self):
        timestamps = np.array([m.time() for m in self.raw_data])
        if not np.all(np.diff(timestamps) > 0):
            log.warn("Timestamps in stream: %s are not ascending" % self.fieldname)
        return timestamps

    def _default_interval(self):
        return int(np.diff(np.asarray(self.timestamps)).mean() / MS_DIVIDER)

    def _default_count(self):
        return len(self.timestamps)

    def _default_values(self):
        return [m.get() for m in self.raw_data]

    def _default_interpolator(self):
        if not self.is_array:
            if self.datatype == "position2d":
                return math.linearInterpolateVector2
            elif self.datatype == "position3d":
                return math.linearInterpolateVector3
            elif self.datatype == "quat":
                return math.linearInterpolateQuaternion
            elif self.datatype == "pose":
                return math.linearInterpolatePose
        return None

    def get(self, dts, selector=None):
        src_ts = self.timestamps
        idx = (np.abs(src_ts-dts)).argmin()

        # None == select only matching
        if selector is None or selector == "matching":
            if src_ts[idx] == dts:
                return self.values[idx]
            else:
                return None

        # interpolation or nearest neighbors
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
            if not idx+1 < self.count:
                ts2 = src_ts[idx]
                idx2 = idx
            else:
                ts2 = src_ts[idx+1]
                idx2 = idx+1

        ediff = ts2 - ts1
        tdiff = dts - ts1

        h = 1.0
        if ediff != 0:
            h = float(tdiff) / float(ediff)

        if selector == "nearest":
            if h < 0.5:
                return self.values[idx1]
            else:
                return self.values[idx2]

        if self.interpolator is not None and selector == "interpolate":
            return self.interpolator(self.values[idx1], self.values[idx2], h)

        raise NotImplemented("Interpolation for %s%s is not implemented." % (self.datatype, "-list" if self.is_array else ""))

