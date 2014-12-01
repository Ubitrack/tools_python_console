__author__ = 'jack'

from atom.api import Atom, List, Str, Enum, Bool, Float, Typed, Value, Long

import new
import logging

from .streamfile import StreamFile, StreamFileSpec

log = logging.getLogger(__name__)


class FieldInterpolator(Atom):
    filespec = Typed(StreamFileSpec)
    is_reference = Bool(False)
    interpolator = Enum(['matching', 'nearest', 'interpolate'])
    latency = Float(0.0)

    streamfile = Typed(StreamFile)

    @property
    def fieldname(self):
        return self.filespec.fieldname

    @property
    def fieldname(self):
        return self.filespec.fieldname

    @property
    def datatype(self):
        return self.filespec.datatype

    @property
    def is_array(self):
        return self.filespec.is_array

    @property
    def timestamps(self):
        return self.streamfile.timestamps

    @property
    def interval(self):
        return self.streamfile.interval

    @property
    def record_count(self):
        return self.streamfile.count

    def _default_streamfile(self):
        return self.filespec.getStreamFile()

    def get(self, dts):
        return self.streamfile.get(dts - self.latency, self.interpolator)


class RecordSource(Atom):
    name = Str()
    fieldspec = List()

    ignore_incomplete_records = Bool(True)

    fieldnames = List()
    record_class = Value()
    reference_field = Typed(FieldInterpolator)

    @property
    def max_record_count(self):
        if self.reference_field is not None:
            return self.reference_field.record_count
        return None

    @property
    def reference_timestamps(self):
        if self.reference_field is not None:
            return self.reference_field.interval
        return None

    @property
    def reference_interval(self):
        if self.reference_field is not None:
            return self.reference_field.timestamps
        return None


    def _default_fieldnames(self):
        return [f.fieldname for f in self.fieldspec]

    def _default_record_class(self):
        attrs = dict(timestamp=Long(),)
        for field in self.fieldspec:
            if field.fieldname in attrs:
                log.warn("Duplicate key: %s in field specification for record source: %s - skipping" % (field.fieldname, self.name))
                continue
            attrs[field.fieldname] = Value()
        return new.classobj("RecordClass_%s" % self.name, (Atom,), attrs)

    def _default_reference_field(self):
        reference = None
        for field in self.fieldspec:
            if field.is_reference:
                return field
        raise ValueError("No reference defined for record source: %s" % self.name)

    def __iter__(self):
        rcls = self.record_class
        fields = self.fieldspec
        reference = self.reference_field
        iir = self.ignore_incomplete_records

        for ts in reference.timestamps:

            record_complete = True
            attrs = dict(timestamp=ts)
            for field in fields:
                value = field.get(ts)
                if value is None:
                    record_complete = False
                attrs[field.fieldname] = value

            if iir and not record_complete:
                log.warn("Skipping incomplete record for timestamp: %s in record source: %s" % (ts, self.name))
                continue

            yield rcls(**attrs)

