__author__ = 'jack'

from atom.api import Atom, List, Str, Enum, Bool, Float, Typed, Value, Coerced

import numpy as np
import new
import logging

from .streamfile import StreamFile, StreamFileSpec
from .recordschema import RecordSchema, Field

RECORD_SELECTORS = ['matching', 'nearest', 'interpolate']
log = logging.getLogger(__name__)



class StreamInterpolator(Atom):
    filespec = Typed(StreamFileSpec)
    is_reference = Bool(False)
    selector = Enum(*RECORD_SELECTORS)
    latency = Float(0.0)

    streamfile = Typed(StreamFile)

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
        return self.streamfile.get(dts - self.latency, self.selector)


class RecordSource(Atom):
    name = Str()
    title = Str()
    fields = List()
    reference_field = Typed(StreamInterpolator)

    schema = Typed(RecordSchema)
    output_fieldnames = List()

    record_class = Value()
    ignore_incomplete_records = Bool(True)

    cached_records = List()

    @property
    def max_record_count(self):
        if self.reference_field is not None:
            return self.reference_field.record_count
        return None

    @property
    def reference_timestamps(self):
        if self.reference_field is not None:
            return self.reference_field.timestamps
        return None

    @property
    def reference_interval(self):
        if self.reference_field is not None:
            return self.reference_field.interval
        return None

    def _default_output_fieldnames(self):
        return ['timestamp'] + [f.name for f in self.schema.fields]

    def _default_schema(self):
        schema_fields = []
        for field in self.fields:
            schema_fields.append(Field(name=field.fieldname,
                                       datatype=field.datatype,
                                       is_array=field.is_array,
                                       selector=field.selector,
                                       is_computed=False,
                                       is_reference=field.is_reference,
                                       latency=field.latency))
        return RecordSchema(fields=schema_fields)

    def _default_record_class(self):
        attrs = dict(timestamp=Coerced(np.int64),)
        for field in self.schema.fields:
            if field.name in attrs:
                log.warn("Duplicate key: %s in field specification for record source: %s - skipping" % (field.name, self.name))
                continue
            attrs[field.name] = Value()
        return new.classobj("RecordClass_%s" % self.name, (Atom,), attrs)

    def _default_reference_field(self):
        for field in self.fields:
            if field.is_reference:
                return field
        raise ValueError("No reference defined for record source: %s" % self.name)

    def clear_cache(self):
        self.cached_records = []

    def __iter__(self):

        if len(self.cached_records) > 0:
            for cached_record in self.cached_records:
                yield cached_record
            return

        rcls = self.record_class
        fields = self.fields
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

            record = rcls(**attrs)
            self.cached_records.append(record)
            yield record

