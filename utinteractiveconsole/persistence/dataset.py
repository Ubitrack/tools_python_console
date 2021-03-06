__author__ = 'jack'

from atom.api import Atom, Value, Dict, List, Str, Float, Typed

import logging
import new

from .recordsource import BaseRecordSource

log = logging.getLogger(__name__)


class DataSet(Atom):

    name = Str()
    title = Str()

    recordsource = Typed(BaseRecordSource)
    processor_factory = Value()
    stream_filters = List()
    attributes = Dict()

    processor = Value()
    reference_timestamps = List()
    interval = Float()

    connector_class = Value()

    cached_records = List()

    @property
    def output_fieldnames(self):
        return self.recordsource.output_fieldnames

    def _default_reference_timestamps(self):
        return self.recordsource.reference_timestamps

    def _default_interval(self):
        return self.recordsource.reference_interval

    def _default_processor(self):
        p_cls = self.processor_factory
        p_kw = dict(recordsource=self.recordsource)
        config_complete = True
        for attr in p_cls.required_attributes:
            if attr in self.attributes:
                p_kw[attr] = self.attributes[attr]
            else:
                log.warn("Missing attribute: %s for streamprocessor in datasource: %s" % (attr, self.name))
                config_complete = False

        if config_complete:
            return p_cls(**p_kw)
        return None        

    def _default_connector_class(self):
        attrs = {}
        names = self.processor.output_fieldnames

        for k in names:
            attrs[k] = Value()

        attrs["names"] = List(default=names)

        def update_data(s, ds):
            for key in names:
                value = getattr(ds, key)
                if hasattr(s, key):
                    setattr(s, key, value)

        attrs['__call__'] = update_data
        return new.classobj("DataConnector_%s_%s" % (self.name, self.processor.name), (Atom,), attrs)

    def clear_cache(self, clear_recordsource=True):
        if clear_recordsource:
            self.recordsource.clear_cache()
        self.cached_records = []

    def __iter__(self):
        if len(self.cached_records) > 0:
            for cached_record in self.cached_records:
                yield cached_record
            return

        if self.processor is None:
            raise StopIteration()

        # apply stream filters
        producer = self.processor
        for sfilter in self.stream_filters:
            producer = sfilter.process(producer)

        for record in producer:
            self.cached_records.append(record)
            yield record

    def export_data(self, store, base_path, skip_fields=None, stream_filters=None):
        import pandas as pd
        from utinteractiveconsole.persistence.pandas_converters import store_data, guess_type
        skip_fields = skip_fields or []

        store.put('%s/schema' % base_path, self.processor.schema.as_dataframe())

        # Store attributes
        for key, value in self.attributes.items():
            try:
                datatype = guess_type(value)
            except TypeError, e:
                log.exception(e)
                continue
            store_data(store, '%s/attributes/%s' % (base_path, key), value, datatype=datatype)

        fieldnames = [f.name for f in self.processor.schema.fields if f.name not in skip_fields]
        data = dict([(k, []) for k in fieldnames])
        timestamps = []

        stream_filters = stream_filters or []

        records = self.cached_records if len(self.cached_records) > 0 else list(self)
        for sfilter in stream_filters:
            records = sfilter.process(records)

        for record in records:
            timestamps.append(record.timestamp)
            for fn in fieldnames:
                data[fn].append(getattr(record, fn))

        store_data(store, '%s/timestamps' % (base_path, ), pd.Series(timestamps))
        for field in self.processor.schema.fields:
            if field.name in skip_fields:
                continue
            try:
                element_count = 1
                if field.is_array and len(data[field.name]) > 0:
                    element_count = len(data[field.name][0])

                store_data(store, '%s/fields/%s' % (base_path, field.name), data[field.name],
                           datatype=field.datatype, is_array=True, element_count=element_count)
            except Exception, e:
                log.error("Error storing field %s from dataset %s" % (field.name, self.name))
                log.exception(e)
