__author__ = 'jack'

from atom.api import Atom, Value, Dict, List, Str, Int, Typed

import logging
import new

from .recordsource import RecordSource

log = logging.getLogger(__name__)


class DataSet(Atom):

    name = Str()
    title = Str()

    recordsource = Typed(RecordSource)
    processor_factory = Value()
    stream_filters = List()
    attributes = Dict()

    processor = Value()
    reference_timestamps = List()
    interval = Int()

    connector_class = Value()


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
            for k,v in ds.__dict__.items():
                if hasattr(s, k):
                    setattr(s, k, v)

        attrs['__call__'] = update_data
        return new.classobj("DataConnector_%s_%s" % (self.name, self.processor.name), (Atom,), attrs)

    def __iter__(self):
        
        if self.processor is None:
            raise StopIteration()

        # apply stream filters
        producer = self.processor
        for sfilter in self.stream_filters:
            producer = sfilter.process(producer)

        for record in producer:
            yield record
