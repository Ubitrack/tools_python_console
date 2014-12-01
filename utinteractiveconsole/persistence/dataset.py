__author__ = 'jack'

from atom.api import Atom, Value, Dict, List, Str, Int, Typed

import logging
import new

from .recordsource import RecordSource

log = logging.getLogger(__name__)


class DataSet(Atom):

    name = Str()
    title = Str()

    datasource = Typed(RecordSource)
    processor_factory = Value()
    attributes = Dict()

    processor = Value()
    reference_timestamps = List()
    interval = Int()

    def _default_reference_timestamps(self):
        return self.datasource.reference_timestamps

    def _default_interval(self):
        return self.datasource.interval

    def _default_processor(self):
        return self.get_streamprocessor(self.datasource.emit())

    # not yet implemented
    # stream_filters = List()

    def make_connector_class(self):
        attrs = {}
        names = self.processor.output_field_names

        for k in names:
            attrs[k] = Value()

        attrs["names"] = List(default=names)

        def update_data(s, ds):
            for k,v in ds.__dict__.items():
                if hasattr(s, k):
                    setattr(s, k, v)

        attrs['__call__'] = update_data
        return new.classobj("DataConnector", (Atom,), attrs)

    def get_streamprocessor(self, data):
        p_cls = self.processor_factory
        p_kw = dict(raw_data=data)
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

    def emit(self):
        if self.processor is not None:
            return self.processor.emit()
        return None
