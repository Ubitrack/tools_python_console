__author__ = 'jack'

from atom.api import Atom, List, Str, Int

import logging

log = logging.getLogger(__name__)


class DataSource(Atom):

    name = Str()
    title = Str()

    data = List()
    record_count = Int()
    reference_timestamps = List()
    interval = Int()

    output_field_names = List()

    def _default_output_field_names(self):
        if self.data is not None and len(self.data) > 0:
            return list(self.data[0]._fields)
        return []

    def emit(self):
        return self.data

