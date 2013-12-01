__author__ = 'jack'
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

from ubitrack.core import math, calibration
from ubitrack.vision import vision
from ubitrack.visualization import visualization


class NoValueException(ValueError):
    pass


class PushSinkAdapter(QtCore.QObject):

    def __init__(self, sink):
        super(PushSinkAdapter, self).__init__()

        self.sink = sink
        self._time = None
        self._value = None
        sink.setCallback(self.cb_handler)


    def cb_handler(self, m):
        self._time = m.time()
        self._value = m
        self.emit(QtCore.SIGNAL('dataReady(long)'), self._time)


    def connect(self, handler):
        return super(PushSinkAdapter, self).connect(self, QtCore.SIGNAL('dataReady(long)'), handler, QtCore.Qt.QueuedConnection)


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

