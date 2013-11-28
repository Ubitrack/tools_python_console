__author__ = 'jack'
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

from ubitrack.core import math, calibration
from ubitrack.vision import vision
from ubitrack.visualization import visualization




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


    @property
    def value(self):
        return self._value


    @property
    def time(self):
        return self._time

