__author__ = 'jack'
import abc
from collections import namedtuple
from pyqtgraph.Qt import QtCore, QtGui

PortInfo = namedtuple("PortInfo", ["name", "port_type", "mode", "data_type", "queued"])

PORT_TYPE_SOURCE = 0
PORT_TYPE_SINK = 1

PORT_MODE_PUSH = 0
PORT_MODE_PULL = 1


class Port(QtCore.QObject):

    def __init__(self, parent, name):
        super(Port, self).__init__()
        self.parent = parent
        self.name = name

    def get_info(self):
        raise NotImplemented


class InPort(Port):

    def __init__(self, parent, name, cb=None):
        super(Port, self).__init__(parent, name)
        self.value = None
        self.cb = cb

    def fullname(self):
        return "%s:%s" % (self.parent if self.parent is not None else "", self.name)

    def handle_receive(self, value):
        self.value = value
        if self.cb is not None:
            self.cb(self, value)


class OutPort(Port):

    def __init__(self, parent, name):
        super(Port, self).__init__(parent, name)

    def send(self, value):
         self.emit(QtCore.SIGNAL('dataReady(PyQt_PyObject)'), value)

    def subscribe(self, inport):
        info = inport.get_info()
        if info.queued:
            return super(OutPort, self).connect(self, QtCore.SIGNAL('dataReady(PyQt_PyObject)'), inport.handle_receive, QtCore.Qt.QueuedConnection)
        else:
            return super(OutPort, self).connect(self, QtCore.SIGNAL('dataReady(PyQt_PyObject)'), inport.handle_receive)


    def unsubscribe(self, inport):
        raise NotImplemented




class ExtensionBase(object):
    """Base class for extensions.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, context):
        self.context = context
        self.widget = None

    @abc.abstractmethod
    def register(self, mgr):
        """Register extension at manager.

        :param mgr: The manager of this extension.
        :type mgr: obj
        :returns: Main Widget.
        """

    @abc.abstractmethod
    def get_name(self):
        """Return human readable name.

        :returns: name of extension.
        """

    @abc.abstractmethod
    def get_ports(self):
        """Return a list of defined ports and infos about them.

        :returns: dict of defined ports (portname: info entries).
        """

    @abc.abstractmethod
    def get_port(self, name):
        """Return a port by name.

        :param portname: name of the local port that should be connected.
        :type portname: str:
        :returns: Port object.
        """


    def connectTo(self, portname, outport):
        """Connect an inport with portname to outport.

        :param portname: name of the local port that should be connected.
        :type portname: str:
        :param outport: port to connect to.
        :type outport: Port:
        :returns: bool success.
        """

        inport = self.get_port(portname)
        outport.subscribe(inport)
