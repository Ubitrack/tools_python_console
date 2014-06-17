__author__ = 'jack'
import abc
from collections import namedtuple
from enaml.qt import QtCore

from atom.api import Subclass, Unicode, Value, Typed

from enaml.widgets.api import Container
from enaml.workbench.api import PluginManifest
from enaml.workbench.ui.api import Workspace

from utinteractiveconsole.app import AppState, ExtensionManager



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

    def update_optparse(self, parser):
        """Register optparse arguments.

        :param parser: add cmdline options.
        :type parser: obj
        :returns: None.
        """


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




class ExtensionWorkspace(Workspace):
    """ A custom Workspace class for extensions.

    This workspace class will instantiate the content and register an
    additional plugin with the workbench when it is started. The extra
    plugin can be used to add addtional functionality to the workbench
    window while this workspace is active. The plugin is unregistered
    when the workspace is stopped.

    """
    #: The enamldef'd Container to create when the workbench is started.
    content_def = Subclass(Container)

    #: The enamldef'd PluginManifest to register on start.
    manifest_def = Subclass(PluginManifest)

    #: Storage for the plugin manifest's id.
    _manifest_id = Unicode()

    # global state for the utic plugins
    appstate = Typed(AppState)
    utic_plugin = Typed(ExtensionBase)

    def start(self):
        """ Start the workspace instance.

        This method will create the container content and register the
        provided plugin with the workbench.

        """
        self.content = self.content_def(appstate=self.appstate, utic_plugin=self.utic_plugin)
        manifest = self.manifest_def(appstate=self.appstate, utic_plugin=self.utic_plugin)
        self._manifest_id = manifest.id
        self.workbench.register(manifest)

    def stop(self):
        """ Stop the workspace instance.

        This method will unregister the workspace's plugin that was
        registered on start.

        """
        # XXX check for running processes / unsaved changes here ?
        self.workbench.unregister(self._manifest_id)

