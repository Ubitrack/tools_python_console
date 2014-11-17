__author__ = 'jack'
import abc
from enaml.qt import QtCore
import stevedore
import logging
import re

from atom.api import Atom, Value, Typed, List, Dict, Str

log = logging.getLogger(__name__)

MODULE = re.compile(r"\w+(\.\w+)*$").match


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



class ConfigurableExtensionManager(stevedore.extension.ExtensionManager):
    """ Subclass allows preloading the entrypoint cache.

    This class should be used in all utic systems, so that we can add loading from non-setuptools entrypoints later.
    """

    def __init__(self, namespace,
                 invoke_on_load=False,
                 invoke_args=(),
                 invoke_kwds={},
                 propagate_map_exceptions=False,
                 preload_entrypoint_cache=None):

        if preload_entrypoint_cache is not None:
            self.ENTRY_POINT_CACHE.update(preload_entrypoint_cache)
        super(ConfigurableExtensionManager, self).__init__(namespace, invoke_on_load=invoke_on_load,
                                                           invoke_args=invoke_args, invoke_kwds=invoke_kwds,
                                                           propagate_map_exceptions=propagate_map_exceptions)


class CustomEntryPoint(Atom):
    name = Str()
    module_name = Value()
    attrs = List()

    def load(self):
        entry = __import__(self.module_name, globals(), globals(), ['__name__'])
        for attr in self.attrs:
            try:
                entry = getattr(entry, attr)
            except AttributeError:
                raise ImportError("%r has no %r attribute" % (entry, attr))
        return entry

    @classmethod
    def parse(cls, key, value):
        """Parse a single entry point from string `src`

            key: some.module:some.attr

        The entry key and module name are required, but the ``:attrs`` part is optional
        """
        try:
            attrs = []
            if ':' in value:
                value, attrs = value.split(':', 1)
                if not MODULE(attrs.rstrip()):
                    raise ValueError
                attrs = attrs.rstrip().split('.')
        except ValueError:
            msg = "CustomEntryPoint must be in 'name=module:attrs' format"
            raise ValueError(msg, "%s=%s" % (key, value))
        else:
            return cls(name=key.strip(), module_name=value.strip(), attrs=attrs)

    @classmethod
    def instances_from_items(cls, items):
        result = {}
        for key, value in items:
            result[key] = cls.parse(key, value).load()
        return result


class WorkspaceExtensionManager(Atom):
    appstate = Value()

    extensions = Dict()
    extension_manager = Typed(ConfigurableExtensionManager)
    extensions_actionitems = List()
    extensions_workspaceplugins = List()

    def _default_extension_manager(self):
        return ConfigurableExtensionManager(
                        namespace='utinteractiveconsole.extension',
                        invoke_on_load=True,
                        invoke_args=(self.appstate.context,),
                    )


    def updateCmdlineParser(self, parser):
        self.extension_manager.map(lambda e: e.obj.update_optparse(parser))


    def initExtensions(self):
        log.info("Init Extensions")
        # load all extensions

        def register(ext, mgr):
            log.info("Register plugin: %s" % ext.name)
            return (ext.name, ext.obj.register(mgr))

        results = self.extension_manager.map(register, self)

        for name, result in results:
            if result.widget is not None:
                result.widget.connect(result.widget, QtCore.SIGNAL('extensionChanged()'), self.updateExtensionInfo)

    def updateExtensionInfo(self):

        for cat_name, cat in self.extensions.items():
            for name, ext in cat.items():
                print cat_name, name, ext.get_name()
                print ext.get_ports()


    def registerExtension(self, name, inst, category="default", action_items=None, workspace_plugins=None, add_globals=None):
        cat = self.extensions.setdefault(category, {})
        if name in cat:
            raise ValueError("Extension with name: %s already registered in category %s." % (name, category))

        cat[name] = inst
        if action_items is not None:
            self.extensions_actionitems.append((name, category, action_items))
        if workspace_plugins is not None:
            self.extensions_workspaceplugins.append((name, category, workspace_plugins))
        if add_globals is not None:
            for k,v in add_globals.items():
                self.appstate.context[k] = v


    def generateWorkspaceActionItems(self, plugin_ext):
        result = []
        for name, category, action_items in self.extensions_actionitems:
            for item in action_items:
                try:
                    item.set_parent(plugin_ext)
                    result.append(item)
                except Exception, e:
                    log.error("error while adding menu action: %s" % item.path)
                    log.exception(e)
        return result

    def generateWorkspaceExtensions(self, plugin_manifest):
        result = []
        for name, category, plugins in self.extensions_workspaceplugins:
            for plugin in plugins:
                try:
                    log.info("Workspace Plugin: %s" % plugin.id)
                    plugin.set_parent(plugin_manifest)
                    result.append(plugin)
                except Exception, e:
                    log.error("error while adding plugin extension: %s" % plugin.id)
                    log.exception(e)

        # autostart = self.get_autostart_config()
        # if autostart is not None:
        #     def autostart_factory(workbench):
        #         log.info("Autostart workspace: %r" % autostart)
        #         return Autostart(plugin_id="utic.%s" % autostart)
        #
        #     ext = Extension(parent=plugin_manifest,
        #                     id='autostart',
        #                     point="enaml.workbench.ui.autostart",
        #                     factory=autostart_factory)
        #     result.append(ext)
        return result

    def get_autostart_config(self):
        ctx = self.appstate.context
        # preference to command line
        opt = ctx.get('options')
        if opt.autostart_workspace is not None:
            return opt.autostart_workspace
        # use config file
        cfg = ctx.get('config')
        if cfg and cfg.has_option('extensions', 'autostart'):
            return cfg.get('extensions', 'autostart')
        # no autostart
        return None
