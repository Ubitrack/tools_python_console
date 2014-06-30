__author__ = 'jack'
import abc
from atom.api import Atom, Str, Dict, Typed, Value

from stevedore import extension
import networkx as nx
import logging

log = logging.getLogger(__name__)




class ModuleBase(object):
    """Base class for modules.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parent, context):
        self.parent = parent
        self.context = context
        self.module_name = None

    def set_module_name(self, mname):
        self.module_name = mname

    def get_module_name(self):
        return self.module_name

    def is_enabled(self):
        cfg = self.context.get("config")
        if self.module_name is None:
            log.error("Module name must be set before usage")
        sname = "%s.modules.%s" % (self.parent.config_ns, self.module_name)
        enabled = False
        if cfg.has_section(sname):
            enabled = cfg.getboolean(sname, "enabled")
        log.info("CalibrationWizard module: %s is %s" % (self.module_name, "enabled" if enabled else "disabled"))
        return enabled


    @abc.abstractmethod
    def get_category(self):
        """Return human readable name.

        :returns: name of category.
        """

    @abc.abstractmethod
    def get_name(self):
        """Return human readable name.

        :returns: name of module.
        """

    @abc.abstractmethod
    def get_dependencies(self):
        """Return a list of dependencies, that need to be satisified.

        :returns: list of module names, that are required to be completed before invoking this module.
        """

    @abc.abstractmethod
    def get_widget_class(self):
        """Return the widget class for this module.

        :returns: widget class.
        """

    @abc.abstractmethod
    def get_controller_class(self):
        """Return a controller class for this module.

        :returns: controller class.
        """

    # @abc.abstractmethod
    # def result_filenames(self):
    #     """Return a list of filenames that represent the result.
    #
    #     :returns: list of filenames.
    #     """


class ModuleManager(Atom):
    context = Value()

    modules_ns = Str('calibration_wizard.modules')
    config_ns = Str('calibration_wizard.config')
    modules = Dict()

    extension_manager = Typed(extension.ExtensionManager)
    graph = Typed(nx.DiGraph)

    def _default_extension_manager(self):
        log.info("Initializing calibration wizard with configuration: %s and modules: %s" % (self.config_ns, self.modules_ns))
        return extension.ExtensionManager(
            namespace=self.modules_ns,
            invoke_on_load=True,
            invoke_args=(self, self.context, ),
        )

    # def is_module_enabled(self, name):
    #     cfg = self.context.get("config")
    #     sname = "%s.modules.%s" % (self.config_ns, name)
    #     if cfg.has_section(sname):
    #         return cfg.getboolean(sname, "enabled")
    #     return False

    def _default_modules(self):
        modules = dict()
        for ext in self.extension_manager:
            mod = ext.obj
            mod.set_module_name(ext.name)
            # if not mod.is_enabled():
            #     continue
            modules[ext.name] = mod
        return modules

    def _default_graph(self):
        g = nx.DiGraph()
        for key, module in self.modules.items():
            g.add_node(key, obj=module, category=module.get_category())

        for key, module in self.modules.items():
            for dep in module.get_dependencies():
                g.add_edge(dep, key)

        return g