__author__ = 'jack'
import abc
from atom.api import Atom, String, Dict
from atom.catom import Member

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
        sname = "%s.%s" % (self.parent.config_ns, self.module_name)
        if cfg.has_section(sname):
            return cfg.getboolean(sname, "enabled")
        return False


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
    context = Member()

    modules_ns = String('calibration_wizard.modules')
    config_ns = String('calibration_wizard')
    modules = Dict()

    # xxx why Member instead of Value ??
    extension_manager = Member()
    graph = Member()

    def _default_extension_manager(self):
        return extension.ExtensionManager(
            namespace=self.modules_ns,
            invoke_on_load=True,
            invoke_args=(self, self.context, ),
        )

    def is_module_enabled(self, name):
        cfg = self.context.get("config")
        sname = "%s.%s" % (self.config_ns, name)
        if cfg.has_section(sname):
            return cfg.getboolean(sname, "enabled")
        return False

    def _default_modules(self):
        modules = dict()
        for ext in self.extension_manager:
            # if not self.is_module_enabled(ext.name):
            #     continue
            ext.obj.set_module_name(ext.name)
            modules[ext.name] = ext.obj
        return modules

    def _default_graph(self):
        g = nx.DiGraph()
        for key, module in self.modules.items():
            g.add_node(key, obj=module, category=module.get_category())

        for key,module in self.modules.items():
            for dep in module.get_dependencies():
                g.add_edge(dep, key)

        return g