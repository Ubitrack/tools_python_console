__author__ = 'jack'
import os
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

    def __init__(self, parent, context, module_name=None):
        self.parent = parent
        self.context = context
        self.module_name = module_name

    def set_module_name(self, mname):
        self.module_name = mname

    def get_module_name(self):
        return self.module_name

    def make_instances(self):
        # starting at version 2 multiple instances of a module can be created
        # this is backwards incompatible, so you need to update all wizard entries
        if self.parent.config_version >= 2:
            log.info('Loading Module with config V2')
            cfg = self.context.get("config")
            sections = cfg.sections()

            sname = self.config_ns
            instances = []
            for section in sections:
                if section.startswith(sname):
                    module_name = section.replace(self.config_module_prefix, "")
                    log.info("Create instance for module %s with id %s" % (self.module_name, module_name))
                    instances.append(self.__class__(self.parent, self.context, module_name=module_name))
            return instances
        else:
            log.info('Loading Module with config V1')
            return [self, ]


    @property
    def config_module_prefix(self):
        return "%s.modules." % (self.parent.config_ns,)

    @property
    def config_ns(self):
        if self.module_name is None:
            log.error("Module name must be set before usage")
            raise ValueError("Module name must be set before usage")
        sname = "%s%s" % (self.config_module_prefix, self.module_name)
        return sname

    def is_enabled(self):
        cfg = self.context.get("config")
        sname = self.config_ns
        enabled = False
        if cfg.has_section(sname):
            enabled = cfg.getboolean(sname, "enabled")
        # log.info("CalibrationWizard module: %s is %s" % (self.module_name, "enabled" if enabled else "disabled"))
        return enabled

    def get_name(self):
        """Return human readable name.

        :returns: name of module.
        """
        cfg = self.context.get("config")
        sname = self.config_ns
        if cfg.has_section(sname):
            return cfg.get(sname, "name")

    def get_dependencies(self):
        """Return a list of dependencies, that need to be satisified.

        :returns: list of module names, that are required to be completed before invoking this module.
        """
        cfg = self.context.get("config")
        sname = self.config_ns
        if cfg.has_section(sname) and cfg.has_option(sname, 'dependencies'):
            deps = cfg.get(sname, "dependencies")
            return [d.strip() for d in deps.split(',')]
        return []

    def get_calib_files(self):
        """Return a list of calib files generated by this module

        :returns: list of filenames names
        """
        cfg = self.context.get("config")
        calib_dir = None
        if cfg.has_section(self.parent.config_ns) and cfg.has_option(self.parent.config_ns, "calibdir"):
            calib_dir = os.path.expanduser(cfg.get(self.parent.config_ns, "calibdir"))
        if calib_dir is not None:
            sname = self.config_ns
            if cfg.has_section(sname) and cfg.has_option(sname, 'calib_files'):
                cfs = cfg.get(sname, "calib_files")
                return [os.path.join(calib_dir, d.strip()) for d in cfs.split(',')]
        return []

    @abc.abstractmethod
    def get_category(self):
        """Return human readable name.

        :returns: name of category.
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

    modules_ns = Str('vharcalibration.plugin.modules')
    config_ns = Str('calibration_wizard.config')
    modules = Dict()

    extension_manager = Typed(extension.ExtensionManager)
    graph = Typed(nx.DiGraph)

    @property
    def config_version(self):
        cfg = self.context.get('config')
        if cfg.has_option("calibration_wizard", "config_version"):
            return cfg.getint("calibration_wizard", "config_version")
        return 1

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

            # XXX hack to enable multiple instances of a module via configuration
            for m in mod.make_instances():
                modules[m.get_module_name()] = m

        return modules

    def _default_graph(self):
        g = nx.DiGraph()
        for key, module in self.modules.items():
            g.add_node(key, obj=module, category=module.get_category())

        for key, module in self.modules.items():
            for dep in module.get_dependencies():
                g.add_edge(dep, key)

        return g