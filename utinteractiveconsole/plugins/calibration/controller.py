__author__ = 'jack'

import os, sys
import glob
import abc
import shutil
from atom.api import Atom, Value, Str, Typed, Dict, List, Float, Bool

import logging

from utinteractiveconsole.uthelpers import UbitrackFacadeBase, UbitrackConnectorBase, UbitrackFacade, ubitrack_connector_class

log = logging.getLogger(__name__)


class PreviewControllerFactory(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, parent, context, config_ns):
        self.parent = parent
        self.context = context
        self.config_ns = config_ns

    @abc.abstractmethod
    def create(self, workspace, name, widget_parent):
        """
        create a controller instance
        """



class PreviewControllerBase(Atom):
    parent = Value()
    widget_parent = Value()
    widget_name = Str()
    workspace = Value()
    context = Value()
    config_ns = Str()

    config = Dict()
    wizard_config = Dict()

    facade = Typed(UbitrackFacadeBase)
    dfg_dir = Str()
    dfg_filename = Str()

    connector = Typed(UbitrackConnectorBase)
    sync_source = Str()


    content = Value()
    screen_ratio = Float(640./480.)

    def _default_config(self):
        cfg = self.context.get("config")
        sname = "%s.preview" % self.config_ns
        if cfg.has_section(sname):
            return dict(cfg.items(sname))
        else:
            log.error("Missing section: [%s] in config" % sname)
            return dict()

    def _default_wizard_config(self):
        cfg = self.context.get("config")
        if cfg.has_section(self.config_ns):
            return dict(cfg.items(self.config_ns))
        else:
            log.error("Missing section: [%s] in config" % self.config_ns)
            return dict()

    def _default_dfg_dir(self):
        return os.path.expanduser(self.wizard_config.get("dfgdir"))

    def _default_dfg_filename(self):
        return os.path.expanduser(self.config.get("dfg_filename"))

    def _default_facade(self):
        facade = UbitrackFacade(context=self.context,)
        fname = os.path.join(self.dfg_dir, self.dfg_filename)
        if os.path.isfile(fname):
            facade.dfg_filename = fname
        else:
            log.warn("Invalid dfg_filename for preview controller: %s" % fname)
        return facade

    def _default_connector(self):
        if not self.sync_source:
            self.sync_source = self.config.get("sync_source")

        if self.sync_source:
            fname = os.path.join(self.dfg_dir, self.facade.dfg_filename)
            if os.path.isfile(fname):
                log.info("Setup Preview Connector with sync_source: %s" % self.sync_source)
                utconnector = ubitrack_connector_class(fname)(sync_source=self.sync_source)
                return utconnector
            else:
                log.error("Invalid dfg_filename specified for preview utconnector of module: %s" % fname)
        else:
            log.error("Missing sync_source for live preview")
        return None



    def initialize(self):
        log.info("Initialize Preview Controller")
        widget_state = self.parent.current_state
        widget_state.observe("on_module_after_load", self.on_module_after_load)
        widget_state.observe("on_module_after_start", self.on_module_after_start)
        widget_state.observe("on_module_before_stop", self.on_module_before_stop)
        widget_state.observe("on_module_before_unload", self.on_module_before_unload)

    def teardown(self):
        log.info("Teardown Preview Controller")
        widget_state = self.parent.current_state
        widget_state.unobserve("on_module_after_load", self.on_module_after_load)
        widget_state.unobserve("on_module_after_start", self.on_module_after_start)
        widget_state.unobserve("on_module_before_stop", self.on_module_before_stop)
        widget_state.unobserve("on_module_before_unload", self.on_module_before_unload)



    def on_module_after_load(self, change):
        pass

    def on_module_after_start(self, change):
        pass

    def on_module_before_stop(self, change):
        pass

    def on_module_before_unload(self, change):
        pass


    def setupPreview(self):
        pass

    def teardownPreview(self):
        pass

    def moduleSetupPreview(self, controller):
        pass

    def moduleTeardownPreview(self, controller):
        pass




class CalibrationController(Atom):
    # class variable
    save_results = True
    show_facade_controls = True

    result_ok = Bool(False)

    module_name = Str()
    config_ns = Str()

    context = Value()
    module = Value()
    widget = Value()
    facade = Typed(UbitrackFacadeBase)
    state = Value()
    wizard_state = Value()

    config = Dict()

    calib_dir = Str()
    dfg_dir = Str()
    results_dir = Str()
    dfg_filename = Str()

    autocomplete_enable = Bool(False)
    autocomplete_maxerror_str = Str()

    def _default_config(self):
        cfg = self.context.get("config")
        sname = "%s.modules.%s" % (self.config_ns, self.module_name)
        if cfg.has_section(sname):
            return dict(cfg.items(sname))
        else:
            log.error("Missing section: [%s] in config" % sname)
            return dict()

    def _default_calib_dir(self):
        return os.path.expanduser(self.wizard_state.config.get("calibdir"))

    def _default_dfg_dir(self):
        return os.path.expanduser(self.wizard_state.config.get("dfgdir"))

    def _default_results_dir(self):
        return os.path.expanduser(self.wizard_state.config.get("resultsdir"))

    def _default_dfg_filename(self):
        if "dfg_filename" in self.config:
            return self.config["dfg_filename"]
        return ""

    def _default_autocomplete_enable(self):
        return self.config.get("autocomplete_enable", "False").strip().lower() == 'true'

    def _default_autocomplete_maxerror_str(self):
        return self.config.get("autocomplete_maxerror", "").strip()


    def setupController(self, active_widgets=None):
        log.info("Setup %s controller" % self.module_name)
        if self.facade is not None and self.dfg_filename:
            fname = os.path.join(self.dfg_dir, self.dfg_filename)
            if os.path.isfile(fname):
                self.facade.dfg_filename = fname
            else:
                log.error("Module %s: Invalid dfg_filename specified for facade: %s" % (self.module_name, fname))


    def teardownController(self, active_widgets=None):
        if self.facade.is_running:
            self.stopCalibration()

    def startCalibration(self):
        self.facade.loadDataflow(self.dfg_filename)
        self.facade.startDataflow()
        self.wizard_state.on_module_after_start(self.module_name)

    def stopCalibration(self):
        self.wizard_state.on_module_before_stop(self.module_name)
        self.facade.stopDataflow()
        self.facade.clearDataflow()

    def saveResults(self, root_dir, extra_files=None):
        calib_files = self.getCalibrationFiles()
        if calib_files:
            calib_path = root_dir
            if not os.path.isdir(calib_path):
                os.makedirs(calib_path)
            for calib_file in calib_files:
                fname = os.path.join(calib_path, os.path.basename(calib_file))
                if os.path.isfile(calib_file):
                    shutil.copy(calib_file, fname)
                else:
                    log.warn("Calibration file not found: %s" % calib_file)

        rec_files = self.getRecordedFiles()
        if rec_files:
            rec_path = os.path.join(root_dir, "record", self.module_name)
            if not os.path.isdir(rec_path):
                os.makedirs(rec_path)
            for rec_file in rec_files:
                if os.path.isfile(rec_file):
                    fname = os.path.join(rec_path, os.path.basename(rec_file))
                    shutil.copy(rec_file, fname)
                else:
                    log.warn("Recorded file not found: %s" % rec_file)

        if extra_files is not None:
            for extra_file in extra_files:
                if os.path.isfile(extra_file):
                    fname = os.path.join(root_dir, os.path.basename(extra_file))
                    shutil.copy(extra_file, fname)
                else:
                    log.warn("Additional file not found: %s" % extra_file)


    def getCalibrationFiles(self):
        if "calib_files" in self.config:
            return [os.path.join(self.calib_dir, f.strip()) for f in self.config["calib_files"].split(",")]
        return []

    def getRecordedFiles(self):
        if "recorddir" in self.config:
            return glob.glob(os.path.join(os.path.expanduser(self.config["recorddir"]), "*"))
        return []


class LiveCalibrationController(CalibrationController):
    connector = Typed(UbitrackConnectorBase)
    sync_source = Str()

    required_sinks = List()

    def _default_connector(self):
        if self.dfg_filename and self.sync_source:
            fname = os.path.join(self.dfg_dir, self.dfg_filename)
            if os.path.isfile(fname):
                log.info("Setup LiveCalibration Connector with sync_source: %s" % self.sync_source)
                utconnector = ubitrack_connector_class(fname)(sync_source=self.sync_source)
                return utconnector
            # else:
            #     log.error("Module %s: Invalid dfg_filename specified for utconnector of module: %s" % (self.module_name, fname))
        return None

    def verify_connector(self):
        is_valid = True
        if self.connector is not None:
            members = [k for k in self.connector.members().keys() if not k.startswith('_')]
            for sink in self.required_sinks:
                if not sink in members:
                    log.error("Missing sink in DFG with name: %s" % sink)
                    is_valid = False
        else:
            is_valid = False
        return is_valid
