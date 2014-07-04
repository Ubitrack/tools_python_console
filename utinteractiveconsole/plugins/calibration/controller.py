__author__ = 'jack'

import os, sys
import glob
import abc
import shutil
from atom.api import Atom, Value, Str, Typed, Dict, Float

import logging

from utinteractiveconsole.uthelpers import UbitrackFacadeBase, UbitrackConnectorBase, ubitrack_connector_class

log = logging.getLogger(__name__)


class PreviewControllerFactory(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, parent, context):
        self.parent = parent
        self.context = context

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

    content = Value()

    screen_ratio = Float(640./480.)

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

    module_name = Str()
    config_ns = Str()

    context = Value()
    widget = Value()
    facade = Typed(UbitrackFacadeBase)
    state = Value()
    wizard_state = Value()

    config = Dict()

    calib_dir = Str()
    srg_dir = Str()
    results_dir = Str()
    dfg_filename = Str()

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

    def _default_srg_dir(self):
        return os.path.expanduser(self.wizard_state.config.get("srgdir"))

    def _default_results_dir(self):
        return os.path.expanduser(self.wizard_state.config.get("resultsdir"))

    def _default_dfg_filename(self):
        if "dfg_filename" in self.config:
            return self.config["dfg_filename"]
        return ""

    @property
    def preview_controller(self):
        if self.wizard_state.controller.preview_controller is not None:
            return self.wizard_state.controller.preview_controller

    @property
    def preview_widget(self):
        if self.wizard_state.controller.preview is not None:
            return self.wizard_state.controller.preview

    def setupController(self, active_widgets=None):
        log.info("Setup %s controller" % self.module_name)
        if self.facade is not None and self.dfg_filename:
            fname = os.path.join(self.facade.dfg_basedir, self.dfg_filename)
            if os.path.isfile(fname):
                self.facade.dfg_filename = fname
            else:
                log.error("Module %s: Invalid dfg_filename specified for facade: %s" % (self.module_name, fname))


    def teardownController(self, active_widgets=None):
        pass

    def setupPreview(self, active_widgets=None):
        if self.preview_controller is not None:
            self.preview_controller.moduleSetupPreview(self)

    def teardownPreview(self, active_widgets=None):
        if self.preview_controller is not None:
            self.preview_controller.moduleTeardownPreview(self)

    def startCalibration(self):
        self.facade.loadDataflow(self.dfg_filename)
        self.facade.startDataflow()

    def stopCalibration(self):
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
            rec_path = os.path.join(root_dir, "record")
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

    def _default_connector(self):
        if self.dfg_filename and self.sync_source:
            fname = os.path.join(self.facade.dfg_basedir, self.dfg_filename)
            if os.path.isfile(fname):
                utconnector = ubitrack_connector_class(fname)(sync_source=self.sync_source)
                return utconnector
            # else:
            #     log.error("Module %s: Invalid dfg_filename specified for utconnector of module: %s" % (self.module_name, fname))
        return None


class MasterSlaveCalibrationController(CalibrationController):
    connector = Typed(UbitrackConnectorBase)
    sync_source = Str()

    def _default_connector(self):
        if self.facade.master is not None:
            facade = self.facade.master
            if facade.dfg_filename and self.sync_source:
                fname = os.path.join(facade.dfg_basedir, facade.dfg_filename)
                if os.path.isfile(fname):
                    utconnector = ubitrack_connector_class(fname)(sync_source=self.sync_source)
                    return utconnector
                # else:
                #     log.error("Module %s: Invalid dfg_filename specified for utconnector of module: %s" % (self.module_name, fname))
        return None

