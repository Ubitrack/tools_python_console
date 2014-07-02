__author__ = 'jack'

import os, sys
import glob
import shutil
from atom.api import Atom, Value, Str, Typed, Dict

import logging

from utinteractiveconsole.uthelpers import UbitrackFacadeBase, UbitrackConnectorBase, ubitrack_connector_class

log = logging.getLogger(__name__)



class CalibrationController(Atom):
    # class variable
    save_results = True

    module_name = Str()
    config_ns = Str()

    context = Value()
    widget = Value()
    facade = Typed(UbitrackFacadeBase)
    state = Value()
    wizard_state = Value()

    config = Dict()

    data_dir = Str()
    dfg_filename = Str()

    def _default_config(self):
        cfg = self.context.get("config")
        sname = "%s.modules.%s" % (self.config_ns, self.module_name)
        if cfg.has_section(sname):
            return dict(cfg.items(sname))
        else:
            log.error("Missing section: [%s] in config" % sname)
            return dict()

    def _default_data_dir(self):
        return os.path.expanduser(self.wizard_state.config.get("datadir"))

    def _default_dfg_filename(self):
        if "dfg_filename" in self.config:
            fname = os.path.expanduser(self.config["dfg_filename"])
            if os.path.isfile(fname):
                return fname
            if "srgdir" in self.wizard_state.config:
                fname = os.path.join( os.path.expanduser(self.wizard_state.config["srgdir"]), self.config["dfg_filename"])
                if os.path.isfile(fname):
                    return fname
        return ""

    def setupController(self, active_widgets=None):
        pass

    def teardownController(self, active_widgets=None):
        pass

    def startCalibration(self):
        self.facade.loadDataflow(self.dfg_filename)
        self.facade.startDataflow()

    def stopCalibration(self):
        self.facade.stopDataflow()
        self.facade.clearDataflow()

    def saveResults(self, root_dir, extra_files=None):
        calib_files = self.getCalibrationFiles()
        if calib_files:
            # XXX use config->calibdir here !!!
            calib_path = os.path.join(root_dir, "calibration")
            if not os.path.isdir(calib_path):
                os.makedirs(calib_path)
            for calib_file in calib_files:
                fname = os.path.join(calib_path, os.path.basename(calib_file))
                if os.path.isfile(calib_file):
                    shutil.copy(calib_file, fname)
                else:
                    log.warn("Calibration file not found: %s" % fname)

        rec_files = self.getRecordedFiles()
        if rec_files:
            rec_path = os.path.join(root_dir, "data")
            if not os.path.isdir(rec_path):
                os.makedirs(rec_path)
            for rec_file in rec_files:
                if os.path.isfile(rec_file):
                    fname = os.path.join(rec_path, os.path.basename(rec_file))
                    shutil.copy(rec_file, fname)
                else:
                    log.warn("Recorded file not found: %s" % fname)

        if extra_files is not None:
            for extra_file in extra_files:
                if os.path.isfile(extra_file):
                    fname = os.path.join(root_dir, os.path.basename(extra_file))
                    shutil.copy(extra_file, fname)
                else:
                    log.warn("Additional file not found: %s" % fname)


    def getCalibrationFiles(self):
        if "calib_files" in self.config:
            return [os.path.join(self.data_dir, f.strip()) for f in self.config["calib_files"].split(",")]
        return []

    def getRecordedFiles(self):
        if "recorddir" in self.config:
            return glob.glob(os.path.join(self.data_dir, self.config["recorddir"], "*"))
        return []


class LiveCalibrationController(CalibrationController):
    connector = Typed(UbitrackConnectorBase)
    sync_source = Str()

    def _default_connector(self):
        if self.dfg_filename and self.sync_source:
            utconnector = ubitrack_connector_class(self.dfg_filename)(sync_source=self.sync_source)
            return utconnector
        return None

