__author__ = 'jack'

import os
import time
import logging
from math import degrees, acos
import numpy as np
from numpy.linalg import norm

log = logging.getLogger(__name__)

from atom.api import Atom, Bool, Str, Value, Typed, List, Dict, Float, Int, Enum
from enaml.qt import QtCore
from enaml.application import deferred_call
from enaml.widgets.api import FileDialogEx

import enaml

with enaml.imports():
    from .views.offline_calibration2 import OfflineCalibrationPanel, OfflineCalibrationResultPanel

from ubitrack.core import math

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

from utinteractiveconsole.plugins.calibration.algorithms.offline_calibration2 import (
    OfflineCalibrationProcessor, OfflineCalibrationParameters, angle_null_correction
)


class BackgroundCalculationThread(QtCore.QThread):
    def __init__(self, processor):
        log.info("Init Background Calculation")
        super(BackgroundCalculationThread, self).__init__()
        self.processor = processor

    def run(self):
        log.info("BackgroundCalculationThread.run()")
        deferred_call(self.set_is_working, True)
        try:
            self.processor.process()
        except Exception, e:
            log.error("Error in BackgroundCalculationThread:")
            log.exception(e)
        finally:
            deferred_call(self.set_is_working, False)

    def set_is_working(self, v):
        self.processor.is_working = v


class OfflineCalibrationController(CalibrationController):
    processor = Typed(OfflineCalibrationProcessor)
    parameters = Typed(OfflineCalibrationParameters)

    bgThread = Typed(BackgroundCalculationThread)

    is_working = Bool(False)
    has_result = Bool(False)

    # results received from background thread
    tooltip_calibration_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    absolute_orientation_result = Value(math.Pose(math.Quaternion(), np.array([0, 0, 0])))
    jointangles_correction_result = Value(angle_null_correction.copy())
    gimbalangles_correction_result = Value(angle_null_correction.copy())

    def _default_parameters(self):
        return OfflineCalibrationParameters()

    def setupController(self, active_widgets=None):
        active_widgets[0].find("btn_start_calibration").visible = False
        active_widgets[0].find("btn_stop_calibration").visible = False
        self.do_reset_parameters()

    # XXX this should use the new configuration.schema implementation !!!
    def do_reset_parameters(self):
        wiz_cfg = self.wizard_state.config
        gbl_cfg = self.context.get("config")

        # load all parameters from the configuration file
        try:
            haptidevice_name = wiz_cfg.get("haptic_device").strip()
            hd_cfg = dict(gbl_cfg.items("ubitrack.devices.%s" % haptidevice_name))
            self.parameters.joint_lengths = np.array([float(hd_cfg["joint_length1"]),
                                           float(hd_cfg["joint_length2"]), ])

            self.parameters.origin_offset = np.array([float(hd_cfg["origin_offset_x"]),
                                           float(hd_cfg["origin_offset_y"]),
                                           float(hd_cfg["origin_offset_z"]),])
        except Exception, e:
            log.error("Error reading Haptic device configuration. Make sure, the configuration file is correct.")
            log.exception(e)

        if self.module.parent.config_version < 2:
            log.warn("This controller (%s) requires config_version >= 2" % self.module_name)

        parameters_sname = "%s.parameters.%s" % (self.config_ns, self.module_name)
        datasource_sname_prefix = "%s.datasources." % self.config_ns
        calibsource_sname_prefix = "%s.calibrations." % self.config_ns

        if gbl_cfg.has_section(parameters_sname):
            self.parameters.tooltip_enabled = gbl_cfg.getboolean(parameters_sname, "tooltip_enabled")
            if self.parameters.tooltip_enabled:
                log.info("Tooltip Calibration Enabled")
                self.parameters.tooltip_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "tooltip_datasource")
                self.parameters.tt_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "tt_minimal_angle_between_measurements")
                self.parameters.tt_use_pose = gbl_cfg.getboolean(parameters_sname, "tt_use_pose")

            if gbl_cfg.has_option(parameters_sname, "fwkbase_position_enabled"):
                self.parameters.fwkbase_position_enabled = gbl_cfg.getboolean(parameters_sname, "fwkbase_position_enabled")
                if self.parameters.fwkbase_position_enabled:
                    log.info("FWKBase Position Calibration Enabled")
                    self.parameters.fwkbase_position_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "fwkbase_position_datasource")

            if gbl_cfg.has_option(parameters_sname, "fwkbase_position2_enabled"):
                self.parameters.fwkbase_position2_enabled = gbl_cfg.getboolean(parameters_sname, "fwkbase_position2_enabled")
                if self.parameters.fwkbase_position2_enabled:
                    log.info("FWKBase Position2 Calibration Enabled")
                    self.parameters.fwkbase_position2_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "fwkbase_position2_datasource")

            self.parameters.absolute_orientation_enabled = gbl_cfg.getboolean(parameters_sname, "absolute_orientation_enabled")
            if self.parameters.absolute_orientation_enabled:
                log.info("Absolute Orientation Calibration Enabled")
                self.parameters.absolute_orientation_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "absolute_orientation_datasource")

                if gbl_cfg.has_option(parameters_sname, "ao_initialize_anglecorrection_calibsource"):
                    self.parameters.ao_initialize_anglecorrection_calibsource = calibsource_sname_prefix + gbl_cfg.get(parameters_sname, "ao_initialize_anglecorrection_calibsource")

                if gbl_cfg.has_option(parameters_sname, "ao_number_of_clusters"):
                    self.parameters.ao_number_of_clusters = gbl_cfg.getint(parameters_sname, "ao_number_of_clusters")

                # experimental absolute orientation with multiple methods
                ao_method = 'fwkpose'
                if gbl_cfg.has_option(parameters_sname, "ao_method"):
                    ao_method = gbl_cfg.get(parameters_sname, "ao_method")
                self.parameters.ao_method = ao_method

                if ao_method == "fwkpose":
                    self.parameters.ao_inital_maxdistance_from_origin = gbl_cfg.getfloat(parameters_sname, "ao_inital_maxdistance_from_origin")
                    self.parameters.ao_minimal_distance_between_measurements = gbl_cfg.getfloat(parameters_sname, "ao_minimal_distance_between_measurements")
                    self.parameters.ao_refinement_expand_coverage = gbl_cfg.getfloat(parameters_sname, "ao_refinement_expand_coverage")
                    self.parameters.ao_refinement_shrink_distance = gbl_cfg.getfloat(parameters_sname, "ao_refinement_shrink_distance")
                elif ao_method == "fwkbase":
                    self.parameters.ao_negate_upvector = gbl_cfg.getboolean(parameters_sname, "ao_negate_upvector")

            self.parameters.joint_angle_calibration_enabled = gbl_cfg.getboolean(parameters_sname, "joint_angle_calibration_enabled")
            if self.parameters.joint_angle_calibration_enabled:
                log.info("Joint-Angle Calibration Enabled")
                self.parameters.joint_angle_calibration_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "joint_angle_calibration_datasource")
                self.parameters.ja_minimal_distance_between_measurements = gbl_cfg.getfloat(parameters_sname, "ja_minimal_distance_between_measurements")
                self.parameters.ja_maximum_distance_to_reference = gbl_cfg.getfloat(parameters_sname, "ja_maximum_distance_to_reference")
                self.parameters.ja_refinement_min_difference = gbl_cfg.getfloat(parameters_sname, "ja_refinement_min_difference")
                self.parameters.ja_refinement_max_iterations = gbl_cfg.getint(parameters_sname, "ja_refinement_max_iterations")
                self.parameters.ja_refinement_shrink_distance = gbl_cfg.getfloat(parameters_sname, "ja_refinement_shrink_distance")

                if gbl_cfg.has_option(parameters_sname, "ja_number_of_clusters"):
                    self.parameters.ja_number_of_clusters = gbl_cfg.getint(parameters_sname, "ja_number_of_clusters")

            self.parameters.reference_orientation_enabled = gbl_cfg.getboolean(parameters_sname, "reference_orientation_enabled")
            if self.parameters.reference_orientation_enabled:
                log.info("Reference Orientation Calibration Enabled")
                self.parameters.reference_orientation_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "reference_orientation_datasource")
                self.parameters.ro_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "ro_minimal_angle_between_measurements")

            self.parameters.gimbal_angle_calibration_enabled = gbl_cfg.getboolean(parameters_sname, "gimbal_angle_calibration_enabled")
            if self.parameters.gimbal_angle_calibration_enabled:
                log.info("Gimbal-Angle Calibration Enabled")
                self.parameters.gimbal_angle_calibration_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "gimbal_angle_calibration_datasource")
                self.parameters.ga_minimal_angle_between_measurements = gbl_cfg.getfloat(parameters_sname, "ga_minimal_angle_between_measurements")
                self.parameters.ga_use_tooltip_offset = gbl_cfg.getboolean(parameters_sname, "ga_use_tooltip_offset")

                if gbl_cfg.has_option(parameters_sname, "ga_number_of_clusters"):
                    self.parameters.ga_number_of_clusters = gbl_cfg.getint(parameters_sname, "ga_number_of_clusters")

            if gbl_cfg.has_option(parameters_sname, "timedelay_estimation_enabled"):
                self.parameters.timedelay_estimation_enabled = gbl_cfg.getboolean(parameters_sname, "timedelay_estimation_enabled")
                if self.parameters.timedelay_estimation_enabled:
                    log.info("Time-Delay Estimation Enabled")
                    self.parameters.timedelay_estimation_datasource = datasource_sname_prefix + gbl_cfg.get(parameters_sname, "timedelay_estimation_datasource")

        else:
            log.warn("No parameters found for offline calibration - using defaults. Define parameters in section: %s" % parameters_sname)

    def do_offline_calibration(self):
        if self.processor is None:
            self.processor = OfflineCalibrationProcessor(
                context=self.context,
                config=self.config,
                facade=self.facade,
                dfg_dir=self.dfg_dir,
                dfg_filename=self.dfg_filename,
            )

            self.processor.observe("is_working", self.defer_update_attr)
            self.processor.result.observe(["has_result",
                                           "tooltip_calibration_result",
                                           "absolute_orientation_result",
                                           "jointangles_correction_result",
                                           "gimbalangles_correction_result"],
                                           self.defer_update_attr)

        self.processor.reset(self.parameters)
        self.bgThread = BackgroundCalculationThread(self.processor)
        self.bgThread.start()

    # called from background thread
    def defer_update_attr(self, change):
        if change is not None:
            deferred_call(setattr, self, change['name'], change['value'])

    def do_visualize_results(self):
        # create figures
        from matplotlib.figure import Figure

        result = self.processor.result
        state = self.wizard_state

        poserr = Figure()
        ax1 = poserr.add_subplot(111)
        ax1.boxplot(result.position_errors)
        ax1.set_title("Position Errors")
        ax1.set_ylim(0, max(*[max(e) for e in result.position_errors]))

        ornerr = Figure()
        ax2 = ornerr.add_subplot(111)
        ax2.boxplot(result.orientation_errors)
        ax2.set_title("Orientation Errors")
        ax2.set_ylim(0, max(*[max(e) for e in result.orientation_errors]))

        # create results panel
        panel = OfflineCalibrationResultPanel(name="utic.%s.visualize_result.%s" % (state.current_task, time.time()),
                                              title="Offline Calibration Results: %s" % state.current_task,
                                              boxplot_position_errors=poserr,
                                              boxplot_orientation_errors=ornerr,
                                              )
        # add to layout
        wizard_controller = self.wizard_state.controller
        parent = wizard_controller.wizview.parent
        panel.set_parent(parent)
        # op = FloatItem(item=panel.name,)
        # parent.update_layout(op)
        panel.show()

    def do_export_data(self):
        filename = FileDialogEx.get_save_file_name()
        if filename:
            # collect some metadata in order to describe the data in the file
            metadata = {}
            state = self.wizard_state
            metadata['current_task'] = state.current_task
            metadata['domain_name'] = state.calibration_domain_name
            metadata['setup_name'] = state.calibration_setup_name
            metadata['user_name'] = state.calibration_user_name
            metadata['platform_name'] = state.calibration_platform_name
            metadata['comments'] = state.calibration_comments
            metadata['datetime'] = state.calibration_datetime
            metadata['module_namespace'] = state.controller.module_manager.modules_ns
            metadata['config_namespace'] = state.controller.module_manager.config_ns

            self.bgThread.processor.export_data(filename, metadata=metadata)


class OfflineCalibrationModule(ModuleBase):
    def get_category(self):
        return "Calibration"

    def get_widget_class(self):
        return OfflineCalibrationPanel

    def get_controller_class(self):
        return OfflineCalibrationController