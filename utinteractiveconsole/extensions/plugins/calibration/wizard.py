__author__ = 'jack'

import os, sys
import networkx as nx
import datetime

from atom.api import Atom, Value, List, Bool, Int, Str, Coerced, Typed, observe
import enaml

from utinteractiveconsole.extensions import ExtensionBase
from utinteractiveconsole.uthelpers import UbitrackSubProcessFacade

from .module import ModuleManager

import logging
log = logging.getLogger(__name__)


# with enaml.imports():
#     from .views.wizard import WizardMain, LoadCalibrationWizardAction


class TaskResult(Atom):
    value = Value()

    calib_files = List()
    recorded_files = List()

    def to_str(self):
        results = []

        if self.value:
            results.append("Result: %s" % str(self.value))
        if self.calib_files:
            results.append("Calibration files: %s" % (", ".join([os.path.basename(f) for f in self.calib_files])))
        if self.recorded_files:
            results.append("Recorded files: %s" % (", ".join([os.path.basename(f) for f in self.recorded_files])))

        if results:
            return "<br/>".join(results)
        return "No results"

    def to_dict(self):
        result = dict()
        if self.value:
            result["value"] = self.value
        if self.calib_files:
            result["calib_files"] = self.calib_files[:]
        if self.recorded_files:
            result["recorded_files"] = self.recorded_files[:]
        return result


class TaskStatus(Atom):
    name = Coerced(str)
    running = Bool(False)
    started = Bool(False)
    completed = Bool(False)
    skipped = Bool(False)

    result = Typed(TaskResult)

    def _default_result(self):
        return TaskResult()

    def to_html(self):
        result_str = ""
        if not self.skipped:
            result_str = "<p>Result: %s</p>" % self.result.to_str()
        return """<h3>Task: %s (%s)</h3>
        %s
        """ % (self.name, "completed" if self.completed else "skipped", result_str)

    def to_dict(self):
        result = dict(name=self.name,
                      completed=self.completed,
                      skipped=self.skipped,)
        tr = self.result.to_dict()
        if tr:
            result["result"] = tr
        return result



class WizardState(Atom):
    context = Value()
    module_manager = Value()
    task_list = List(Coerced(str))
    task_idx = Int()
    tasks = List(TaskStatus)
    completed = Bool(False)

    current_task = Coerced(str)
    current_module = Value()

    active_widgets = Value()

    calibration_setup_idx = Int(-1)
    calibration_setup = Str()
    calibration_operator = Str()
    calibration_comments = Str()
    calibration_datetime = Str()
    calibration_dataok = Bool(False)

    facade = Typed(UbitrackSubProcessFacade)

    def _default_facade(self):
        facade = UbitrackSubProcessFacade(context=self.context)
        return facade

    def _default_calibration_datetime(self):
        return datetime.datetime.now().strftime("%Y%d%m-%H%M")

    def start(self):
        self.facade.start()

    def stop(self):
        self.facade.stop()


    def _default_tasks(self):
        return [TaskStatus(name=n) for n in self.task_list]

    def _default_current_task(self):
        return self.task_list[self.task_idx]

    def _default_current_module(self):
        return self.module_manager.modules[self.current_task]

    def _default_active_widgets(self):
        widget_cls = self.current_module.get_widget_class()
        ctrl = self.current_module.get_controller_class()(context=self.context,
                                                          facade=self.facade,
                                                          wizard_state=self,
                                                          state=self.tasks[self.task_idx],)
        if self.facade is not None and ctrl.dfg_filename:
            fname = os.path.join(self.facade.dfg_basedir, ctrl.dfg_filename)
            if os.path.isfile(fname):
                self.facade.dfg_filename = fname
            else:
                log.error("Invalid file specified for module: %s" % fname)

        # eventually cache the module widgets and/or controllers ?
        return [widget_cls(module=self.current_module,
                           module_state=self.tasks[self.task_idx],
                           module_controller=ctrl,),
                ]

    @observe("calibration_setup_idx")
    def _handle_calibration_setup_idx_change(self, change):
        self.calibration_setup = ["optitrack_omni", "art_premium", "faro_premium", "optitrack_omni_2nd", "art_premium_2nd", "faro_premium_2nd"][change["value"]]

    @observe("task_idx")
    def _handle_idx_change(self, change):
        if change["type"] == "update" and change["name"] == "task_idx":
            self.current_task = self.task_list[self.task_idx]
            self.current_module = self.module_manager.modules[self.current_task]


    # @observe("current_task")
    # def _handle_current_task_change(self, change):
    #     if change["type"] == "update" and change["name"] == "current_task":
            widget_cls = self.current_module.get_widget_class()
            ctrl = self.current_module.get_controller_class()(module_name=self.current_module.get_module_name(),
                                                              context=self.context,
                                                              facade=self.facade,
                                                              state=self.tasks[self.task_idx],
                                                              wizard_state=self,)

            if self.facade is not None and ctrl.dfg_filename:
                fname = os.path.join(self.facade.dfg_basedir, ctrl.dfg_filename)
                if os.path.isfile(fname):
                    self.facade.dfg_filename = fname
                else:
                    log.error("Invalid file specified for module: %s" % fname)

            # eventually cache the module widgets and/or controllers ?
            self.active_widgets = [widget_cls(module=self.current_module,
                                              module_state=self.tasks[self.task_idx],
                                              module_controller=ctrl,),
                                   ]



class CalibrationWizard(ExtensionBase):

    wizview = None
    module_manager = None
    module_graph = None
    current_state = None

    wizard_modules_ns = 'calibration_wizard.modules'
    wizard_config_ns = 'calibration_wizard'

    def update_optparse(self, parser):
        parser.add_option("--wizard-modules-ns",
                      action="store", dest="wizard_modules_ns", default=self.wizard_modules_ns,
                      help="python package name where calibration wizard modules are loaded from.")

        parser.add_option("--wizard-config-ns",
                      action="store", dest="wizard_config_ns", default=self.wizard_config_ns,
                      help="configuration namespace for calibration wizard")


    def register(self, mgr):
        win = self.context.get("win")
        options = self.context.get("options")

        if self.module_manager is None:
            self.module_manager = ModuleManager(context=self.context,
                                                modules_ns=options.wizard_modules_ns,
                                                config_ns=options.wizard_config_ns,
                                                )

        if self.module_graph is None:
            self.module_graph = self.module_manager.graph

        def cleanup(*args):
            self.wizview = None
            # only if window was exited
            self.current_state.stop()
            self.current_state = None

        def start_wizard():
            if self.current_state is None:
                self.current_state = WizardState(task_idx=0,
                                                 task_list=[m for m in nx.topological_sort(self.module_manager.graph) if self.module_manager.modules[m].is_enabled()],
                                                 module_manager=self.module_manager,
                                                 context=self.context,
                                                 )
                self.current_state.start()
            if self.wizview is None:
                self.wizview = WizardMain(controller=self,
                                          state=self.current_state)
                self.wizview.observe("closed", cleanup)

            self.wizview.show()

            # start the watchdog timer for subprocess control
            self.wizview.control.subprocess.timer.start()



        action = LoadCalibrationWizardAction(handler=start_wizard)
        menu_item = dict(menu_id="vhar_calibration", menu_title="&CalibrationWizard", menu_action=action)
        mgr.registerExtension("vhar_calibration", self, category="vhar_calibration", menu_items=[menu_item, ],
                              add_globals=dict(calibration_wizard=self))

        return self

    def get_name(self):
        return "Calibration Wizard"

    def get_ports(self):
        return []

    def get_port(self, name):
        raise ValueError("tbd")




    def on_ok(self, content):
        state = self.current_state
        if state is not None:
            log.info("Mark task %s as completed" % state.task_list[state.task_idx])
            mstate = state.tasks[state.task_idx]
            mstate.running = False
            mstate.completed = True

            if state.active_widgets:
                # currently assumes len(1)
                mctrl = state.active_widgets[0].module_controller
                mstate.result.calib_files = mctrl.getCalibrationFiles()
                mstate.result.recorded_files = mctrl.getRecordedFiles()

                if mctrl.save_results:
                    cfg = state.context.get("config")
                    if cfg.has_section("vharcalib"):
                        vc_cfg = dict(cfg.items("vharcalib"))
                        results_path = os.path.join(vc_cfg["rootdir"], vc_cfg.get("resultsdir", "results"), state.calibration_setup, state.calibration_datetime, mctrl.module_name)
                        if not os.path.isdir(results_path):
                            os.makedirs(results_path)
                        mctrl.saveResults(results_path)

            self.next_task(content)

    def on_skip(self, content):
        state = self.current_state
        if state is not None:
            log.info("Mark task %s as skipped" % state.task_list[state.task_idx])
            state.tasks[state.task_idx].running = False
            state.tasks[state.task_idx].skipped = True
            self.next_task(content)

    def next_task(self, content):
        state = self.current_state
        if state is not None:
            if len(state.task_list) > state.task_idx + 1:
                state.task_idx += 1
            else:
                state.completed = True
                self.wizview.close()
                return

            log.info("Start task: %s" % state.task_list[state.task_idx])
            state.tasks[state.task_idx].started = True
            state.tasks[state.task_idx].running = True


    def previous_task(self, content):
        state = self.current_state
        if state is not None:
            if state.task_idx > 0:
                state.task_idx -= 1
            else:
                return

            log.info("Start task: %s" % state.task_list[state.task_idx])
            state.tasks[state.task_idx].started = True
            state.tasks[state.task_idx].running = True