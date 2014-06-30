__author__ = 'jack'

import os
import sys
import datetime
import logging

import networkx as nx
from atom.api import Atom, Value, List, Dict, Bool, Int, Str, Coerced, Typed, observe
import enaml
from enaml.workbench.ui.api import ActionItem
from enaml.workbench.api import Extension
from enaml.workbench.core.command import Command

from enaml.layout.api import InsertItem

from utinteractiveconsole.extension import ExtensionBase
from utinteractiveconsole.workspace import ExtensionWorkspace
from utinteractiveconsole.uthelpers import UbitrackSubProcessFacade, UbitrackFacade, UbitrackFacadeBase
from .module import ModuleManager


log = logging.getLogger(__name__)


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

    config = Dict()
    facade = Typed(UbitrackFacadeBase)

    def _default_config(self):
        cfg = self.context.get("config")
        if cfg.has_section(self.module_manager.config_ns):
            return dict(cfg.items(self.module_manager.config_ns))
        else:
            log.error("Missing section: [%s] in config" % self.module_manager.config_ns)
            return dict()

    @property
    def facade_handler_type(self):
        return self.config.get("facade_handler", "subprocess")

    def _default_facade(self):
        fht = self.facade_handler_type
        facade = None
        if fht == "subprocess":
            facade = UbitrackSubProcessFacade(context=self.context,
                                              config_ns=self.module_manager.config_ns,
                                              )
        elif fht == "inprocess":
            facade = UbitrackFacade(context=self.context,
                                              config_ns=self.module_manager.config_ns,
                                              )
        else:
            log.error("Invalid facade_handler configured in section: %s" % self.module_manager.config_ns)
        return facade

    def _default_calibration_datetime(self):
        return datetime.datetime.now().strftime("%Y%d%m-%H%M")

    def start(self):
        log.info("Start Facade")
        self.facade.start()

    def stop(self):
        log.info("Stop Facade")
        self.facade.stop()


    def _default_tasks(self):
        if len(self.task_list) == 0:
            return []
        return [TaskStatus(name=n) for n in self.task_list]

    def _default_current_task(self):
        if len(self.task_list) == 0:
            log.error("Invalid wizard configuration: no modules enabled")
            return ''
        return self.task_list[self.task_idx]

    def _default_current_module(self):
        if not self.current_task:
            return None
        return self.module_manager.modules[self.current_task]

    def _default_active_widgets(self):
        if self.current_module is None:
            return []
        widget_cls = self.current_module.get_widget_class()
        ctrl = self.current_module.get_controller_class()(context=self.context,
                                                          facade=self.facade,
                                                          wizard_state=self,
                                                          state=self.tasks[self.task_idx],
                                                          module_name=self.current_module.module_name,
                                                          config_ns=self.module_manager.config_ns)


        # XXX this is duplicated code - check with calibration controller and see where it belongs ..
        if self.facade is not None and ctrl.dfg_filename:
            fname = os.path.join(self.facade.dfg_basedir, ctrl.dfg_filename)
            if os.path.isfile(fname):
                self.facade.dfg_filename = fname
            else:
                log.error("Invalid file specified for module: %s" % fname)

        # eventually cache the module widgets and/or controllers ?
        aw = [widget_cls(module=self.current_module,
                         module_state=self.tasks[self.task_idx],
                         module_controller=ctrl,),
              ]

        ctrl.setupController(active_widgets=aw)
        return aw

    @observe("calibration_setup_idx")
    def _handle_calibration_setup_idx_change(self, change):
        self.calibration_setup = ["optitrack_omni", "art_premium", "faro_premium", "optitrack_omni_2nd", "art_premium_2nd", "faro_premium_2nd"][change["value"]]

    @observe("task_idx")
    def _handle_idx_change(self, change):
        if change["type"] == "update" and change["name"] == "task_idx":
            self.current_task = self.task_list[self.task_idx]
            self.current_module = self.module_manager.modules[self.current_task]

            widget_cls = self.current_module.get_widget_class()
            ctrl = self.current_module.get_controller_class()(module_name=self.current_module.get_module_name(),
                                                              context=self.context,
                                                              facade=self.facade,
                                                              state=self.tasks[self.task_idx],
                                                              wizard_state=self,
                                                              config_ns=self.module_manager.config_ns)

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
            ctrl.setupController(active_widgets=self.active_widgets)



class WizardController(Atom):

    context = Value()


    module_manager = Typed(ModuleManager)
    module_graph = Typed(nx.DiGraph)
    current_state = Typed(WizardState)

    wizview = Value()

    def initialize(self, config):

        if self.module_manager is None:
            self.module_manager = ModuleManager(context=self.context,
                                                modules_ns=config['module_namespace'],
                                                config_ns=config['config_namespace'],
                                                )
        if len(self.module_manager.modules.keys()) == 0:
            log.warn("Wizard without modules found: %s" % config['name'])
            return False

        if self.module_graph is None:
            self.module_graph = self.module_manager.graph

        if self.current_state is None:
            self.current_state = WizardState(task_idx=0,
                                             task_list=[m for m in nx.topological_sort(self.module_manager.graph)
                                                        if self.module_manager.modules[m].is_enabled()],
                                             module_manager=self.module_manager,
                                             context=self.context,
                                             )

        return True

    def show(self):
        if self.wizview is not None:
            self.wizview.show()

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
                self.wizview.destroy()
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



class CalibrationWizard(ExtensionBase):

    wizard_instances = {}

    def register(self, mgr):
        name = "calibration_wizard"
        category = "calibration"


        def plugin_factory(workbench):

            with enaml.imports():
                from .views.wizard_main import WizardMain, WizardManifest

            space = ExtensionWorkspace(appstate=mgr.appstate, utic_plugin=self)
            space.window_title = 'Calibration Wizard'
            space.content_def = WizardMain
            space.manifest_def = WizardManifest
            return space

        plugin = Extension(id=name,
                           point="enaml.workbench.ui.workspaces",
                           factory=plugin_factory)


        action = ActionItem(path="/workspace/calibration_wizard",
                            label="Calibration Wizard",
                            shortcut= "Ctrl+W",
                            before="load_dataflow",
                            command="enaml.workbench.ui.select_workspace",
                            parameters={'workspace': "utic.%s" % name, }
                            )

        mgr.registerExtension(name, self, category=category,
                              action_items=[action, ],
                              workspace_plugins=[plugin, ],
                              add_globals=dict(calibration_wizards=self.wizard_instances))

        return self

    def get_name(self):
        return "Calibration Wizard"

    def get_ports(self):
        return []

    def get_port(self, name):
        raise ValueError("tbd")

    def get_wizard_config(self):
        # eventually cache this one - or reify ..
        wcfg = []
        config = self.context.get('config')
        if config is not None:
            if config.has_section('calibration_wizard'):
                wizards = config.get('calibration_wizard', 'wizards')
                if wizards is not None:
                    for wizard_name in [n.strip() for n in wizards.split(',')]:
                        config_ok = True
                        wizard_def = None
                        wizard_config = None
                        if config.has_section('calibration_wizard.%s' % wizard_name):
                            wizard_def = dict(config.items('calibration_wizard.%s' % wizard_name))
                        else:
                            config_ok = False
                            log.warn('Missing config section: calibration_wizard.%s' % wizard_name)

                        if wizard_def is not None and wizard_def.get('config_namespace') is not None:
                            if config.has_section(wizard_def.get('config_namespace')):
                                wizard_config = dict(config.items(wizard_def.get('config_namespace')))
                            else:
                                config_ok = False
                                log.warn('Missing config section: %s' % wizard_def.get('config_namespace'))
                        else:
                            config_ok = False

                        if config_ok:
                            wcfg.append([wizard_name, wizard_def, wizard_config])
                        else:
                            log.warn('Invalid wizard config: %s' % wizard_name)
        return wcfg

    # dynamically generate wizards from configuration
    def generateWorkspaceActionItems(self, plugin_ext):
        result = []
        wcfg = self.get_wizard_config()

        for name, wizard_def, wizard_cfg in wcfg:
            log.info("Activate calibration wizard: %s" % wizard_def.get('name'))

            item = ActionItem(path="/calibration/%s" % name,
                              label=wizard_def.get('name'),
                              command='utic.commands.extensions.calibration_wizard.launch_%s' % name,
                              parameters={'wizard_name': name,
                                          'wizard_def': wizard_def,
                                          'wizard_cfg': wizard_cfg,}
                              )
            result.append(item)

        return result

    def generateWorkspaceCommands(self, plugin_manifest):
        result = []
        wcfg = self.get_wizard_config()

        for name, wizard_def, wizard_cfg in wcfg:

            cmd = Command(id='utic.commands.extensions.calibration_wizard.launch_%s' % name,
                          description="Launch %s" % wizard_def.get('name'),
                          handler=self.launch_wizard,
                          )
            result.append(cmd)
        return result

    # XXX should be split into event-handler and launch_wizard method
    def launch_wizard(self, ev):
        params = ev.parameters
        wizard_instances = self.wizard_instances
        name = params.get('wizard_name')
        wizard_def = params.get('wizard_def')



        if name in self.wizard_instances:
            wizard = self.wizard_instances[name]
            wizard.show()
            # add to layout
            # XXX maybe better to use workspace.find("content") or simlar
            parent = ev.workbench.get_plugin('enaml.workbench.ui').workspace.content.area
            op = InsertItem(item=name,)
            parent.update_layout(op)

        else:
            wizard = WizardController(context=self.context)
            if not wizard.initialize(wizard_def):
                log.error("Error launching wizard: %s" % name)
                return

            with enaml.imports():
                from .views.wizard_main import WizardView

            workspace = ev.workbench.get_plugin('enaml.workbench.ui').workspace

            def cleanup(*args):
                log.info("cleanup for wizard: %s" % name)
                if wizard.wizview is not None:
                    wizard.wizview.unobserve("closed", cleanup)
                wizard.wizview = None

                if wizard.current_state is not None:
                    wizard.current_state.stop()
                wizard.current_state = None

                if name in wizard_instances:
                    wizard_instances.pop(name)
                workspace.unobserve("stopped", cleanup)

            if wizard.current_state is not None:
                wizard.current_state.start()

            if wizard.wizview is None:
                wizard.wizview = WizardView(name=name,
                                            title="Calibrate: %s" % wizard_def.get('name'),
                                            controller=wizard,
                                            state=wizard.current_state)
                wizard.wizview.observe("closed", cleanup)
                workspace.observe("stopped", cleanup)

                # add to layout
                parent = workspace.content.area
                wizard.wizview.set_parent(parent)
                op = InsertItem(item=name,)
                parent.update_layout(op)

            wizard.show()

            # start the watchdog timer for subprocess control
            wizard.wizview.control.subprocess.timer.start()
            wizard_instances[name] = wizard
