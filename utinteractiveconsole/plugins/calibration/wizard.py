__author__ = 'jack'

import os
import sys
import datetime
import logging
import stevedore

import networkx as nx
from atom.api import Atom, Value, Event, List, Dict, Bool, Int, Str, Coerced, Typed, observe
import enaml
from enaml.workbench.ui.api import ActionItem
from enaml.workbench.api import Extension
from enaml.workbench.core.command import Command

from enaml.layout.api import InsertItem

from utinteractiveconsole.extension import ExtensionBase
from utinteractiveconsole.workspace import ExtensionWorkspace
from utinteractiveconsole.uthelpers import (UbitrackSubProcessFacade, UbitrackFacade,
                                            UbitrackFacadeBase)

from .module import ModuleManager
with enaml.imports():
    from utinteractiveconsole.plugins.calibration.views.module_templates import ModuleContainer



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
    controller = Value()

    module_manager = Value()
    task_list = List(Coerced(str))
    task_idx = Int()
    tasks = List(TaskStatus)
    completed = Bool(False)

    current_task = Coerced(str)
    current_module = Value()

    active_widgets = Value()

    config = Dict()
    wizard_name = Str()
    facade = Typed(UbitrackFacadeBase)

    calibration_domain_name = Str()
    calibration_setup_name = Str()
    calibration_user_name = Str()
    calibration_platform_name = Str()
    calibration_comments = Str()
    calibration_datetime = Str()
    calibration_dataok = Bool(False)

    calibration_external_tracker_change = Bool(False)
    calibration_haptic_device_change = Bool(False)
    calibration_existing_delete_files = Bool(True)

    enable_start_calibration_button = Bool(False)
    enable_stop_calibration_button = Bool(False)

    show_back_button = Bool(True)
    show_skip_button = Bool(True)
    show_next_button = Bool(True)

    enable_back_button = Bool(False)
    enable_skip_button = Bool(False)
    enable_next_button = Bool(True)

    text_back_button = Str('Back')
    text_skip_button = Str('Skip')
    text_next_button = Str('Next')

    # Events
    on_wizard_after_start = Event()
    on_module_after_load = Event()
    on_module_after_start = Event()
    on_module_before_stop = Event()
    on_module_before_unload = Event()
    on_wizard_before_stop = Event()


    @observe('task_list', 'task_idx')
    def _update_gui_data(self, change):
        self.text_next_button = 'Next' if len(self.task_list) > self.task_idx + 1 else "Finish"
        self.show_back_button = bool(self.task_idx > 1)
        self.show_skip_button = bool(len(self.task_list) > self.task_idx + 1)
        self.enable_back_button = bool(self.task_idx > 1)
        self.enable_skip_button = bool(self.task_idx > 0 and len(self.task_list) > self.task_idx + 1)

    def _default_config(self):
        cfg = self.context.get("config")
        if cfg.has_section(self.module_manager.config_ns):
            return dict(cfg.items(self.module_manager.config_ns))
        else:
            log.error("Missing section: [%s] in config" % self.module_manager.config_ns)
            return dict()

    def _default_wizard_name(self):
        name = self.config.get("name", None)
        if name is None:
            log.warn("Wizard config at: %s does not specify name" % self.module_manager.config_ns)
            return "Undefined Calibration Wizard"
        return name

    @property
    def facade_handler_type(self):
        return self.config.get("facade_handler", "subprocess")

    def _default_facade(self):
        fht = self.facade_handler_type
        facade = None
        if fht == "subprocess":
            facade = UbitrackSubProcessFacade(context=self.context,)
        elif fht == "inprocess":
            facade = UbitrackFacade(context=self.context,)
        else:
            log.error("Invalid facade_handler configured in section: %s" % self.module_manager.config_ns)
        return facade

    def _default_calibration_datetime(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M")

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
        widget_content_cls = self.current_module.get_widget_class()
        ctrl = self.current_module.get_controller_class()(context=self.context,
                                                          facade=self.facade,
                                                          wizard_state=self,
                                                          state=self.tasks[self.task_idx],
                                                          module_name=self.current_module.module_name,
                                                          module=self.current_module,
                                                          config_ns=self.module_manager.config_ns)

        widget_cls = ModuleContainer(widget_content_cls, type(self.facade))
        aw = widget_cls(module=self.current_module,
                        module_state=self.tasks[self.task_idx],
                        module_controller=ctrl,
                        )
        ctrl.setupController(active_widgets=aw)
        self.on_module_after_load(ctrl.module_name)
        return aw

    @observe("task_idx")
    def _handle_idx_change(self, change):
        if change["type"] == "update" and change["name"] == "task_idx":
            # teardown the existing controller
            for w in self.active_widgets:
                if w.module_controller is not None:
                    self.on_module_before_unload(w.module_controller.module_name)
                    w.module_controller.teardownController(active_widgets=self.active_widgets)

            # switch to selected task
            self.current_task = self.task_list[self.task_idx]
            self.current_module = self.module_manager.modules[self.current_task]

            # create controller and widgets for current task
            widget_content_cls = self.current_module.get_widget_class()
            ctrl = self.current_module.get_controller_class()(module_name=self.current_module.get_module_name(),
                                                              module=self.current_module,
                                                              context=self.context,
                                                              facade=self.facade,
                                                              state=self.tasks[self.task_idx],
                                                              wizard_state=self,
                                                              config_ns=self.module_manager.config_ns)

            widget_cls = ModuleContainer(widget_content_cls, type(self.facade))
            self.active_widgets = widget_cls(module=self.current_module,
                                             module_state=self.tasks[self.task_idx],
                                             module_controller=ctrl,
                                             )
            ctrl.setupController(active_widgets=self.active_widgets)
            self.on_module_after_load(ctrl.module_name)



class WizardController(Atom):

    context = Value()
    workbench = Value()

    module_manager = Typed(ModuleManager)
    module_graph = Typed(nx.DiGraph)
    current_state = Typed(WizardState)

    wizview = Value()
    # XXX needs better structure .. maybe ..
    preview = Value()
    preview_controller = Value()

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

        modules_ok = True
        for m in nx.topological_sort(self.module_manager.graph):
            if not m in self.module_manager.modules:
                log.warn("Invalid configuration - Missing module: %s" % m)
                modules_ok = False

        if not modules_ok:
            log.warn("Available modules: %s" % ",".join(self.module_manager.modules.keys()))
            return False

        if self.current_state is None:
            self.current_state = WizardState(task_idx=0,
                                             task_list=[m for m in nx.topological_sort(self.module_manager.graph)
                                                        if self.module_manager.modules[m].is_enabled()],
                                             module_manager=self.module_manager,
                                             context=self.context,
                                             controller=self,
                                             )
                                            # XXX something missing here ?

        self.current_state.on_wizard_after_start()
        return True

    def teardown(self):
        if self.preview_controller is not None:
            self.preview_controller.teardown()

        self.current_state.on_wizard_before_stop()
        self.wizview.proxy.widget.close()


    def show(self):
        if self.wizview is not None:
            self.wizview.show()
        if self.preview is not None:
            self.preview.show()

    def hide(self):
        if self.wizview is not None:
            self.wizview.hide()
        if self.preview is not None:
            self.preview.hide()

    def on_next(self, content):
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
                    if 'resultsdir' in state.config:
                        results_path = os.path.join(os.path.expanduser(state.config["resultsdir"]),
                                                    state.calibration_datetime)
                        if not os.path.isdir(results_path):
                            os.makedirs(results_path)
                        mctrl.saveResults(results_path)

            self.next_task(content)

    def on_back(self, content):
        state = self.current_state
        if state is not None:
            #log.info("Mark task %s as completed" % state.task_list[state.task_idx])
            mstate = state.tasks[state.task_idx]
            # mstate.started = False
            mstate.running = False
            # mstate.completed = False

            self.previous_task(content)

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
                self.teardown()
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
    live_previews = {}

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
                            before="close",
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
        live_previews = self.live_previews
        name = params.get('wizard_name')
        wizard_def = params.get('wizard_def')
        wizard_cfg = params.get('wizard_cfg')

        if name in wizard_instances:
            wizard = wizard_instances[name]
            wizard.show()
            da = ev.workbench.get_plugin('enaml.workbench.ui').workspace.content.find("wizard_dockarea")
            op = InsertItem(item=name, target="new", position="right")
            da.update_layout(op)

            if name in live_previews:
                preview = live_previews[name]
                preview.show()
                op = InsertItem(item=preview.name, target=name, position="right")
                da.update_layout(op)

        else:
            wizard = WizardController(context=self.context, workbench=ev.workbench)
            if not wizard.initialize(wizard_def):
                log.error("Error launching wizard: %s" % name)
                return

            with enaml.imports():
                from .views.wizard_main import WizardView

            workspace = ev.workbench.get_plugin('enaml.workbench.ui').workspace

            def cleanup(*args):
                log.info("Cleanup for wizard: %s" % name)

                if wizard.preview is not None:
                    wizard.preview.destroy()
                    wizard.preview = None

                if wizard.wizview is not None:
                    wizard.wizview.unobserve("closed", cleanup)
                    wizard.wizview = None

                if wizard.current_state is not None:
                    log.info("Stop currently active processes")
                    wizard.current_state.stop()
                    wizard.current_state = None

                if name in wizard_instances:
                    wizard_instances.pop(name)
                workspace.unobserve("stopped", cleanup)
                log.info("Finished cleanup for wizard: %s" % name)

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
                parent = workspace.content.find("wizard_dockarea")
                wizard.wizview.set_parent(parent)
                op = InsertItem(item=wizard.wizview.name,)
                parent.update_layout(op)

            if wizard.preview is None and wizard_cfg.get("livepreview", None) is not None:
                preview_controller_name = wizard_cfg.get("livepreview")
                mgr = stevedore.extension.ExtensionManager
                pce = mgr(namespace="vharcalibration.controllers.preview",
                          invoke_on_load=True,
                          invoke_args=(wizard, self.context, wizard_def.get("config_namespace") ),
                          )
                if preview_controller_name in pce.names():
                    pc = pce[preview_controller_name].obj.create(workspace, name, parent)
                    wizard.preview_controller = pc
                    pc.initialize()
                else:
                    log.warn("Invalid LivePreview config: %s" % preview_controller_name)

            wizard.show()

            # start the watchdog timer for subprocess control
            if hasattr(wizard.wizview, "control"):
                wizard.wizview.control.subprocess.timer.start()

            wizard_instances[name] = wizard
