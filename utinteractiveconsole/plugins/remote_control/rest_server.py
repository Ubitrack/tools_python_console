__author__ = 'jack'
import json
import inspect

from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.web.resource import NoResource
from twisted.web.server import Site

from twisted.internet.task import deferLater
from twisted.web.server import NOT_DONE_YET


class DelayedResource(Resource):


    def _delayedRender(self, request):
        request.write(json.dumps({"success": True}))
        request.finish()

    def render_GET(self, request):
        d = deferLater(reactor, 0, lambda: request)
        d.addCallback(self._delayedRender)
        return NOT_DONE_YET

    def render_POST(self, request):
        print request.content.read()
        d = deferLater(reactor, 0, lambda: request)
        d.addCallback(self._delayedRender)
        return NOT_DONE_YET


class ApplicationStateInformation(Resource):
    isLeaf = True

    def __init__(self, context, workspace):
        self.context = context
        self.workspace = workspace

    def render_GET(self, request):
        response = {'success': True }
        ws = self.workspace
        application_state = {}
        if ws.utic_plugin is not None:
            for name, ctrl in ws.utic_plugin.wizard_instances.items():
                wizard_state = dict(name=name)
                cs = ctrl.current_state
                wizard_state['current_task'] = cs.current_task
                wizard_state['task_idx'] = cs.task_idx
                wizard_state['task_list'] = cs.task_list
                task_status = []
                for ts in cs.tasks:
                    s = {}
                    s['name'] = ts.name
                    s['started'] = ts.started
                    s['running'] = ts.running
                    s['completed'] = ts.completed
                    s['skipped'] = ts.skipped
                    task_status.append(s)
                wizard_state['task_status'] = task_status
                application_state[name] = wizard_state

            response['result'] = application_state

        return json.dumps(response)

class WizardCommands(Resource):
    isLeaf = True

    def __init__(self, context, workspace):
        self.context = context
        self.workspace = workspace

    def render_GET(self, request):
        from enaml.workbench.ui.action_item import ActionItem

        response = {'success': True }
        ws = self.workspace
        wb = self.context['workbench']
        cwm = wb.get_manifest("uticmain.calibration_wizard")

        application_commands = []
        for ext in cwm.extensions:
            for ai in ext.get_children(ActionItem):
                application_commands.append(dict(label=ai.label, path=ai.path, command=ai.command))

        response['result'] = application_commands
        return json.dumps(response)

    def render_POST(self, request):
        from enaml.workbench.ui.action_item import ActionItem

        response = {'success': True }
        request_obj = {}
        try:
            request_obj = json.loads(request.content.read())
        except Exception, e:
            response["success"] = False
            response["msg"] = "Exception: %s" % str(e)

        if "command" in request_obj:
            ws = self.workspace
            wb = self.context['workbench']
            cwm = wb.get_manifest("uticmain.calibration_wizard")
            item = None
            for ext in cwm.extensions:
                for ai in ext.get_children(ActionItem):
                    if ai.command == request_obj["command"]:
                        item = ai
                        break
                if item is not None:
                    break

            if item is None:
                response["success"] = False
                response["msg"] = "Invalid command"
            else:
                core = wb.get_plugin('enaml.workbench.core')
                core.invoke_command(item.command, item.parameters, item)
        else:
            response["success"] = False
            response["msg"] = "Invalid request"

        return json.dumps(response)

class WizardControl(Resource):

    def __init__(self, context, workspace):
        self.context = context
        self.workspace = workspace

    def render_GET(self, request):
        from enaml.widgets.push_button import PushButton

        response = {'success': True }
        ws = self.workspace
        wizard_actions = {}
        if ws.utic_plugin is not None:
            for name, ctrl in ws.utic_plugin.wizard_instances.items():
                actions = []
                for obj in ctrl.wizview.find_all("btn_.*", True):
                    if isinstance(obj, PushButton) and obj.visible and obj.enabled:
                        actions.append(obj.name)
                wizard_actions[name] = actions

        response['result'] = wizard_actions
        return json.dumps(response)

    def render_POST(self, request):
        from enaml.widgets.push_button import PushButton

        response = {'success': True }
        request_obj = {}
        try:
            request_obj = json.loads(request.content.read())
        except Exception, e:
            response["success"] = False
            response["msg"] = "Exception: %s" % str(e)

        if "action" in request_obj:
            ws = self.workspace
            action = None
            if ws.utic_plugin is not None:
                for name, ctrl in ws.utic_plugin.wizard_instances.items():
                    for obj in ctrl.wizview.find_all("btn_.*", True):
                        if isinstance(obj, PushButton) and obj.visible and obj.enabled and obj.name == request_obj["action"]:
                            action = obj
                            break
            if action is None:
                response["success"] = False
                response["msg"] = "Invalid action"
            else:
                action.clicked()
        else:
            response["success"] = False
            response["msg"] = "Invalid request"

        return json.dumps(response)


class RemoteControlAPIServer(Resource):
    """
    """

    def __init__(self, context, workspace):
        self.context = context
        self.workspace = workspace

    def getChildWithDefault(self, path, request):
        if path == "application_state":
            return ApplicationStateInformation(self.context, self.workspace)
        elif path == "wizard_commands":
            return WizardCommands(self.context, self.workspace)
        elif path == "wizard_control":
            return WizardControl(self.context, self.workspace)
        else:
            return NoResource()

if __name__ == '__main__':  # pragma: no cover
    site = Site(RemoteKVSResource())
    reactor.listenTCP(2048, site)
    reactor.run()