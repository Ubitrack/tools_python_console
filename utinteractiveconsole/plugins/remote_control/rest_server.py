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





class RemoteControlAPIServer(Resource):
    """
    """

    def __init__(self, context, workspace):
        self.context = context
        self.workspace = workspace

    def getChildWithDefault(self, path, request):
        if path == "application_state":
            return ApplicationStateInformation(self.context, self.workspace)
        else:
            return NoResource()

if __name__ == '__main__':  # pragma: no cover
    site = Site(RemoteKVSResource())
    reactor.listenTCP(2048, site)
    reactor.run()