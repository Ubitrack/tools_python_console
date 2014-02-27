__author__ = 'jack'
from multiprocessing import Process, Pipe
import logging

log = logging.getLogger(__name__)

class SubProcess(object):

    def __call__(self, conn, name, logConfig="log4cpp.conf", components_path="/usr/local/lib/ubitrack", poll_timeout=0.01, debug=False):
        from ubitrack.core import math, measurement, util
        from ubitrack.facade import facade

        if debug:
            logging.basicConfig(level=logging.DEBUG)

        util.initLogging(logConfig)
        facade = facade.AdvancedFacade(components_path)

        is_running = True

        while is_running:
            is_ready = conn.poll(poll_timeout)
            if is_ready:
                msg = conn.recv()
                log.debug("SubProcess received message: %s" % (msg,))

                if isinstance(msg, dict) and "cmd" in msg:
                    # command
                    cmd = msg["cmd"]
                    if cmd == "loadDataflow":
                        if "filename" in msg:
                            dfg_filename = msg["filename"]
                            log.info("SubProcess %s loadDataflow: %s" % (name, dfg_filename))
                            facade.loadDataflow(dfg_filename, True)
                        else:
                            log.error("Subprocess %s missing filename" % (name,))
                    elif cmd == "startDataflow":
                        log.info("SubProcess %s startDataflow")
                        facade.startDataflow()
                    elif cmd == "stopDataflow":
                        log.info("SubProcess %s stopDataflow")
                        facade.stopDataflow()
                    elif cmd == "clearDataflow":
                        log.info("SubProcess %s clearDataflow")
                        facade.clearDataflow()
                    elif cmd == "killEverything":
                        log.info("SubProcess %s killEverything")
                        facade.killEverything()
                    elif cmd == "exit":
                        log.info("SubProcess %s is exiting")
                        is_running = False

        conn.send(["terminate"])
        conn.close()




class SubProcessManager(object):

    def __init__(self, name, debug=False, logConfig="log4cpp.conf", components_path="/usr/local/lib/ubitrack", poll_timeout=0.2):
        self.name = name
        self.logConfig = logConfig
        self.components_path = components_path
        self.poll_timeout = poll_timeout

        self.parent_conn, self.child_conn = Pipe()
        self.process = None
        self.started = False

    def start(self):
        if self.started:
           return

        kwargs = dict(logConfig=self.logConfig,
                      components_path=self.components_path,
                      poll_timeout=self.poll_timeout)

        self.process = Process(target=SubProcess(),
                               args=(self.child_conn, self.name),
                               kwargs=kwargs)
        log.info("start subprocess %s" % (self.name,))
        self.process.start()
        self.started = True

    def stop(self, timeout=10.0):
        self.started = False
        if not self.is_alive():
            return

        # could be implemented better ..
        # if self.is_running():
        self.stopDataflow()
        self.clearDataflow()

        # notify process to shut down
        self.send_messages({"cmd": "exit"})

        log.info("stop subprocess %s" % (self.name,))
        self.process.join(timeout)
        if self.process.is_alive():
            log.info("terminate subprocess %s" % (self.name,))
            self.process.terminate()


    def is_alive(self):
        return self.process.is_alive()

    def get_messages(self, timeout=None):
        messages = []
        if not self.is_alive():
            log.error("process %s not alive" % self.name)
            return messages

        while self.parent_conn.poll(timeout):
            messages.append(self.parent_conn.recv())

        log.debug("received messages from %s: %s" % (self.name, messages))
        return messages

    def send_messages(self, messages):
        if not self.is_alive():
            log.error("process %s not alive" % self.name)
            return

        if not isinstance(messages, list):
            messages = [messages,]

        log.debug("send messages to %s: %s" % (self.name, messages))
        for msg in messages:
            self.parent_conn.send(msg)

    def loadDataflow(self, dfg_filename, ignored=None):
        self.send_messages({"cmd": "loadDataflow", "filename": dfg_filename})

    def startDataflow(self):
        self.send_messages({"cmd": "startDataflow"})

    def stopDataflow(self):
        self.send_messages({"cmd": "stopDataflow"})

    def clearDataflow(self):
        self.send_messages({"cmd": "clearDataflow"})

