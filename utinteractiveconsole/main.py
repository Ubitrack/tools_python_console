__author__ = 'mvl'

# HACK TO WORKAROUND DLL IMPORT ERROR !!
# preload all ubitrack modules to prevent the use of enaml import hooks
from ubitrack.core import math, measurement, util
from ubitrack.facade import facade
from ubitrack.vision import vision
from ubitrack.visualization import visualization

del math
del measurement
del util
del facade
del vision
del visualization
# END HACK

import os, sys
import ConfigParser
from optparse import OptionParser
import logging
import warnings


# hack to register the opengl widget factory
from enaml.qt import qt_factories
def create_openglwidget():
    from enaml_opengl.qt.qt_opengl_widget import QtOpenGLWidget
    return QtOpenGLWidget

qt_factories.QT_FACTORIES['OpenGLWidget'] = create_openglwidget
# end hack

import enaml
from enaml.workbench.ui.api import UIWorkbench

from ubitrack.core import util
from utinteractiveconsole.app import AppState
from utinteractiveconsole.extension import WorkspaceExtensionManager
from utinteractiveconsole.guilogging import Syslog

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
#warnings.simplefilter("always")

import logging.config


def main():

    parser = OptionParser()

    parser.add_option("-l", "--logconfig",
                  action="store", dest="logconfig", default="/etc/mvl/log4cpp.conf",
                  help="log4cpp config file")

    parser.add_option("--py-logconfig",
                  action="store", dest="py_logconfig", default="",
                  help="python logging config file")

    # parser.add_option("-w", "--workspace",
    #               action="store", dest="autostart_workspace", default=None,
    #               help="Automatically select workspace on startup")
    #
    parser.add_option("-L", "--show-logwindow",
                  action="store_true", dest="show_logwindow", default=False,
                  help="Show logging window in gui")

    parser.add_option("-C", "--configfile",
                  action="append", dest="configfile", default=["~/utic.conf", ],
                  help="Interactive console config file")


    syslog = Syslog()

    appstate = AppState(context=dict(),
                        syslog=syslog)
    extensions = WorkspaceExtensionManager(appstate=appstate)

    extensions.updateCmdlineParser(parser)
    appstate.extensions = extensions
    appstate.context['extensions'] = extensions

    (options, args) = parser.parse_args()

    if options.py_logconfig != "" and os.path.isfile(options.py_logconfig):
        log.info("Reloading logging config from: %s" % options.py_logconfig)
        logging.config.fileConfig(options.py_logconfig, disable_existing_loggers=False)
        log.info("Done reloading config.")

    if options.show_logwindow:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(syslog.handler)

    appstate.args = args
    appstate.options = options

    appstate.context['args'] = args
    appstate.context['options'] = options

    log.info("Invocation args: %s" % " ".join(sys.argv))

    # XXX care about windows default paths here
    cfgfiles = []
    for cfgname in os.environ.get("UTIC_CONFIG_FILE", "/etc/mvl/utic.conf").split(os.pathsep):
        if os.path.isfile(cfgname):
            cfgfiles.append(cfgname)

    for cfgfname in options.configfile:
        if os.path.isfile(cfgfname):
            cfgfiles.append(cfgfname)
        else:
            log.warn("Skipping non-existant config file: %s" % cfgfname)

    config = ConfigParser.ConfigParser()
    try:
        log.info("Loading config files: %s" % (",".join(cfgfiles)))
        config.read(cfgfiles)
        appstate.context['config'] = config
    except Exception, e:
        log.error("Error parsing config file(s): %s" % (cfgfiles,))
        log.exception(e)


    if len(args) < 1:
        filename = None
    else:
        filename = args[0]
    appstate.context['filename'] = filename

    with enaml.imports():
        from utinteractiveconsole.ui.views.manifest import ApplicationManifest

    workbench = UIWorkbench()

    util.initLogging(options.logconfig)
    # XXX use weakref here !!
    appstate.context['appstate'] = appstate
    extensions.initExtensions()

    manifest = ApplicationManifest(appstate=appstate, extension_mgr=extensions)
    manifest.initialize()

    workbench.register(manifest)
    appstate.context['workbench'] = workbench

    workbench.run()


if __name__ == '__main__':
    main()
