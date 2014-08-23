__author__ = 'mvl'
import os, sys
import ConfigParser
from optparse import OptionParser
import logging


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
from utinteractiveconsole.app import AppState, ExtensionManager
from utinteractiveconsole.guilogging import Syslog

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)





def main():

    parser = OptionParser()

    parser.add_option("-l", "--logconfig",
                  action="store", dest="logconfig", default="/etc/mvl/log4cpp.conf",
                  help="log4cpp config file")

    # parser.add_option("-w", "--workspace",
    #               action="store", dest="autostart_workspace", default=None,
    #               help="Automatically select workspace on startup")
    #
    parser.add_option("-L", "--show-logwindow",
                  action="store_true", dest="show_logwindow", default=False,
                  help="Show logging window in gui")

    parser.add_option("-C", "--configfile",
                  action="store", dest="configfile", default="~/utic.conf",
                  help="Interactive console config file")


    syslog = Syslog()

    appstate = AppState(context=dict(),
                        syslog=syslog)
    extensions = ExtensionManager(appstate=appstate)

    extensions.updateCmdlineParser(parser)
    appstate.extensions = extensions
    appstate.context['extensions'] = extensions

    (options, args) = parser.parse_args()

    if options.show_logwindow:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(syslog.handler)

    appstate.args = args
    appstate.options = options

    appstate.context['args'] = args
    appstate.context['options'] = options

    # XXX care about windows default paths here
    cfgfiles = []
    if (os.path.isfile(os.environ.get("UTIC_CONFIG_FILE", "/etc/mvl/utic.conf"))):
        cfgfiles.append(os.environ["UTIC_CONFIG_FILE"])

    if (os.path.isfile(options.configfile)):
        cfgfiles.append(options.configfile)

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
