__author__ = 'mvl'
import os, sys
import ConfigParser
from optparse import OptionParser
import logging

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

    parser.add_option("-L", "--skip-logwindow",
                  action="store_false", dest="show_logwindow", default=True,
                  help="Show logging window in gui")

    parser.add_option("-C", "--configfile",
                  action="store", dest="configfile", default="~/utic.conf",
                  help="Interactive console config file")


    syslog = Syslog()

    appstate = AppState(context=dict(),
                        syslog=syslog)
    extensions = ExtensionManager(appstate=appstate)

    extensions.updateCmdlineParser(parser)
    appstate.extensions=extensions
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
        from utinteractiveconsole.ui.manifest import ApplicationManifest

    workbench = UIWorkbench()

    util.initLogging(options.logconfig)
    extensions.initExtensions()
    # XXX use weakref here !!
    appstate.context['appstate'] = appstate

    manifest = ApplicationManifest(appstate=appstate, extension_mgr=extensions)
    manifest.initialize()

    workbench.register(manifest)
    workbench.run()


if __name__ == '__main__':
    main()
