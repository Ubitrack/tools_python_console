__author__ = 'jack'
import sys
from enaml.qt import QtCore, QtGui

import sys
from atom.api import Value, Dict, set_default

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager




class IPythonWidget(RawWidget):
    """ A Qt4 implementation of an Enaml IPythonWidget.

    """

    __slots__ = '__weakref__'


    #: the shell context
    context = d_(Dict())

    kernel_manager = d_(Value())
    kernel_client = d_(Value())
    kernel = d_(Value())

    #: .
    hug_width = set_default('weak')
    hug_height = set_default('weak')

    #--------------------------------------------------------------------------
    # Initialization API
    #--------------------------------------------------------------------------
    def create_widget(self, parent):
        """ Create the IPython Widget.

        """
        print "create ipy widget"
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push(self.context)

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
            QtGui.QApplication.quit()

        widget = RichIPythonWidget(parent=parent)
        widget.kernel_manager = self.kernel_manager
        widget.kernel_client = self.kernel_client
        widget.exit_requested.connect(stop)

        self.observe("context", self._handle_update)

        # XXX BAAAD
        widget.__class__.parent = lambda x: None

        return widget

    def _handle_update(self, change):
        print "update IPython context: ", str(change)
        self.kernel.shell.push(change["value"])