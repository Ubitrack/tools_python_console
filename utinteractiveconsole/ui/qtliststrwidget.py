""" Enaml widget for editing a list of string
"""

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
from atom.api import (Bool, List, observe, set_default, Unicode, Enum, Int)

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_
from enaml.qt.QtGui import QListWidget, QListWidgetItem, QAbstractItemView
from enaml.qt.QtCore import *


class QtListStrWidget(RawWidget):
    """ A Qt4 implementation of an Enaml ProxyListStrView.

    """

    __slots__ = '__weakref__'

    #: The list of str being viewed
    items = d_(List(Unicode()))

    #: The index of the currently selected str
    selected_index = d_(Int(-1))
    
    #: The currently selected str
    selected_item = d_(Unicode())

    #: Whether or not the items should be read only
    read_only = d_(Bool(False))

    #: Whether or not the items should be checkable
    checkable = d_(Bool(True))

    #: Whether or not the items should be editable
    editable = d_(Bool(True))
    
    #: List of operations the user can perform
    # operations = d_(List(Enum( 'delete', 'insert', 'append', 'edit', 'move' ),
    #                    [ 'delete', 'insert', 'append', 'edit', 'move' ] ))

    #: .
    hug_width = set_default('weak')
    
    #--------------------------------------------------------------------------
    # Initialization API
    #--------------------------------------------------------------------------
    def create_widget(self, parent):
        """ Create the QListView widget.

        """
        # Create the list model and accompanying controls:
        widget = QListWidget(parent)
        for item in self.items:
            self.add_item(widget, item)

        if self.read_only:
            widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        else:
            widget.itemSelectionChanged.connect(self.on_selection)
            widget.itemChanged.connect(self.on_edit)

        # set selected_item here so that first change fires an 'update' rather than 'create' event
        self.selected_item = ''
        
        return widget

    def add_item(self, widget, item):
        itemWidget = QListWidgetItem(item)
        if self.checkable:
            itemWidget.setCheckState(Qt.Checked)
        if self.editable:
            _set_item_flag(itemWidget, Qt.ItemIsEditable, True)
        widget.addItem(itemWidget)

    #--------------------------------------------------------------------------
    # Signal Handlers
    #--------------------------------------------------------------------------
    def on_selection(self):
        """ The signal handler for the index changed signal.

        """
        widget = self.get_widget()
        self.selected_index = widget.currentRow()
        self.selected_item = self.items[widget.currentRow()]                   

    def on_edit(self, item):
        """ The signal handler for the item changed signal.
        """
        widget = self.get_widget()
        self.items[widget.currentRow()] = item.text()
        self.selected_item = item.text()

    #--------------------------------------------------------------------------
    # ProxyListStrView API
    #--------------------------------------------------------------------------

    def set_items(self, items, widget = None):
        """
        """
        widget = self.get_widget()
        widget.clear()
        for item in items:
            self.add_item(widget, item)

    def get_selected_items(self):
        widget = self.get_widget()
        items = []
        for idx in range(widget.count()):
            itemWidget = widget.item(idx)
            if itemWidget.checkState() == Qt.Checked:
                items.append(itemWidget.text())
        return items

    def set_checkstate(self, item, state):
        widget = self.get_widget()
        if item in self.items:
            idx = self.items.index(item)
            itemWidget = widget.item(idx)
            if state:
                itemWidget.setCheckState(Qt.Checked)
            else:
                itemWidget.setCheckState(Qt.Unchecked)



    #--------------------------------------------------------------------------
    # Observers
    #--------------------------------------------------------------------------
    # @observe('items', 'operations')
    @observe('items')
    def _update_proxy(self, change):
        """ An observer which sends state change to the proxy.

        """
        # The superclass handler implementation is sufficient.
        name = change['name']
        if self.get_widget() is not None:
            if name == 'items':
                self.set_items(self.items)       

# Helper methods
def _set_item_flag(item, flag, enabled):
    """ Set or unset the given item flag for the item.

    """
    flags = item.flags()
    if enabled:
        flags |= flag
    else:
        flags &= ~flag
    item.setFlags(flags)
