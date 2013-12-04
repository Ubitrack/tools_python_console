# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:27:48 2013

@author: silvester

Demonstration of Point Cloud tool using OpenGL bindings in PyQtGraph

Based on GLScatterPlotItem.py example from PyQtGraph
License: MIT

"""
from enaml.qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import pyqtgraph as pg
import sys
from atom.api import Atom, Float, Value, observe, Coerced, Int, ForwardTyped, Typed, List, Bool, set_default, Unicode, Enum
from enaml.layout.geometry import Size

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_



#: Cyclic guard flags
VIEW_SYNC_FLAG = 0x1
ITEM_CHANGE_FLAG = 0x2


class MyGLViewWidget(gl.GLViewWidget):
    """ Override GLViewWidget with enhanced behavior and Atom integration.

    """
    #: Fired in update() method to synchronize listeners.
    sigUpdate = QtCore.Signal()

    def mousePressEvent(self, ev):
        """ Store the position of the mouse press for later use.

        """
        super(MyGLViewWidget, self).mousePressEvent(ev)
        self._downpos = self.mousePos

    def mouseReleaseEvent(self, ev):
        """ Allow for single click to move and right click for context menu.

        Also emits a sigUpdate to refresh listeners.
        """
        super(MyGLViewWidget, self).mouseReleaseEvent(ev)
        if self._downpos == ev.pos():
            if ev.button() == 2:
                print 'show context menu'
            elif ev.button() == 1:
                x = ev.pos().x() - self.width() / 2
                y = ev.pos().y() - self.height() / 2
                self.pan(-x, -y, 0, relative=True)
                print self.opts['center']
        self._prev_zoom_pos = None
        self._prev_pan_pos = None
        self.sigUpdate.emit()

    def mouseMoveEvent(self, ev):
        """ Allow Shift to Move and Ctrl to Pan.

        """
        shift = ev.modifiers() & QtCore.Qt.ShiftModifier
        ctrl = ev.modifiers() & QtCore.Qt.ControlModifier
        if shift:
            y = ev.pos().y()
            if not hasattr(self, '_prev_zoom_pos') or not self._prev_zoom_pos:
                self._prev_zoom_pos = y
                return
            dy = y - self._prev_zoom_pos
            def delta():
                return -dy * 5
            ev.delta = delta
            self._prev_zoom_pos = y
            self.wheelEvent(ev)
        elif ctrl:
            pos = ev.pos().x(), ev.pos().y()
            if not hasattr(self, '_prev_pan_pos') or not self._prev_pan_pos:
                self._prev_pan_pos = pos
                return
            dx = pos[0] - self._prev_pan_pos[0]
            dy = pos[1] - self._prev_pan_pos[1]
            self.pan(dx, dy, 0, relative=True)
            self._prev_pan_pos = pos
        else:
            super(MyGLViewWidget, self).mouseMoveEvent(ev)










class QtGLViewWidget(RawWidget):
    """ A Qt4 implementation of an Enaml GLViewWidget.

    """

    __slots__ = '__weakref__'

    #: The scene that should be displayed
    scene = d_(ForwardTyped(lambda: Scene3D))

    #: Camera FOV center
    center = d_(Value(pg.Vector(0, 0, 0)))

    #: Distance of camera from center
    distance = d_(Float(1.))

    #: Horizontal field of view in degrees
    fov = d_(Float(60.))

    #: Camera's angle of elevation in degrees.
    elevation = d_(Coerced(float, (30.,)))

    #: Camera's azimuthal angle in degrees.
    azimuth = d_(Float(45.))

    #: Cyclic notification guard flags.
    _guard = d_(Int(0))


    #: .
    hug_width = set_default('weak')

    #--------------------------------------------------------------------------
    # Initialization API
    #--------------------------------------------------------------------------
    def create_widget(self, parent):
        """ Create the QListView widget.

        """
        print "create glwidget"
        widget = MyGLViewWidget(parent)

        widget.sigUpdate.connect(self._update_model)
        for attr in ['azimuth', 'distance', 'fov', 'center', 'elevation']:
            widget.opts[attr] = getattr(self, attr)

        return widget


    def _update_model(self):
        """ Synchronize view attributes to the model.

        """
        if self._guard & ITEM_CHANGE_FLAG:
            return
        self._guard &= VIEW_SYNC_FLAG
        widget = self.get_widget()
        for (key, value) in widget.opts.items():
            setattr(self, key, value)
        self._guard &= ~VIEW_SYNC_FLAG

    #--------------------------------------------------------------------------
    # Signal Handlers
    #--------------------------------------------------------------------------
    def on_update_items(self):
        """ The signal handler for the index changed signal.

        """
        self.set_items(self.scene.items)
        self.on_update()

    def on_update(self):
        """ The signal handler for the index changed signal.

        """
        widget = self.get_widget()
        widget.updateGL()

    #--------------------------------------------------------------------------
    # ProxyListStrView API
    #--------------------------------------------------------------------------

    def set_items(self, items):
        """
        """
        if self._guard & VIEW_SYNC_FLAG:
            return
        self._guard &= ITEM_CHANGE_FLAG
        widget = self.get_widget()
        for item in widget.items:
            widget.removeItem(item)

        if self.scene._grid:
            self.add_item(self.scene._grid)
        if self.scene._orientation_axis:
            self.add_item(self.scene._orientation_axis)

        for item in items:
            widget.addItem(item._item)

        self._guard &= ~ITEM_CHANGE_FLAG


    #--------------------------------------------------------------------------
    # Observers
    #--------------------------------------------------------------------------

    @observe('azimuth', 'distance', 'fov', 'center', 'elevation')
    def _update_view(self, change):
        """ Synchronize model attributes to the view.

        """
        if self._guard & (VIEW_SYNC_FLAG) or change['type'] == 'create':
            return
        widget = self.get_widget()
        widget.opts[change['name']] = change['value']
        widget.update()


    @observe('scene')
    def _update_scene(self, change):
        """ An observer which sends state change to the proxy.

        """
        widget = self.get_widget()
        if change['type'] == 'create':
            return
        self._guard |= ITEM_CHANGE_FLAG
        for item in change['oldvalue']:
            widget.removeItem(item._item)

        for item in change['value']:
            widget.addItem(item._item)
        self._guard &= ~ITEM_CHANGE_FLAG




class GLAxisItem(Atom):
    """ An Axis Item Manager.

    Shows a coordinate root.

    """
    #: (4,4) array of floats specifying a transform.
    transform = Coerced(np.ndarray, coercer=np.ndarray)

     #: GLAxisItem instance.
    _item = Value()

    def _default_transform(self):
        return np.eye(4)

    def _default__item(self):
        """ Create a GLAxisItem item with our current attributes.

        """
        print "create axisitem"
        return gl.GLAxisItem()

    @observe('transform')
    def _item_change(self, change):
        """ Pass changes to point properties to the GLScatterPlot object.

        """
        if change['name'] == 'transform':
            self._item.resetTransform()
            self._item.applyTransform(QtGui.QMatrix4x4(change['value'].flatten()), False)


class Scene3D(Atom):
    """ A 3D Scene Manager.

    Maintains a gl items and its scene.

    """
    __slots__ = '__weakref__'

    #: GLGraphicsItem manager
    items = List(Typed(GLAxisItem, ()))

    #: GLGridItem instance
    _grid = Value()

    #: GLAxisItem instance.
    _orientation_axes = Value()

    #: Cyclic notification guard flags.
    _guard = Int(0)

    def _default__grid(self, parent=None):
        print "create grid"
        return gl.GLGridItem()

    def _default__orientation_axes(self, parent=None):
        print "create origin"
        return gl.GLAxisItem()
















