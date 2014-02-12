# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:27:48 2013

@author: silvester

Demonstration of Point Cloud tool using OpenGL bindings in PyQtGraph

Based on GLScatterPlotItem.py example from PyQtGraph
License: MIT

"""
import sys
from math import pi, radians
from OpenGL.GL import *
from enaml.qt import QtCore, QtGui

if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
    from pyqtgraph import setConfigOptions
    setConfigOptions(useOpenGL=True)

import pyqtgraph.opengl as gl
import numpy as np
import pyqtgraph as pg
import sys
from atom.api import Atom, Float, Value, observe, Coerced, Int, ForwardTyped, Typed, List, Bool, set_default, Unicode, Enum
from enaml.layout.geometry import Size

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_

from OpenGL import GLU
from ubitrack.core import util, math, measurement

import logging
log = logging.getLogger(__name__)

#: Cyclic guard flags
VIEW_SYNC_FLAG = 0x1
ITEM_CHANGE_FLAG = 0x2

# from openglcontext (mcfletcher)
class DragWatcher(object):
    """Class providing semantics for fractional and absolute dragging

    With this class you can track the start position of a drag action
    and query for both absolute distance dragged, and distance as a
    fraction of the distance to the edges of the window.
    """
    def __init__(self, startX, startY, totalX, totalY ):
        """Initialise the DragWatcher

        startX, startY -- initial coordinates for the drag
        totalX, totalY -- overall dimensions of the context
        """
        self.start = startX, startY
        self.total = totalX, totalY
    def fractions (self, newX, newY):
        """Calculate fractional delta from the start point

        newX, newY -- new selection point from which to calculate
        """
        if (newX, newY) == self.start:
            return 0.0, 0.0
        values = []
        for index, item in ((0, newX), (1, newY)):
            if item < self.start[index]:
                value = float(item-self.start[index]) / self.start[index]
            else:
                value = float(item-self.start[index]) / (self.total[index]-self.start[index])
            values.append (value)
        return values
    def distances (self, newX, newY ):
        """Calculate absolute distances from start point

        newX, newY -- new selection point from which to calculate
        """
        if (newX, newY) == self.start:
            return 0, 0
        else:
            return newX-self.start[0], newY-self.start[1]

class Trackball:
    '''Trackball mechanism for interactive rotation

    Use the trackball utility to rotate a viewpoint
    around a fixed world-space coordinate (center).

    This trackball is a simple x/y grid of polar
    coordinates.  Dragging to the left rotates the
    eye around the object to view the left side,
    similarly for right, top, bottom.
    '''
    def __init__ (
        self, position, quaternion,
        center,
        originalX, originalY, width, height,
        dragAngle=pi/1.5,

    ):
        """Initialise the Trackball

        position -- object-space original position (camera pos)

        quaternion -- camera orientation as a quaternion

        originalX, originalY -- the initial screen
            coordinates of the drag

        width, height -- the dimensions of the screen
            (newX-originalX)/(fractional width) used by
            trackball algorithm

        center -- the x,y,z world coordinates around which
            we are to rotate the application will need to
            use some heuristic to determine the most appropriate
            center of rotation.  For instance, when the user
            first clicks, check for an object in the "center" of
            the display, use the center of that object (or
            possibly the midpoint between the greatest and least
            Z-buffer values) projected back into world space
            coordinates.  If there is no available object,
            potentially use the maximum and minimum of the whole
            Z buffer. If there are no rendered elements at all
            then use some multiple of the near frustum (20 or
            30, for example)
        dragAngle -- maximum rotation angle for a drag
        """
        self.watcher = DragWatcher(originalX, originalY, width, height)
        self.originalPosition = position
        self.originalQuaternion = quaternion
        self.xAxis = self.originalQuaternion.transformVector(np.asarray([1.0, 0.0, 0.0]))
        self.yAxis = self.originalQuaternion.transformVector(np.asarray([0.0, 1.0, 0.0]))
        self.zAxis = self.originalQuaternion.transformVector(np.asarray([0.0, 0.0, 1.0]))

        self.center = np.asarray(center[:3])
        self.dragAngle = dragAngle
        self.vector = self.originalPosition - self.center

    def cancel (self):
        """Cancel drag rotation, return pos,quat to original values"""
        return self.originalPosition, self.originalQuaternion

    def update(self, newX, newY , alt=False):
        """Update with new x,y drag coordinates

        newX, newY -- the new screen coordinates for the drag

        returns a new position and quaternion orientation
        """
        # get the drag fractions
        x, y = self.watcher.fractions(newX, newY)
        # multiply by the maximum drag angle
        # note that movement in x creates rotation about y & vice-versa
        # note that OpenGL coordinates make y reversed from "normal" rotation
        yRotation, xRotation = x * self.dragAngle, -y * self.dragAngle
        # calculate the results, keeping in mind that translation in one axis is rotation around the other
        if alt:
            xRot = math.Quaternion(self.xAxis, xRotation)
        else:
            xRot = math.Quaternion(self.zAxis, xRotation)

        yRot = math.Quaternion(self.yAxis, yRotation)


        # the vector is already rotated by originalQuaternion
        # and positioned at the origin, so just needs
        # the adjusted x + y rotations + un-positioning
        pos = (math.Quaternion(xRot * yRot).transformVector(self.vector)) + self.center
        rot = math.Quaternion(self.originalQuaternion * xRot * yRot)
        return math.Pose(rot, pos)

# end of openglcontext reuse



class MyGLViewWidget(gl.GLViewWidget):
    """ Override GLViewWidget with enhanced behavior and Atom integration.

    """
    #: Fired in update() method to synchronize listeners.
    sigUpdate = QtCore.Signal()

    def __init__(self, parent=None):
        super(MyGLViewWidget, self).__init__(parent=parent)
        # should be computed based on initial elevation and azimuth
        rot = math.Quaternion( math.Quaternion(np.asarray([1.0, 0.0, 0.0]), 0.0) *
                               math.Quaternion(np.asarray([1.0, 0.0, 0.0]), 0.0))
        self.opts["camera_pose"] = math.Pose(rot, np.asarray([0.0, 0.0, 0.5]))


    def viewMatrix(self):
        return QtGui.QMatrix4x4(self.opts["camera_pose"].invert().toMatrix().flatten())


    def mousePressEvent(self, ev):
        """ Store the position of the mouse press for later use.

        """
        self.mousePos = ev.pos()
        self._downpos = self.mousePos
        if ev.buttons() == QtCore.Qt.LeftButton:
            center = self.opts['camera_pose'].translation()
            position = self.opts['camera_pose'].translation()
            orientation = self.opts['camera_pose'].rotation()
            self._trackball = Trackball(position, orientation, center,
                                        ev.pos().x(), ev.pos().y(),
                                        self.width(), self.height())

    def _drag(self, dx, dy, dz):
        pose = self.opts['camera_pose']
        new_pose = pose * math.Pose(math.Quaternion(), np.asarray([float(-dx)/50.0, float(-dy)/50.0, float(-dz)/50.0]))
        self.opts['camera_pose'] = new_pose
        self.update()


    def mouseMoveEvent(self, ev):
        """ Allow Shift to Move and Ctrl to Pan.

        """
        shift = ev.modifiers() & QtCore.Qt.ShiftModifier
        ctrl = ev.modifiers() & QtCore.Qt.ControlModifier
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()

        if ev.buttons() == QtCore.Qt.LeftButton:
            if self._trackball is not None:
                self.opts["camera_pose"] = self._trackball.update(ev.pos().x(), ev.pos().y(), shift)
                self.update()
        elif ev.buttons() == QtCore.Qt.RightButton:
            if shift:
                self._drag(diff.x(), 0, diff.y())
            else:
                self._drag(diff.x(), diff.y(), 0)



    def mouseReleaseEvent(self, ev):
        """ Allow for single click to move and right click for context menu.

        Also emits a sigUpdate to refresh listeners.
        """
        self._trackball = None
        self.sigUpdate.emit()

    def wheelEvent(self, ev):
        print "WE ", ev
        if (ev.modifiers() & QtCore.Qt.ControlModifier):
            self.opts['fov'] *= 0.999**ev.delta()
        else:
            pose = self.opts['camera_pose']
            new_pose = pose * math.Pose(math.Quaternion(), np.asarray([0.0, 0.0, ev.delta()/300.0]))
            self.opts['camera_pose'] = new_pose
        self.update()









class MyGLAxisItem(gl.GLGraphicsItem.GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`

    Displays three lines indicating origin and orientation of local coordinate system.

    """

    def __init__(self, size=None, linewidth=None, colors=None, antialias=True, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        if size is None:
            size = QtGui.QVector3D(1, 1, 1)
        if linewidth is None:
            linewidth = 1.0
        if colors is None:
            colors = [(1, 0, 0, 0.6),  # x red
                      (0, 1, 0, 0.6),  # y green
                      (0, 0, 1, 0.6),  # z blue
            ]
        self.antialias = antialias
        self.setSize(size=size)
        self.setLinewidth(linewidth)
        self.setColors(colors)
        self.setGLOptions(glOptions)

    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x, y, z]
        self.update()

    def size(self):
        return self.__size[:]


    def setLinewidth(self, linewidth):
        """
        Set the linewidth of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        self.__linewidth = linewidth
        self.update()

    def linewidth(self):
        return self.__linewidth


    def setColors(self, colors):
        """
        Set the colors of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        self.__colors = colors
        self.update()

    def colors(self):
        return self.__colors[:]


    def paint(self):

        try:
            #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            #glEnable( GL_BLEND )
            #glEnable( GL_ALPHA_TEST )
            self.setupGLState()

            glPushAttrib(GL_LINE_BIT)

            glLineWidth(self.linewidth())

            colors = self.colors()

            if self.antialias:
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

            glBegin(GL_LINES)
            x, y, z = self.size()

            glColor4f(*colors[2])  # z is green
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, z)

            glColor4f(*colors[1])  # y is yellow
            glVertex3f(0, 0, 0)
            glVertex3f(0, y, 0)

            glColor4f(*colors[0])  # x is blue
            glVertex3f(0, 0, 0)
            glVertex3f(x, 0, 0)

            glEnd()

            glPopAttrib()
        except Exception, e:
            log.exception(e)



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
    hug_height = set_default('weak')

    #--------------------------------------------------------------------------
    # Initialization API
    #--------------------------------------------------------------------------
    def create_widget(self, parent):
        """ Create the QListView widget.

        """
        widget = MyGLViewWidget(parent)

        widget.sigUpdate.connect(self._update_model)

        for attr in ['azimuth', 'distance', 'fov', 'center', 'elevation']:
            widget.opts[attr] = getattr(self, attr)

        if self.scene.grid:
            widget.addItem(self.scene.grid.item)
        if self.scene.orientation_axes:
            widget.addItem(self.scene.orientation_axes.item)

        for item in self.scene.items:
            widget.addItem(item.item)

        def _handle_update(change):
            if change['type'] == 'create':
                return
            self.on_update()

        self.scene.observe("needs_update", _handle_update)

        return widget


    def _update_model(self):
        """ Synchronize view attributes to the model.

        """
        if self._guard & ITEM_CHANGE_FLAG:
            return
        self._guard &= VIEW_SYNC_FLAG
        widget = self.get_widget()
        for (key, value) in widget.opts.items():
            if not key in ['azimuth', 'distance', 'fov', 'center', 'elevation']:
                continue
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

        if self.scene.grid:
            widget.addItem(self.scene.grid.item)
        if self.scene.orientation_axes:
            widget.addItem(self.scene.orientation_axes.item)

        for item in items:
            widget.addItem(item.item)

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
        if change['type'] == 'create':
            return
        self._guard |= ITEM_CHANGE_FLAG
        self.on_update_items()
        self._guard &= ~ITEM_CHANGE_FLAG


class GLGraphicsItem(Atom):
    """ A Gereric GraphicsItem Manager.

    """
    #: (4,4) array of floats specifying a transform.
    transform = Coerced(np.ndarray, coercer=np.ndarray)

     #: GLAxisItem instance.
    item = Value()

    visible = Bool(True)

    def _default_transform(self):
        return np.eye(4)


    @observe('transform', 'visible')
    def _item_change(self, change):
        """ Pass changes to point properties to the GLGraphicsItem object.

        """
        if change['name'] == 'transform':
            self.item.resetTransform()
            self.item.applyTransform(QtGui.QMatrix4x4(change['value'].flatten()), False)
        elif change['name'] == 'visible':
            if change['value']:
                self.item.show()
            else:
                self.item.hide()


class GLAxisItem(GLGraphicsItem):
    """ An Axis Item Manager.

    Shows a coordinate root.

    x=blue
    y=yellow
    z=green

    """

    size = Value((1.0, 1.0, 1.0))
    linewidth = Value(1.0)
    colors = Value([(1, 0, 0, 0.6),  # x red
                   (0, 1, 0, 0.6),  # y green
                   (0, 0, 1, 0.6),  # z blue
                    ])


    def _default_item(self):
        """ Create an item with our current attributes.

        """
        return MyGLAxisItem(size=QtGui.QVector3D(*self.size), linewidth=self.linewidth, colors=self.colors)

    @observe('size', 'linewidth', 'colors')
    def _data_change(self, change):
        """ Pass changes to point properties to the object.

        """
        if change['name'] == 'size':
            self.item.setSize(size=QtGui.QVector3D(*change["value"]))
        elif change['name'] == 'linewidth':
            self.item.setLinewidth(change["value"])
        elif change['name'] == 'colors':
            self.item.setColors(change["value"])


class GLGridItem(GLGraphicsItem):
    """ An Grid Item Manager.

    Shows a coordinate root.

    """

    size = Value((1.0, 1.0, 1.0))


    def _default_item(self):
        """ Create a GLGridItem item with our current attributes.

        """
        return gl.GLGridItem(size=QtGui.QVector3D(*self.size))

    @observe('size')
    def _data_change(self, change):
        """ Pass changes to point properties to the GlGridItem object.

        """
        if change['name'] == 'size':
            self.item.setSize(size=QtGui.QVector3D(*change["value"]))



class GLLinePlotItem(GLGraphicsItem):
    """ An Grid Item Manager.

    Shows a line plot.

    """
    #: (N,3) array of floats specifying line point locations.
    pos = Coerced(np.ndarray, coercer=np.ndarray)

    #: (N,4) array of floats (0.0-1.0) specifying pot colors
    #: OR a tuple of floats specifying a single color for all spots.
    color = Value([1.0, 1.0, 1.0, 0.5])

    linewidth = Value(1.0)

    def _default_item(self):
        """ Create a GlLinePlotItem item with our current attributes.

        """
        return gl.GLLinePlotItem(pos=self.pos, color=self.color, width=self.linewidth)

    @observe('pos')
    def _data_change(self, change):
        """ Pass changes to point properties to the GlLinePlotItem object.

        """
        if change['name'] == 'pos':
            self.item.setData(pos=self.pos)


class GLScatterPlotItem(GLGraphicsItem):
    """ An ScatterPlot Item Manager.

    Shows a scatter plot.

    """

    #: (N,3) array of floats specifying point locations.
    pos = Coerced(np.ndarray, coercer=np.ndarray)

    #: (N,4) array of floats (0.0-1.0) specifying pot colors
    #: OR a tuple of floats specifying a single color for all spots.
    color = Value([1.0, 1.0, 1.0, 0.5])

    #: (N,) array of floats specifying spot sizes or a value for all spots.
    size = Value(5)


    def _default_item(self):
        """ Create a GLAxisItem item with our current attributes.

        """
        return gl.GLScatterPlotItem(pos=self.pos, color=self.color,
                                    size=self.size)

    @observe('color', 'pos', 'size')
    def _data_change(self, change):
        """ Pass changes to point properties to the GLScatterPlot object.

        """
        kwargs = {change['name']: change['value']}
        self.item.setData(**kwargs)


class Scene3D(Atom):
    """ A 3D Scene Manager.

    Maintains a gl items and its scene.

    """
    __slots__ = '__weakref__'

    #: GLGraphicsItem manager
    items = List(Typed(GLGraphicsItem, ()))

    #: GLGridItem instance
    grid = Value()

    #: GLAxisItem instance.
    orientation_axes = Value()

    needs_update = d_(Bool(False))

    #: Cyclic notification guard flags.
    _guard = Int(0)

    def _default_grid(self, parent=None):
        return GLGridItem()

    def _default_orientation_axes(self, parent=None):
        return GLAxisItem()
















