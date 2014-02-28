""" Enaml widget for editing a list of string
"""

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
from atom.api import (observe, set_default, Int, Long, Value, ForwardTyped, Event)

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_
from enaml.qt.QtCore import *

from OpenGL.GL import *
import OpenGL.GL.framebufferobjects as glfbo
from enaml.qt import QtOpenGL, QtCore
import numpy as np
import pyqtgraph.functions as fn

from ubitrack.core import calibration
from ubitrack.visualization import visualization
from utinteractiveconsole.ui.pyqtgraphw import Scene3D, ITEM_CHANGE_FLAG, VIEW_SYNC_FLAG

import logging
log = logging.getLogger(__name__)




class QtVirtualCameraWidget(QtOpenGL.QGLWidget):

    ShareWidget = None

    #: Fired in update() method to synchronize listeners.
    sigUpdate = QtCore.Signal()


    def __init__(self, key_event_handler=None, mouse_event_handler=None,
                 cam_width=1024, cam_height=768, cam_near=0.01, cam_far=10.0,
                 camera_intrinsics=None, parent=None):

        if QtVirtualCameraWidget.ShareWidget is None:
            ## create a dummy widget to allow sharing objects (textures, shaders, etc) between views
            QtVirtualCameraWidget.ShareWidget = QtOpenGL.QGLWidget()

        QtOpenGL.QGLWidget.__init__(self, parent, QtVirtualCameraWidget.ShareWidget)

        self.setFocusPolicy(QtCore.Qt.ClickFocus)


        self.bgtexture = visualization.BackgroundImage()
        self.camera_intrinsics = camera_intrinsics
        self.camera_pose = None

        self.camera_width = float(cam_width)
        self.camera_height = float(cam_height)
        self.camera_near = float(cam_near)
        self.camera_far = float(cam_far)

        self.screen_width = cam_width
        self.screen_height = cam_height

        self.items = []

        self.key_event_handler = key_event_handler
        self.mouse_event_handler = mouse_event_handler
        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        self.keysPressed = {}
        self.keyTimer = QtCore.QTimer()
        self.keyTimer.timeout.connect(self.evalKeyState)

        self.makeCurrent()

    def setBackgroundTexture(self, m):
        self.bgtexture.imageIn(m)

    def setCameraIntrinsics(self, mat):
        self.camera_intrinsics = mat

    def setCameraPose(self, mat):
        self.camera_pose = mat

    def addItem(self, item):
        self.items.append(item)
        if hasattr(item, 'initializeGL'):
            self.makeCurrent()
            try:
                item.initializeGL()
            except:
                self.checkOpenGLVersion('Error while adding item %s to GLViewWidget.' % str(item))

        item._setView(self)
        self.update()

    def removeItem(self, item):
        self.items.remove(item)
        item._setView(None)
        self.update()


    def setItems(self, items):
        for item in self.items:
            self.removeItem(item)

        for item in items:
            self.addItem(item)


    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        self.resizeGL(self.width(), self.height())

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self.screen_width = w
        self.screen_height = h
        #self.update()



    def setProjection(self):
        ## Create the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.camera_intrinsics is not None:
            proj = calibration.projectionMatrix3x3ToOpenGL(0., self.camera_width, 0., self.camera_height, self.camera_near, self.camera_far, self.camera_intrinsics)
            glMultMatrixd(proj.T)


    def setModelview(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # XXX Needs more thoughts .. a transpose could be required here to be consistent with the global matrix handling
        # if self.camera_pose is not None:
        #     pose = self.camera_pose.toMatrix()
        #     glMultMatrixd(pose)


    def paintGL(self):
        self.setProjection()
        self.setModelview()
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )

        if self.bgtexture is not None:
            self.bgtexture.draw(self.screen_width, self.screen_height)

        self.drawItemTree()


    def drawItemTree(self, item=None):
        if item is None:
            items = [x for x in self.items if x.parentItem() is None]
        else:
            items = item.childItems()
            items.append(item)
        items.sort(lambda a,b: cmp(a.depthValue(), b.depthValue()))
        for i in items:
            if not i.visible():
                continue
            if i is item:
                try:
                    glPushAttrib(GL_ALL_ATTRIB_BITS)
                    i.paint()
                except:
                    import pyqtgraph.debug
                    pyqtgraph.debug.printExc()
                    msg = "Error while drawing item %s." % str(item)
                    ver = glGetString(GL_VERSION)
                    if ver is not None:
                        ver = ver.split()[0]
                        if int(ver.split('.')[0]) < 2:
                            print(msg + " The original exception is printed above; however, pyqtgraph requires OpenGL version 2.0 or greater for many of its 3D features and your OpenGL version is %s. Installing updated display drivers may resolve this issue." % ver)
                        else:
                            print(msg)

                finally:
                    glPopAttrib()
            else:
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                try:
                    tr = i.transform()
                    a = np.array(tr.copyDataTo()).reshape((4,4))
                    glMultMatrixf(a.transpose())
                    self.drawItemTree(i)
                finally:
                    glMatrixMode(GL_MODELVIEW)
                    glPopMatrix()


    def mousePressEvent(self, ev):
        self.mousePos = ev.pos()
        if self.mouse_event_handler:
            self.mouse_event_handler(("press", ev))

    def mouseMoveEvent(self, ev):
        if self.mouse_event_handler:
            self.mouse_event_handler(("move", ev))

    def mouseReleaseEvent(self, ev):
        if self.mouse_event_handler:
            self.mouse_event_handler(("release", ev))

    def wheelEvent(self, ev):
        if self.mouse_event_handler:
            self.mouse_event_handler(("wheel", ev))

    def keyPressEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()

        if self.key_event_handler:
            self.key_event_handler(("press", ev, self.keysPressed))


    def keyReleaseEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()

        if self.key_event_handler:
            self.key_event_handler(("release", ev, self.keysPressed))

    def checkOpenGLVersion(self, msg):
        ## Only to be called from within exception handler.
        ver = glGetString(GL_VERSION).split()[0]
        if int(ver.split('.')[0]) < 2:
            import pyqtgraph.debug
            pyqtgraph.debug.printExc()
            raise Exception(msg + " The original exception is printed above; however, utInteractiveConsole requires OpenGL version 2.0 or greater for many of its 3D features and your OpenGL version is %s. Installing updated display drivers may resolve this issue." % ver)
        else:
            raise


    def evalKeyState(self):
        speed = 2.0
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key_Right:
                    pass
                elif key == QtCore.Qt.Key_Left:
                    pass
                elif key == QtCore.Qt.Key_Up:
                    pass
                elif key == QtCore.Qt.Key_Down:
                    pass
                elif key == QtCore.Qt.Key_PageUp:
                    pass
                elif key == QtCore.Qt.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()



    def readQImage(self):
        """
        Read the current buffer pixels out as a QImage.
        """
        w = self.width()
        h = self.height()
        self.repaint()
        pixels = np.empty((h, w, 4), dtype=np.ubyte)
        pixels[:] = 128
        pixels[...,0] = 50
        pixels[...,3] = 255

        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels)

        # swap B,R channels for Qt
        tmp = pixels[..., 0].copy()
        pixels[..., 0] = pixels[..., 2]
        pixels[..., 2] = tmp
        pixels = pixels[::-1]  # flip vertical

        img = fn.makeQImage(pixels, transpose=False)
        return img


    def renderToArray(self, size, format=GL_BGRA, type=GL_UNSIGNED_BYTE, textureSize=1024, padding=256):
        w,h = map(int, size)

        self.makeCurrent()
        tex = None
        fb = None
        try:
            output = np.empty((w, h, 4), dtype=np.ubyte)
            fb = glfbo.glGenFramebuffers(1)
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, fb )

            glEnable(GL_TEXTURE_2D)
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            texwidth = textureSize
            data = np.zeros((texwidth,texwidth,4), dtype=np.ubyte)

            ## Test texture dimensions first
            glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
                raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % shape[:2])
            ## create teture
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.transpose((1,0,2)))

            self.opts['viewport'] = (0, 0, w, h)  # viewport is the complete image; this ensures that paintGL(region=...)
                                                  # is interpreted correctly.
            p2 = 2 * padding
            for x in range(-padding, w-padding, texwidth-p2):
                for y in range(-padding, h-padding, texwidth-p2):
                    x2 = min(x+texwidth, w+padding)
                    y2 = min(y+texwidth, h+padding)
                    w2 = x2-x
                    h2 = y2-y

                    ## render to texture
                    glfbo.glFramebufferTexture2D(glfbo.GL_FRAMEBUFFER, glfbo.GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

                    self.paintGL(region=(x, h-y-h2, w2, h2), viewport=(0, 0, w2, h2))  # only render sub-region

                    ## read texture back to array
                    data = glGetTexImage(GL_TEXTURE_2D, 0, format, type)
                    data = np.fromstring(data, dtype=np.ubyte).reshape(texwidth,texwidth,4).transpose(1,0,2)[:, ::-1]
                    output[x+padding:x2-padding, y+padding:y2-padding] = data[padding:w2-padding, -(h2-padding):-padding]

        finally:
            self.opts['viewport'] = None
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            if tex is not None:
                glDeleteTextures([tex])
            if fb is not None:
                glfbo.glDeleteFramebuffers([fb])

        return output










class VirtualCameraWidget(RawWidget):
    """ A Qt4 implementation of an Enaml ProxyVirtualCameraWidget.

    """

    __slots__ = '__weakref__'

    camera_width = d_(Int(1024))
    camera_height = d_(Int(768))

    #: The scene that should be displayed
    scene = d_(ForwardTyped(lambda: Scene3D))

    #: The camera image measurement
    background_texture = d_(Value())
    
    #: The camera intrinsics measurement
    camera_intrinsics = d_(Value())

    #: The camera pose measurement
    camera_pose = d_(Value())

    #: The current timestamp
    current_timestamp = d_(Long())

    #: Cyclic notification guard flags.
    _guard = d_(Int(0))

    key_events = d_(Event())

    #: .
    hug_width = set_default('weak')
    hug_height = set_default('weak')

    #--------------------------------------------------------------------------
    # Initialization API
    #--------------------------------------------------------------------------
    def create_widget(self, parent):
        """ Create the QListView widget.

        """
        # Create the list model and accompanying controls:
        widget = QtVirtualCameraWidget(parent=parent,
                                       key_event_handler=self.key_events,
                                       cam_width=self.camera_width,
                                       cam_height=self.camera_height,
                                       # add properties for camera near, far, ...
                                       )

        widget.sigUpdate.connect(self._update_model)

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
        # for (key, value) in widget.opts.items():
        #     if not key in ['azimuth', 'distance', 'fov', 'center', 'elevation']:
        #         continue
        #     setattr(self, key, value)
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


    def readQImage(self):
        widget = self.get_widget()
        if widget is not None:
            return widget.readQImage()

    def renderToArray(self, size, format=GL_BGRA, type=GL_UNSIGNED_BYTE, textureSize=1024, padding=256):
        widget = self.get_widget()
        if widget is not None:
            return widget.renderToArray(size, format=format, type=type, textureSize=textureSize, padding=padding)


    #--------------------------------------------------------------------------
    # Observers
    #--------------------------------------------------------------------------


    @observe('scene')
    def _update_scene(self, change):
        """ An observer which sends state change to the proxy.

        """
        if change['type'] == 'create':
            return
        self._guard |= ITEM_CHANGE_FLAG
        self.on_update_items()
        self._guard &= ~ITEM_CHANGE_FLAG

    @observe('current_timestamp',)
    def _update_gl(self, change):
        widget = self.get_widget()
        if widget:
            if change["type"] == "update":
                if self.background_texture is not None:
                    widget.setBackgroundTexture(self.background_texture)
                if self.camera_intrinsics is not None:
                    widget.setCameraIntrinsics(self.camera_intrinsics)
                if self.camera_pose is not None:
                    widget.setCameraPose(self.camera_pose)



    @observe('camera_width', 'camera_height')
    def _update_camera_parameters(self, change):
        widget = self.get_widget()
        if widget:
            if change["type"] == "update":
                if change["name"] == "camera_width":
                    print self.camera_width, type(self.camera_width)
                    widget.camera_width = self.camera_width
                elif change["name"] == "camera_height":
                    print self.camera_height, type(self.camera_height)
                    widget.camera_height = self.camera_height

