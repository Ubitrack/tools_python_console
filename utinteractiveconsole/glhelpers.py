__author__ = 'jack'

from OpenGL.GL import *
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui, QtOpenGL
from pyqtgraph import Vector
import numpy as np

from ubitrack.core import math, calibration
from ubitrack.vision import vision
from ubitrack.visualization import visualization

from utinteractiveconsole import uthelpers

__all__ = ['GLBackgroundImage']

class GLBackgroundImage(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`

    Displays image data as a textured quad.
    """


    def __init__(self, bgimage):
        """
        """
        self.backgroundimage = bgimage
        GLGraphicsItem.__init__(self)

    def initializeGL(self):
        pass


    def paint(self):
        view = self.view()
        if self.backgroundimage is not None and view is not None:
            self.backgroundimage.draw(view.width(), view.height())







class VirtualCameraWidget(QtOpenGL.QGLWidget):
    """
example:

from utinteractiveconsole.uthelpers import PushSinkAdapter
sink = df.getApplicationPushSinkVisionImage("pattern_40")
psa = PushSinkAdapter(sink)
from utinteractiveconsole.glhelpers import VirtualCameraWidget
vcw = VirtualCameraWidget(psa)
vcw.show()



    """

    ShareWidget = None

    def __init__(self, camera_sink=None, pose_sink=None, intrinsic_sink=None,
                 cam_width=640, cam_height=480, cam_near=0.01, cam_far=10.0, parent=None):

        if VirtualCameraWidget.ShareWidget is None:
            ## create a dummy widget to allow sharing objects (textures, shaders, etc) between views
            VirtualCameraWidget.ShareWidget = QtOpenGL.QGLWidget()

        QtOpenGL.QGLWidget.__init__(self, parent, VirtualCameraWidget.ShareWidget)

        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.camera_sink = camera_sink
        self.pose_sink = pose_sink
        self.intrinsic_sink = intrinsic_sink

        self.bgtexture = visualization.BackgroundImage() if camera_sink is not None else None
        self.camera_pose = None
        self.camera_intrinsics = None

        if self.camera_sink and isinstance(self.camera_sink, uthelpers.PushSinkAdapter):
            def new_img_handler(ts):
                self.bgtexture.imageIn(self.camera_sink.value)
                self.updateGL()

            self.camera_sink.connect(new_img_handler)


        self.camera_width = float(cam_width)
        self.camera_height = float(cam_height)
        self.camera_near = float(cam_near)
        self.camera_far = float(cam_far)

        self.screen_width = cam_width
        self.screen_height = cam_height

        self.items = []
        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        self.keysPressed = {}
        self.keyTimer = QtCore.QTimer()
        self.keyTimer.timeout.connect(self.evalKeyState)

        self.makeCurrent()

    def addItem(self, item):
        self.items.append(item)
        if hasattr(item, 'initializeGL'):
            self.makeCurrent()
            try:
                item.initializeGL()
            except:
                self.checkOpenGLVersion('Error while adding item %s to GLViewWidget.' % str(item))

        item._setView(self)
        #print "set view", item, self, item.view()
        self.update()

    def removeItem(self, item):
        self.items.remove(item)
        item._setView(None)
        self.update()


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
            # calculate when data arrives ..
            proj = calibration.projectionMatrix3x3ToOpenGL(0., self.camera_width, 0., self.camera_height, self.camera_near, self.camera_far, self.camera_intrinsics)
            glMultMatrixd(proj)

        # XXX use code from Ubitrack Projection


    def setModelview(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.camera_pose is not None:
            # calculate when data arrives ..
            pose = self.camera_pose.invert().toMatrix()
            glMultMatrixd(pose)

        # XXX use code from Ubitrack VirtualCamera



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

    def mouseMoveEvent(self, ev):
        pass

    def mouseReleaseEvent(self, ev):
        pass

    def wheelEvent(self, ev):
        pass

    def keyPressEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()

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