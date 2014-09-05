__author__ = 'jack'
from atom.api import Typed, Event, observe
from enaml.core.declarative import d_
from enaml_opengl.scenegraph_node import GraphicsNode

from ubitrack.core import math, measurement
from ubitrack.vision import vision
from ubitrack.visualization import visualization

class BackgroundTexture(GraphicsNode):

    bgtexture = Typed(visualization.BackgroundImage)

    image_in = Event(vision.ImageMeasurement)

    def _default_bgtexture(self):
        return visualization.BackgroundImage()

    @observe("image_in")
    def _update_background(self, change):
        self.bgtexture.imageIn(change['value'])
        self.trigger_update()


    def render_node(self, context):
        cs = context.get("canvas_size")
        self.bgtexture.draw(cs.width, cs.height)