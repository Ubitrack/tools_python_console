__author__ = 'MVL'
import numpy as np
from numpy.linalg import norm
import logging

from atom.api import Bool, Value, Typed, Int, Float
import enaml
with enaml.imports():
    from .views.offline_datacollection import OfflineDataCollectionPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController

log = logging.getLogger(__name__)

class OfflineDataCollectionController(CalibrationController):
    pass



class OfflineDataCollectionModule(ModuleBase):

    def get_category(self):
        return "Data Collection"

    def get_widget_class(self):
        return OfflineDataCollectionPanel

    def get_controller_class(self):
        return OfflineDataCollectionController