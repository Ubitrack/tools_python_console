__author__ = 'jack'

import os
import logging
import numpy as np

log = logging.getLogger(__name__)

from atom.api import Event, Bool, Typed, observe
from enaml.qt import QtCore
from enaml.application import deferred_call

import enaml
with enaml.imports():
    from .views.offline_calibration import OfflineCalibrationPanel

from utinteractiveconsole.plugins.calibration.module import ModuleBase
from utinteractiveconsole.plugins.calibration.controller import CalibrationController


class BackgroundCalculationThread(QtCore.QThread):

    def __init__(self, task, ctrl):
        log.info("Init Background Calculation")
        super(BackgroundCalculationThread, self).__init__()
        self.task = task
        self.ctrl = ctrl

    def run(self):
        log.info("BackgroundCalculationThread.run()")
        deferred_call(self.set_is_working, True)
        try:
            self.task.run()
        except Exception, e:
            log.error("Error in BackgroundCalculationThread:")
            log.exception(e)
        finally:
            deferred_call(self.set_is_working, False)

    def set_is_working(self, v):
        self.ctrl.is_working = v


class OfflineCalibrationController(CalibrationController):

    bgThread = Typed(BackgroundCalculationThread)
    is_working = Bool(False)


class OfflineCalibrationModule(ModuleBase):

    def get_category(self):
        return "Calibration"

    def get_widget_class(self):
        return OfflineCalibrationPanel

    def get_controller_class(self):
        return OfflineCalibrationController