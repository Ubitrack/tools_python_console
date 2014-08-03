__author__ = 'jack'

from atom.api import Atom, Value, List, observe




class TooltipCalibration(Atom):

    facade = Value()

    data_tracker_poses = List()

    result_tooltip_offset = Value()

    @observe("facade")
    def handle_facade_change(self, change):
        print change


