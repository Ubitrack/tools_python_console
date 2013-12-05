__author__ = 'MVL'


from atom.api import Atom, List, Float, Int, Value, observe
import enaml


# Import our Enaml EmployeeView
with enaml.imports():
    from utinteractiveconsole.ui.views.playback_control import PlaybackControlWindow, PlaybackControl

class Records(Atom):
    """ A simple class representing a records object.

    """
    items = List()

    # This property holds the timeout interval in milliseconds.
    interval = Int()

    current_position = Int(0)

    current_value = Value()

    @observe("current_position")
    def _select_curval(self, change):
        idx = change["value"]
        if idx < len(self.items):
            self.current_value = self.items[idx]





if __name__ == '__main__':
    from enaml.qt.qt_application import QtApplication

    # Create an employee with a boss
    data = Records(
        items=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], interval=500,
    )

    app = QtApplication()

    # Create a view and show it.
    view = PlaybackControlWindow(data=data)

    def curpos_observer(change):
        print "curpos: %s, value: %s" % (data.current_position, data.items[data.current_position])

    data.observe("current_position", curpos_observer)

    view.show()

    app.start()