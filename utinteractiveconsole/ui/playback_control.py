__author__ = 'MVL'


from atom.api import Atom, List, Float, Int, observe
import enaml
from enaml.qt.qt_application import QtApplication


class Records(Atom):
    """ A simple class representing a person object.

    """
    items = List()

    interval = Int()

    current_position = Int(0)

    @observe("current_position")
    def debug_curpos(self, change):
        print self.current_position


if __name__ == '__main__':
    # Create an employee with a boss
    data = Records(
        items=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], interval=500,
    )

    # Import our Enaml EmployeeView
    with enaml.imports():
        from utinteractiveconsole.ui.views.playback_control import PlaybackControlWindow

    app = QtApplication()
    # Create a view and show it.
    view = PlaybackControlWindow(data=data)
    view.show()

    app.start()