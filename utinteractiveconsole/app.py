__author__ = 'mvl'
import logging

from atom.api import Atom, Value, Typed, Dict, Event

from .guilogging import Syslog
from .extension import WorkspaceExtensionManager

log = logging.getLogger(__name__)


class AppState(Atom):
    context = Dict()
    extensions = Typed(WorkspaceExtensionManager)
    current_workspace = Value()

    args = Value()
    options = Value()

    syslog = Typed(Syslog)

    workspace_started = Event()
    workspace_stopped = Event()


