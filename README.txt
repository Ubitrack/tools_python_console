UbiTrack Interactive Console

Author: Ulrich Eck <ueck@net-labs.de>

ubitrackinteractive console provides a framework for rapid prototyping of UbiTrack Gui Applications.
It builds upon ubitrack_python and the Qt framework.

Requirements:
==============

- ubitrack (use cmake/build/install)
- atom (install from github.com/nucleic/atom, checkout master, python setup.py install)
- enaml (install from github.com/nucleic/enaml, checkout master, python setup.py install)
- stevedore (pip install stevedore)
- pyqtgraph (pip install pyqtgraph)
- networkx (pip install networkx)
- pyopengl (pip install pyopengl)
- lxml (part of Anaconda, otherwise pip install lxml)
- setuptools (should be on any uptodate system ..)

Installation:
==============

use the following command for a local developer installation::
  python setup.py develop


otherwise::
  python setup.py install


Documentation:
==============

more docs to follow