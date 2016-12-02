UbiTrack Interactive Console

Author: Ulrich Eck <ueck@net-labs.de>

ubitrackinteractive console provides a framework for rapid prototyping of UbiTrack Gui Applications.
It builds upon ubitrack_python and the Qt framework.

Requirements:
==============

- ubitrack (use cmake/build/install from github.com/Ubitrack/tools_python_console)
- atom (conda install atom)
- enaml (conda install enaml)
- stevedore (pip install stevedore)
- pyqtgraph (conda install pyqtgraph)
- networkx (conda install networkx)
- pyopengl (conda install pyopengl)
- lxml (conda install lxml)
- setuptools (should be on any uptodate system ..)
- twisted (conda install twisted)

Installation:
==============

use the following command for a local developer installation::
  python setup.py develop

otherwise::
  python setup.py install

the installation does not install requirements automatically. Typically requirements are installed like::

  pip install <name>


Configuration:
==============

please copy utic.ini.example to some/path utic.ini and set the environment variable UTIC_CONFIG_FILE to point to it.

Required sections are::

    [ubitrack]
    components_path = H:\UbiTrack\install\bin\ubitrack



Documentation:
==============

more docs to follow

Problem solving:
================
if utvision or utvisualization break with dll import error, or incompatible qt versions are reported, then deinstall pyqt from anaconda (conda remove pyqt) and install pyqt from http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.1/PyQt4-4.11.1-gpl-Py2.7-Qt4.8.6-x32.exe