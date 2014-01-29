#!/usr/bin/env python
try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

setup(name = "utinteractiveconsole",
      version = '0.1',
      description = "Interactive Console for UbiTrack",
      author = "Ulrich Eck",
      author_email = "ulrich.eck@magicvisionlab.com",
      url = "http://www.magicvisionlab.com",
      packages = find_packages('.'),
      #package_data = {'git.test' : ['fixtures/*']},
      #package_dir = {'git':'git'},
      license = "BSD License",
      requires=(
        'pyqtgraph',
        'pyopengl',
        'stevedore',
        'lxml',
      ),
      zip_safe=False,
      long_description = """\
This module controls my garden""",
      classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        entry_points={
            'console_scripts': [
                'utic = utinteractiveconsole.main:main',
                ],
            'utinteractiveconsole.extension': [
                'load_dataflow = utinteractiveconsole.extensions.load_dataflow:LoadDataflow',
            ],
        },
      )