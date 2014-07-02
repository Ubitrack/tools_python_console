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
      package_data = {'utinteractiveconsole' : ['srgs/*', 'resources/icons/*', 'ui/views/*.enaml']},
      #package_dir = {'git':'git'},
      license = "BSD License",
      requires=(
        'pyqtgraph',
        'pyopengl',
        'stevedore',
        'lxml',
        'atom',
        'enaml',
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
                'load_dataflow = utinteractiveconsole.plugins.load_dataflow:LoadDataflow',
                'calibration_wizard = utinteractiveconsole.plugins.calibration.wizard:CalibrationWizard',
            ],
            'vharcalibration.default_camera.module': [
                'camera_intrinsics = utinteractiveconsole.plugins.calibration.modules.camera_intrinsics:CameraIntrinsicsCalibrationModule',
                'camera_handeye = utinteractiveconsole.plugins.calibration.modules.camera_handeye:CameraHandEyeCalibrationModule',
            ],
            'vharcalibration.stereo_camera.module': [
                'camera_intrinsics_left = utinteractiveconsole.plugins.calibration.modules.camera_intrinsics:CameraIntrinsicsCalibrationModule',
                'camera_intrinsics_right = utinteractiveconsole.plugins.calibration.modules.camera_intrinsics:CameraIntrinsicsCalibrationModule',
                'camera_stereo = utinteractiveconsole.plugins.calibration.modules.camera_stereo:CameraStereoCalibrationModule',
                'camera_handeye = utinteractiveconsole.plugins.calibration.modules.camera_handeye:CameraHandEyeCalibrationModule',
            ],
            'vharcalibration.default_workspace.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'externaltracker_calibration = utinteractiveconsole.plugins.calibration.modules.externaltracker_calibration:ExternalTrackerCalibrationModule',
                'hapticdevice_calibration = utinteractiveconsole.plugins.calibration.modules.hapticdevice_calibration:HapticDeviceCalibrationModule',
                'tooltip_calibration = utinteractiveconsole.plugins.calibration.modules.tooltip_calibration:TooltipCalibrationModule',
                'absolute_orientation_init_calibration = utinteractiveconsole.plugins.calibration.modules.absolute_orientation_init_calibration:AbsoluteOrientationInitCalibrationModule',
                'absolute_orientation_calibration = utinteractiveconsole.plugins.calibration.modules.absolute_orientation_calibration:AbsoluteOrientationCalibrationModule',
                'timedelay_estimation_calibration = utinteractiveconsole.plugins.calibration.modules.timedelay_estimation:TimeDelayEstimationModule',
                'hapticworkspace_init_calibration = utinteractiveconsole.plugins.calibration.modules.hapticworkspace_init_calibration:HapticWorkspaceInitCalibrationModule',
                'hapticworkspace_calibration = utinteractiveconsole.plugins.calibration.modules.hapticworkspace_calibration:HapticWorkspaceCalibrationModule',
                'hapticgimbal_init_calibration = utinteractiveconsole.plugins.calibration.modules.hapticgimbal_init_calibration:HapticGimbalInitCalibrationModule',
                'hapticgimbal_calibration = utinteractiveconsole.plugins.calibration.modules.hapticgimbal_calibration:HapticGimbalCalibrationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
        },
      )