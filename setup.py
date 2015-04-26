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
      package_data = {'utinteractiveconsole' : ['srgs/*',
                                                'resources/icons/*',
                                                'ui/views/*.enaml',
                                                'plugins/views/*.enaml',
                                                'plugins/calibration/views/*.enaml',
                                                'plugins/calibration/modules/views/*.enaml',
                                                'plugins/calibration/controllers/views/*.enaml',
                                                ]},
      #package_dir = {'git':'git'},
      license = "BSD License",
      requires=(
        'pyqtgraph',
        'pyopengl',
        'stevedore',
        'lxml',
        'atom',
        'enaml',
        'twisted',
        'numpy',
        'scipy',
        'pyyaml',
        'pandas',
        'matplotlib',
        'lxml'
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
                'calibration_wizard = utinteractiveconsole.plugins.calibration.wizard:CalibrationWizard',
                'remote_control = utinteractiveconsole.plugins.remote_control.main:RemoteControl',
            ],
            # 'vharcalibration.controllers.preview': [
            #     'time_delay_estimation_preview = utinteractiveconsole.plugins.calibration.controllers.preview_time_delay_estimation:TimeDelayEstimationPreviewFactory',
            #     'phantom_colocation_preview = utinteractiveconsole.plugins.calibration.controllers.preview_phantom_colocation:PhantomColocationPreviewFactory',
            # ],
            'vharcalibration.default_camera.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'camera_intrinsics = utinteractiveconsole.plugins.calibration.modules.camera_intrinsics:CameraIntrinsicsCalibrationModule',
                'camera_handeye = utinteractiveconsole.plugins.calibration.modules.camera_handeye:CameraHandEyeCalibrationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.stereo_camera.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'camera_intrinsics_left = utinteractiveconsole.plugins.calibration.modules.camera_intrinsics:CameraIntrinsicsCalibrationModule',
                'camera_intrinsics_right = utinteractiveconsole.plugins.calibration.modules.camera_intrinsics:CameraIntrinsicsCalibrationModule',
                'camera_stereo = utinteractiveconsole.plugins.calibration.modules.camera_stereo:CameraStereoCalibrationModule',
                'camera_handeye = utinteractiveconsole.plugins.calibration.modules.camera_handeye:CameraHandEyeCalibrationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.time_delay_estimation.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'tooltip_calibration = utinteractiveconsole.plugins.calibration.modules.tooltip_calibration:TooltipCalibrationModule',
                'absolute_orientation_calibration = utinteractiveconsole.plugins.calibration.modules.absolute_orientation_calibration:AbsoluteOrientationCalibrationModule',
                'timedelay_estimation_calibration = utinteractiveconsole.plugins.calibration.modules.timedelay_estimation:TimeDelayEstimationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.ismar14.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'tooltip_calibration = utinteractiveconsole.plugins.calibration.modules.tooltip_calibration:TooltipCalibrationModule',
                'offline_datacollection = utinteractiveconsole.plugins.calibration.modules.offline_datacollection:OfflineDataCollectionModule',
                'offline_calibration = utinteractiveconsole.plugins.calibration.modules.offline_calibration2:OfflineCalibrationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.time_delay_estimation_3way.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'tracker_handeye = utinteractiveconsole.plugins.calibration.modules.tracker_handeye:TrackerHandEyeCalibrationModule',
                'tooltip_calibration = utinteractiveconsole.plugins.calibration.modules.tooltip_calibration:TooltipCalibrationModule',
                'absolute_orientation_calibration_3way = utinteractiveconsole.plugins.calibration.modules.absolute_orientation_calibration_3way:AbsoluteOrientationCalibrationModule',
                'timedelay_estimation_calibration_3way = utinteractiveconsole.plugins.calibration.modules.timedelay_estimation_3way:TimeDelayEstimationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.hapticroom_calibration_initial.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'tooltip_calibration = utinteractiveconsole.plugins.calibration.modules.tooltip_calibration:TooltipCalibrationModule',
                'absolute_orientation_calibration = utinteractiveconsole.plugins.calibration.modules.absolute_orientation_calibration:AbsoluteOrientationCalibrationModule',
                'offline_datacollection = utinteractiveconsole.plugins.calibration.modules.offline_datacollection:OfflineDataCollectionModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.hapticroom_calibration_complete.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'offline_datacollection = utinteractiveconsole.plugins.calibration.modules.offline_datacollection:OfflineDataCollectionModule',
                'offline_calibration = utinteractiveconsole.plugins.calibration.modules.offline_calibration2:OfflineCalibrationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.workspace_calibration_3way.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'tracker_handeye = utinteractiveconsole.plugins.calibration.modules.tracker_handeye:TrackerHandEyeCalibrationModule',
                'tooltip_calibration = utinteractiveconsole.plugins.calibration.modules.tooltip_calibration:TooltipCalibrationModule',
                'offline_datacollection = utinteractiveconsole.plugins.calibration.modules.offline_datacollection:OfflineDataCollectionModule',
                'offline_calibration = utinteractiveconsole.plugins.calibration.modules.offline_calibration2:OfflineCalibrationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
            'vharcalibration.basic_workspace.module': [
                'calibration_start = utinteractiveconsole.plugins.calibration.modules.calibration_start:CalibrationStartModule',
                'absolute_orientation_refpoints_calibration = utinteractiveconsole.plugins.calibration.modules.absolute_orientation_refpoints:AbsoluteOrientationRefPointsCalibrationModule',
                'calibration_result = utinteractiveconsole.plugins.calibration.modules.calibration_result:CalibrationResultModule',
            ],
        },
      )