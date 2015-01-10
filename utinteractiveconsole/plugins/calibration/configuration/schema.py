__author__ = 'jack'

from atom.api import Atom, Value, List, Dict, Str, Bool, Int, Float, Enum, Typed, Coerced

import logging

from utinteractiveconsole.configuration.schema import Device, Plugin
from utinteractiveconsole.persistence.streamfile import UBITRACK_DATATYPES
from utinteractiveconsole.persistence.recordsource import RECORD_SELECTORS

log = logging.getLogger(__name__)


class PhantomHapticDevice(Device):
    joint_length1 = Float(0.20955)
    joint_length2 = Float(0.20955)
    origin_offset_x = Float(0.0)
    origin_offset_y = Float(0.0)
    origin_offset_z = Float(0.0)


class CalibrationWizards(Plugin):
    wizards = List()

    root_directory_template = Str('$(config_directory)/vharcalibration')
    calib_directory_template = Str('$(root_directory)/calib/$(wizard_name)/$(domain_name)/$(setup_name)/$(user_name)')
    dfg_directory_template = Str('$(root_directory)/dfgs/$(wizard_name)/$(domain_name)/$(setup_name)/$(user_name)/$(platform_name)')
    results_directory_template = Str('$(data_directory)/$(wizard_name)/$(setup_name)/$(user_name)')
    calib_initialization_directory_template = Str('$(config_directory)/vharcalibration/calib/init')


class CalibrationWizard(Atom):
    name = Str()
    title = Str()

    # deprecated
    modules_namespace = Str()

    facade_handler = Enum('inprocess',
                          'subprocess')

    root_directory = Str()

    calibration_directory = Str()
    dfg_directory = Str()
    results_directory = Str()

    report_filename = Str('calibration_report.yaml')

    modules = List()
    datasources = Dict()
    calibsources = Dict()

    # could be implemented using calibsources ...
    calibration_init_files = Dict()

    # subclass needed ?
    haptic_device = Typed(PhantomHapticDevice)


class CalibrationWizardModule(Atom):
    enabled = Bool(False)
    name = Str()
    dependencies = List()
    parameters = Value()

    dfg_filename = Str()
    calibration_files = List()
    record_directory = Str()

    autocomplete_enabled = Bool(False)
    autocomplete_maxerror = Value()

    task_description = Str()


class CalibrationWizardStreamFile(Atom):
    fieldname = Str()

    filename = Str()
    datatype = Enum(*UBITRACK_DATATYPES)
    is_array = Bool(False)

    selector = Enum(*RECORD_SELECTORS)


class CalibrationWizardCalibFile(Atom):
    fieldname = Str()

    filename = Str()
    datatype = Enum(*UBITRACK_DATATYPES)
    is_array = Bool(False)


class CalibrationWizardDataSource(Atom):
    name = Str()

    data_directory = Str()
    reference_field = Str()
    stream_files = List()


class CalibrationWizardCalibSource(Atom):
    name = Str()

    # data_directory = Str()
    calib_files = List()


class OfflineCalibrationParameters(Atom):
    # global
    stream_skip_first_nseconds = Float(0.0)

    # tooltip calibration
    tooltip_enabled = Bool(True)
    tooltip_datasource = Str()
    tt_minimal_angle_between_measurements = Float(0.1)
    tt_use_pose = Bool(False)

    # fwkbase_position - tracking target attached to joint1
    fwkbase_position_enabled = Bool(False)
    fwkbase_position_datasource = Str()

    # fwkbase_position2 - tracking target attached to turret
    fwkbase_position2_enabled = Bool(False)
    fwkbase_position2_datasource = Str()

    # absolute orientation calibration
    absolute_orientation_enabled = Bool(True)
    absolute_orientation_datasource = Str()
    ao_method = Enum('fwkpose', 'fwkbase')
    ao_negate_upvector = Bool(False)
    ao_inital_maxdistance_from_origin = Float(0.1)
    ao_minimal_distance_between_measurements = Float(0.01)
    ao_refinement_expand_coverage = Float(1.5)
    ao_refinement_shrink_distance = Float(1.0)
    ao_initialize_anglecorrection_calibsource = Str()
    ao_number_of_clusters = Int(0)

    # joint angles calibration
    joint_angle_calibration_enabled = Bool(True)
    joint_angle_calibration_datasource = Str()
    ja_minimal_distance_between_measurements = Float(0.01)
    ja_maximum_distance_to_reference = Float(0.02)
    ja_refinement_min_difference = Float(0.00001)
    ja_refinement_max_iterations = Int(5)
    ja_refinement_shrink_distance = Float(0.9)
    ja_number_of_clusters = Int(0)
    ja_use_2nd_order = Bool(False)
    ja_exclude_calibration_samples_from_evaluation = Bool(True)

    # reference orientation
    reference_orientation_enabled = Bool(False)
    reference_orientation_datasource = Str()
    ro_minimal_angle_between_measurements = Float(0.1)

    # gimbal angles calibration
    gimbal_angle_calibration_enabled = Bool(True)
    gimbal_angle_calibration_datasource = Str()
    ga_minimal_angle_between_measurements = Float(0.1)
    ga_use_tooltip_offset = Bool(False)
    ga_number_of_clusters = Int(0)
    ga_use_2nd_order = Bool(False)
    ga_exclude_calibration_samples_from_evaluation = Bool(True)

    # time-delay estimation
    timedelay_estimation_enabled = Bool(False)
    timedelay_estimation_datasource = Str()

    result_evaluation_enabled = Bool(False)
    result_evaluation_datasource = Str()

def from_ini_file(ini_cfg, global_config=None):

    if not ini_cfg.has_section('calibration_wizard'):
        log.warn('No calibration wizard configuration found.')
        return

    if not ini_cfg.has_option('calibration_wizard', 'config_version') or \
        ini_cfg.getint('calibration_wizard', 'config_version') < 2:

        log.warn('Invalid config version for calibration wizard.')
        return

    ini_sections = ini_cfg.sections()

    devices = {}
    for section_name in [sn for sn in ini_sections if sn.startswith('ubitrack.devices.')]:
        device = PhantomHapticDevice(
            name=section_name.rsplit('.')[-1],
            joint_length1=ini_cfg.getfloat(section_name, 'joint_length1'),
            joint_length2=ini_cfg.getfloat(section_name, 'joint_length2'),
            origin_offset_x=ini_cfg.getfloat(section_name, 'origin_offset_x'),
            origin_offset_y=ini_cfg.getfloat(section_name, 'origin_offset_y'),
            origin_offset_z=ini_cfg.getfloat(section_name, 'origin_offset_z'),
        )
        devices[device.name] = device

    active_wizards = [v.strip() for v in ini_cfg.get('calibration_wizard', 'wizards').split(',')]

    calibration_wizards = []

    for aw in active_wizards:
        aw_ns = 'calibration_wizard.%s' % aw
        if not ini_cfg.has_section(aw_ns):
            log.warn('Missing config section: %s' % aw_ns)
            continue

        config_ns = ini_cfg.get(aw_ns, 'config_namespace')
        module_ns = ini_cfg.get(aw_ns, 'module_namespace')

        if not ini_cfg.has_section(config_ns):
            log.warn('Missing config section: %s' % config_ns)
            continue

        calibration_wizard_modules = []
        module_config_prefix = '%s.modules.' % config_ns
        for section_name in [sn for sn in ini_sections if sn.startswith(module_config_prefix)]:
            module_name = section_name.replace(module_config_prefix, '')

            cwm = CalibrationWizardModule(
                name=module_name,
                enabled=ini_cfg.getboolean(section_name, 'enabled'),
            )
            # optional attributes for calibration_wizard_module
            if ini_cfg.has_option(section_name, 'dfg_filename'):
                cwm.dfg_filename = ini_cfg.get(section_name, 'dfg_filename')
            if ini_cfg.has_option(section_name, 'calib_files'):
                cwm.calibration_files = [v.strip() for v in ini_cfg.get(section_name, 'calib_files').split(',')]
            if ini_cfg.has_option(section_name, 'recorddir'):
                cwm.record_directory = ini_cfg.get(section_name, 'recorddir')
            if ini_cfg.has_option(section_name, 'dependencies'):
                cwm.dependencies = [v.strip() for v in ini_cfg.get(section_name, 'dependencies').split(',')]
            if ini_cfg.has_option(section_name, 'autocomplete_enable'):
                cwm.autocomplete_enabled = ini_cfg.getboolean(section_name, 'autocomplete_enable')
            if ini_cfg.has_option(section_name, 'autocomplete_maxerror'):
                cwm.autocomplete_maxerror = ini_cfg.get(section_name, 'autocomplete_maxerror')
            if ini_cfg.has_option(section_name, 'taskdescription'):
                cwm.task_description = ini_cfg.get(section_name, 'taskdescription')

            module_parameters_ns = '%s.parameters.%s' % (config_ns, module_name)
            if ini_cfg.has_section(module_parameters_ns):
                if module_name.startswith('offline_calibration'):
                    module_parameters = OfflineCalibrationParameters()

                    if ini_cfg.has_option(module_parameters_ns, 'stream_skip_first_nseconds'):
                        module_parameters.stream_skip_first_nseconds = ini_cfg.getfloat(module_parameters_ns, 'stream_skip_first_nseconds')

                    if ini_cfg.has_option(module_parameters_ns, 'tooltip_enabled'):
                        module_parameters.tooltip_enabled = ini_cfg.getboolean(module_parameters_ns, 'tooltip_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'tooltip_datasource'):
                        module_parameters.tooltip_datasource = ini_cfg.get(module_parameters_ns, 'tooltip_datasource')
                    if ini_cfg.has_option(module_parameters_ns, 'tt_minimal_angle_between_measurements'):
                        module_parameters.tt_minimal_angle_between_measurements = ini_cfg.getfloat(module_parameters_ns, 'tt_minimal_angle_between_measurements')
                    if ini_cfg.has_option(module_parameters_ns, 'tt_use_pose'):
                        module_parameters.tt_use_pose = ini_cfg.getboolean(module_parameters_ns, 'tt_use_pose')

                    if ini_cfg.has_option(module_parameters_ns, 'fwkbase_position_enabled'):
                        module_parameters.fwkbase_position_enabled = ini_cfg.getboolean(module_parameters_ns, 'fwkbase_position_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'fwkbase_position_datasource'):
                        module_parameters.fwkbase_position_datasource = ini_cfg.get(module_parameters_ns, 'fwkbase_position_datasource')

                    if ini_cfg.has_option(module_parameters_ns, 'fwkbase_position2_enabled'):
                        module_parameters.fwkbase_position2_enabled = ini_cfg.getboolean(module_parameters_ns, 'fwkbase_position2_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'fwkbase_position2_datasource'):
                        module_parameters.fwkbase_position2_datasource = ini_cfg.get(module_parameters_ns, 'fwkbase_position2_datasource')

                    if ini_cfg.has_option(module_parameters_ns, 'absolute_orientation_enabled'):
                        module_parameters.absolute_orientation_enabled = ini_cfg.getboolean(module_parameters_ns, 'absolute_orientation_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'absolute_orientation_datasource'):
                        module_parameters.absolute_orientation_datasource = ini_cfg.get(module_parameters_ns, 'absolute_orientation_datasource')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_method'):
                        module_parameters.ao_method = ini_cfg.get(module_parameters_ns, 'ao_method')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_negate_upvector'):
                        module_parameters.ao_negate_upvector = ini_cfg.getboolean(module_parameters_ns, 'ao_negate_upvector')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_inital_maxdistance_from_origin'):
                        module_parameters.ao_inital_maxdistance_from_origin = ini_cfg.getfloat(module_parameters_ns, 'ao_inital_maxdistance_from_origin')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_minimal_distance_between_measurements'):
                        module_parameters.ao_minimal_distance_between_measurements = ini_cfg.getfloat(module_parameters_ns, 'ao_minimal_distance_between_measurements')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_refinement_expand_coverage'):
                        module_parameters.ao_refinement_expand_coverage = ini_cfg.getfloat(module_parameters_ns, 'ao_refinement_expand_coverage')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_refinement_shrink_distance'):
                        module_parameters.ao_refinement_shrink_distance = ini_cfg.getfloat(module_parameters_ns, 'ao_refinement_shrink_distance')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_initialize_anglecorrection_calibsource'):
                        module_parameters.ao_initialize_anglecorrection_calibsource = ini_cfg.get(module_parameters_ns, 'ao_initialize_anglecorrection_calibsource')
                    if ini_cfg.has_option(module_parameters_ns, 'ao_number_of_clusters'):
                        module_parameters.ao_number_of_clusters = ini_cfg.getint(module_parameters_ns, 'ao_number_of_clusters')

                    if ini_cfg.has_option(module_parameters_ns, 'joint_angle_calibration_enabled'):
                        module_parameters.joint_angle_calibration_enabled = ini_cfg.getboolean(module_parameters_ns, 'joint_angle_calibration_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'joint_angle_calibration_datasource'):
                        module_parameters.joint_angle_calibration_datasource = ini_cfg.get(module_parameters_ns, 'joint_angle_calibration_datasource')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_minimal_distance_between_measurements'):
                        module_parameters.ja_minimal_distance_between_measurements = ini_cfg.getfloat(module_parameters_ns, 'ja_minimal_distance_between_measurements')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_maximum_distance_to_reference'):
                        module_parameters.ja_maximum_distance_to_reference = ini_cfg.getfloat(module_parameters_ns, 'ja_maximum_distance_to_reference')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_refinement_min_difference'):
                        module_parameters.ja_refinement_min_difference = ini_cfg.getfloat(module_parameters_ns, 'ja_refinement_min_difference')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_refinement_max_iterations'):
                        module_parameters.ja_refinement_max_iterations = ini_cfg.getint(module_parameters_ns, 'ja_refinement_max_iterations')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_refinement_shrink_distance'):
                        module_parameters.ja_refinement_shrink_distance = ini_cfg.getfloat(module_parameters_ns, 'ja_refinement_shrink_distance')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_number_of_clusters'):
                        module_parameters.ja_number_of_clusters = ini_cfg.getint(module_parameters_ns, 'ja_number_of_clusters')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_use_2nd_order'):
                        module_parameters.ja_use_2nd_order = ini_cfg.getboolean(module_parameters_ns, 'ja_use_2nd_order')
                    if ini_cfg.has_option(module_parameters_ns, 'ja_exclude_calibration_samples_from_evaluation'):
                        module_parameters.ja_exclude_calibration_samples_from_evaluation = ini_cfg.getboolean(module_parameters_ns, 'ja_exclude_calibration_samples_from_evaluation')

                    if ini_cfg.has_option(module_parameters_ns, 'reference_orientation_enabled'):
                        module_parameters.reference_orientation_enabled = ini_cfg.getboolean(module_parameters_ns, 'reference_orientation_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'reference_orientation_datasource'):
                        module_parameters.reference_orientation_datasource = ini_cfg.get(module_parameters_ns, 'reference_orientation_datasource')
                    if ini_cfg.has_option(module_parameters_ns, 'ro_minimal_angle_between_measurements'):
                        module_parameters.ro_minimal_angle_between_measurements = ini_cfg.getfloat(module_parameters_ns, 'ro_minimal_angle_between_measurements')

                    if ini_cfg.has_option(module_parameters_ns, 'gimbal_angle_calibration_enabled'):
                        module_parameters.gimbal_angle_calibration_enabled = ini_cfg.getboolean(module_parameters_ns, 'gimbal_angle_calibration_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'gimbal_angle_calibration_datasource'):
                        module_parameters.gimbal_angle_calibration_datasource = ini_cfg.get(module_parameters_ns, 'gimbal_angle_calibration_datasource')
                    if ini_cfg.has_option(module_parameters_ns, 'ga_minimal_angle_between_measurements'):
                        module_parameters.ga_minimal_angle_between_measurements = ini_cfg.getfloat(module_parameters_ns, 'ga_minimal_angle_between_measurements')
                    if ini_cfg.has_option(module_parameters_ns, 'ga_use_tooltip_offset'):
                        module_parameters.ga_use_tooltip_offset = ini_cfg.getboolean(module_parameters_ns, 'ga_use_tooltip_offset')
                    if ini_cfg.has_option(module_parameters_ns, 'ga_number_of_clusters'):
                        module_parameters.ga_number_of_clusters = ini_cfg.getint(module_parameters_ns, 'ga_number_of_clusters')
                    if ini_cfg.has_option(module_parameters_ns, 'ga_use_2nd_order'):
                        module_parameters.ga_use_2nd_order = ini_cfg.getboolean(module_parameters_ns, 'ga_use_2nd_order')
                    if ini_cfg.has_option(module_parameters_ns, 'ga_exclude_calibration_samples_from_evaluation'):
                        module_parameters.ga_exclude_calibration_samples_from_evaluation = ini_cfg.getboolean(module_parameters_ns, 'ga_exclude_calibration_samples_from_evaluation')

                    if ini_cfg.has_option(module_parameters_ns, 'timedelay_estimation_enabled'):
                        module_parameters.timedelay_estimation_enabled = ini_cfg.getboolean(module_parameters_ns, 'timedelay_estimation_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'timedelay_estimation_datasource'):
                        module_parameters.timedelay_estimation_datasource = ini_cfg.get(module_parameters_ns, 'timedelay_estimation_datasource')

                    if ini_cfg.has_option(module_parameters_ns, 'result_evaluation_enabled'):
                        module_parameters.result_evaluation_enabled = ini_cfg.getboolean(module_parameters_ns, 'result_evaluation_enabled')
                    if ini_cfg.has_option(module_parameters_ns, 'result_evaluation_datasource'):
                        module_parameters.result_evaluation_datasource = ini_cfg.get(module_parameters_ns, 'result_evaluation_datasource')

                else:
                    module_parameters = {}
                    for k, v in ini_cfg.items(module_parameters_ns):
                        module_parameters[k] = v

                cwm.parameters = module_parameters

            calibration_wizard_modules.append(cwm)

        datasources = {}
        datasource_config_prefix = '%s.datasources.' % config_ns
        for section_name in [sn for sn in ini_sections if sn.startswith(datasource_config_prefix)]:
            stream_files = []
            for k, v in [i for i in ini_cfg.items(section_name) if i[0].startswith('item.')]:
                fname = k.replace('item.', '')
                sfarray = False
                sfname, sfdt, sfsel = [e.strip() for e in v.split(',')]
                if sfdt.endswith('-list'):
                    sfarray = True
                    sfdt = sfdt.replace('-list', '')
                sf = CalibrationWizardStreamFile(
                    fieldname=fname,
                    filename=sfname,
                    datatype=sfdt,
                    is_array=sfarray,
                    selector=sfsel,
                )
                stream_files.append(sf)

            ds = CalibrationWizardDataSource(
                name=section_name.replace(datasource_config_prefix, ''),
                data_directory=ini_cfg.get(section_name, 'data_directory'),
                reference_field=ini_cfg.get(section_name, 'reference'),
                stream_files=stream_files,
            )
            datasources[ds.name] = ds

        calibsources = {}
        calibsource_config_prefix = '%s.calibsources.' % config_ns
        for section_name in [sn for sn in ini_sections if sn.startswith(calibsource_config_prefix)]:
            calib_files = []
            for k, v in [i for i in ini_cfg.items(section_name) if i[0].startswith('item.')]:
                fname = k.replace('item.', '')
                cfarray = False
                cfname, cfdt = [e.strip() for e in v.split(',')]
                if cfdt.endswith('-list'):
                    cfarray = True
                    cfdt = cfdt.replace('-list', '')
                cf = CalibrationWizardCalibFile(
                    fieldname=fname,
                    filename=cfname,
                    datatype=cfdt,
                    is_array=cfarray,
                )
                calib_files.append(cf)

            cs = CalibrationWizardCalibSource(
                name=section_name.replace(datasource_config_prefix, ''),
                # calib_directory=ini_cfg.get(config_ns, 'calibdir'),
                calib_files=calib_files,
            )
            calibsources[ds.name] = cs

        init_files_ns = '%s.initialize_files' % config_ns
        init_files = {}
        if ini_cfg.has_section(init_files_ns):
            for k, v in ini_cfg.items(init_files_ns):
                init_files[k] = v

        cw = CalibrationWizard(
            name=aw,
            title=ini_cfg.get(config_ns, 'name'),
            modules_namespace=module_ns,
            root_directory=ini_cfg.get(config_ns, 'rootdir'),
            calibration_directory=ini_cfg.get(config_ns, 'calibdir'),
            dfg_directory=ini_cfg.get(config_ns, 'dfgdir'),
            results_directory=ini_cfg.get(config_ns, 'resultsdir'),
            modules=calibration_wizard_modules,
            datasources=datasources,
            calibsources=calibsources,
            calibration_init_files=init_files,
        )
        # optional attributes for calibration_wizard
        if ini_cfg.has_option(config_ns, 'facade_handler'):
            cw.facade_handler = ini_cfg.get(config_ns, 'facade_handler')
        if ini_cfg.has_option(config_ns, 'haptic_device'):
            cw.haptic_device = devices.get(ini_cfg.get(config_ns, 'haptic_device'))
        if ini_cfg.has_option(config_ns, 'report_filename'):
            cw.report_filename = devices.get(ini_cfg.get(config_ns, 'report_filename'))

        calibration_wizards.append(cw)

    cws = CalibrationWizards(
        wizards=calibration_wizards,
    )

    if global_config is not None:
        global_config.plugins['calibration_wizard'] = cws
        global_config.devices = devices

    return cws