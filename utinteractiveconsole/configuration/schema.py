__author__ = 'jack'

from atom.api import Atom, Value, List, Dict, Str, Bool, Int, Float, Enum, Typed, Coerced


class GlobalConfig(Atom):
    domain_name = Str()
    setup_name = Str()
    user_name = Str()
    platform_name = Str()

    config_directory = Str()
    data_directory = Str()
    record_directory = Str()

    components_path = Str()
    logging_config = Str()
    py_logging_config = Str()

    plugins = Dict()
    devices = Dict()


class Device(Atom):
    name = Str()

class Plugin(Atom):
    name = Str()


def from_ini_file(ini_cfg):
    return GlobalConfig(
        domain_name=ini_cfg.get('ubitrack', 'domain_name'),
        setup_name=ini_cfg.get('ubitrack', 'setup_name'),
        user_name=ini_cfg.get('ubitrack', 'user_name'),
        platform_name=ini_cfg.get('ubitrack', 'platform_name'),
        config_directory=ini_cfg.get('ubitrack', 'config_directory'),
        data_directory=ini_cfg.get('ubitrack', 'data_directory'),
        record_directory=ini_cfg.get('ubitrack', 'record_directory'),
        components_path=ini_cfg.get('ubitrack', 'components_path'),
    )