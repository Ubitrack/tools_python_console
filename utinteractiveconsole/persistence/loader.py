__author__ = 'jack'

from atom.api import Atom, Value, Dict, List

import yaml
import logging
import os
import sys
import numpy as np

from ubitrack.core import util, measurement, math
from utinteractiveconsole.extension import CustomEntryPoint
from utinteractiveconsole.playback import (loadData, DSC)

from .dataset import DataSet
from .datasource import DataSource

log = logging.getLogger(__name__)


MS_DIVIDER = 1000000.0
UBITRACK_DATATYPES = ['distance', 'position2D', 'position3D', 'quat', 'pose', 'mat33', 'mat34', 'mat44']

def instances_from_config(config, pathname):
    path = pathname.split(".")

    if config is None:
        raise ValueError("Invalid config object")

    section = config
    for p in path:
        if p in section:
            section = section[p]
        else:
            raise KeyError("Missing path in config: %s (%s)" % (p, pathname))

    if not isinstance(section, dict):
        raise ValueError("Invalid config found in: %s - expected dictionary" % pathname)

    return CustomEntryPoint.instances_from_items(section.items())


def get_datasourceloader(filename, working_directory=None):
    if working_directory is None:
        working_directory = os.path.dirname(filename)
    config = None
    if os.path.isfile(filename):
        try:
            log.info("Loading datasources from file: %s" % (filename,))
            config = yaml.load(open(filename, 'r'))
        except Exception, e:
            log.error("Error parsing config file(s): %s" % (filename,))
            log.exception(e)
    if config is not None:
        return DataSourceLoader(config=config, working_directory=working_directory)
    return None


class DataSourceLoader(Atom):

    config = Value()
    working_directory = Value()

    datasources = Dict()
    datasource_names = List()

    datasets = Dict()
    dataset_names = List()

    streamreaders = Dict()
    streaminterpolators = Dict()
    streamprocessors = Dict()
    calibreaders = Dict()

    def _default_streamreaders(self):
        result = instances_from_config(self.config, "import.streamreaders")
        # log.info("Available stream readers: %s" % ",".join(result.keys()))
        return result

    def _default_streaminterpolators(self):
        result = instances_from_config(self.config, "import.streaminterpolators")
        # log.info("Available stream interpolators: %s" % ",".join(result.keys()))
        return result

    def _default_streamprocessors(self):
        result = instances_from_config(self.config, "import.streamprocessors")
        # log.info("Available stream processors: %s" % ",".join(result.keys()))
        return result

    def _default_calibreaders(self):
        result = instances_from_config(self.config, "import.calibreaders")
        # log.info("Available calib readers: %s" % ",".join(result.keys()))
        return result


    def load_datasource(self, config, datasource_sname):
        log.info("Load Datasource: %s" % datasource_sname)
        ds_cfg = config[datasource_sname]
        title = ds_cfg.get('title')

        data_directory = os.path.join(self.working_directory, ds_cfg.get("data_directory"))

        # parse schema
        reference_column = ds_cfg["reference"]
        columns = ds_cfg["columns"]

        reference_data = None
        items = []
        for k, v in columns.items():
            if k == reference_column:
                reference_data = (k, v)
            else:
                items.append((k, v))

        def mkDSC(name, spec):
            filename = spec["filename"]
            reader = self.streamreaders.get(spec["reader"], None)
            if reader is None:
                raise ValueError("Invalid Configuration for datasource element: %s -> reader not found: %s"
                                 % (name, spec["reader"]))
            interpolator_key = spec.get("interpolator")
            interpolator = None
            if interpolator_key is not None:
                interpolator = self.streaminterpolators.get(interpolator_key, None)
                if interpolator is None:
                    raise ValueError("Invalid Configuration for datasource element: %s -> interpolator not found: %s"
                                     % (name, interpolator_key))
            return DSC(name, filename, reader, interpolator=interpolator)

        records = loadData(data_directory,
                        mkDSC(*reference_data),
                        items=(mkDSC(*i) for i in items))
        # some useful metadata
        record_count = len(records)
        ts_data = [p.timestamp for p in records]
        # ms accuracy ok for now
        interval = int(np.diff(np.asarray(ts_data)).mean() / MS_DIVIDER)
        return DataSource(name=datasource_sname, title=title, data=records,
                          record_count=record_count, reference_timestamps=ts_data, interval=interval)


    def load_dataset(self, config, dataset_sname):
        log.info("Load Dataset: %s" % dataset_sname)
        ds_cfg = config[dataset_sname]
        title = ds_cfg['title']
        calib_directory = os.path.join(self.working_directory, ds_cfg.get("calib_directory"))

        datasource = self.datasources.get(ds_cfg['datasource'])
        processor_factory = self.streamprocessors.get(ds_cfg['processor'])
        attributes = {}
        for key, spec in ds_cfg.get('attributes', {}).items():
            try:
                if spec['type'] == 'calibfile':
                    reader = self.calibreaders.get(spec['reader'])
                    if reader is not None:
                        log.info("Load calibfile from: %s for key: %s" % (spec['filename'], key))
                        attributes[key] = reader(os.path.join(calib_directory, spec['filename']).encode(sys.getfilesystemencoding()))
                    else:
                        log.warn("Calibfile reader not found: %s" % spec['reader'])
                elif spec['type'] == 'instance':
                    loader = CustomEntryPoint.parse('loader', spec['loader']).load()
                    if loader is not None:
                        log.info("create instance for key: %s" % key)
                        args = spec.get('args', [])
                        kwargs = spec.get('kwargs', {})
                        attributes[key] = loader(self.working_directory, ds_cfg, *args, **kwargs)
            except Exception, e:
                log.error("Error while computing attributes for dataset: %s" % dataset_sname)
                log.exception(e)

        return DataSet(name=dataset_sname, title=title, datasource=datasource,
                       processor_factory=processor_factory, attributes=attributes)


    def _default_datasources(self):
        """default datasources"""
        config = self.config
        result = {}
        if config is not None and "datasources" in config:
            if not isinstance(config["datasources"], dict):
                raise ValueError("Invalid datasources configuration")
            all_datasources = config["datasources"].keys()
            for datasource_sname in sorted(all_datasources):
                try:
                    data = self.load_datasource(config["datasources"], datasource_sname)
                    record_count = data.record_count
                    if record_count > 0:
                        log.info('Datasource: %s loaded %d records with fieldnames: %s' %
                                 (datasource_sname, record_count, ','.join(data.output_field_names)))
                        result[datasource_sname] = data
                    else:
                        log.warn('No records loaded!')
                except Exception, e:
                    log.error("Error while loading datasource: %s" % datasource_sname)
                    log.exception(e)

        return result

    def _default_datasets(self):
        """default datasets"""
        config = self.config
        result = {}
        if config is not None and "datasets" in config:
            if not isinstance(config["datasets"], dict):
                raise ValueError("Invalid datasets configuration")
            all_datasets = config["datasets"].keys()
            for dataset_sname in sorted(all_datasets):
                try:
                    data = self.load_dataset(config["datasets"], dataset_sname)
                    result[dataset_sname] = data
                except Exception, e:
                    log.error("Error while loading dataset: %s" % dataset_sname)
                    log.exception(e)

        return result


    def _default_datasource_names(self):
        return sorted(self.datasources.keys())

    def _default_dataset_names(self):
        return sorted(self.datasets.keys())
