__author__ = 'jack'

from atom.api import Atom, Value, Dict, List

import yaml
import logging
import os
import sys
import numpy as np

from ubitrack.core import util, measurement, math
from utinteractiveconsole.extension import CustomEntryPoint

from .dataset import DataSet
from .recordsource import RecordSource, FieldInterpolator
from .streamfile import StreamFileSpec

log = logging.getLogger(__name__)

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


def get_recordsourceloader(filename, working_directory=None):
    if working_directory is None:
        working_directory = os.path.dirname(filename)
    config = None
    if os.path.isfile(filename):
        try:
            log.info("Loading recordsources from file: %s" % (filename,))
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

    recordsources = Dict()
    recordsource_names = List()

    datasets = Dict()
    dataset_names = List()

    streamprocessors = Dict()
    calibreaders = Dict()

    def _default_streamprocessors(self):
        result = instances_from_config(self.config, "import.streamprocessors")
        # log.info("Available stream processors: %s" % ",".join(result.keys()))
        return result

    def _default_calibreaders(self):
        result = instances_from_config(self.config, "import.calibreaders")
        # log.info("Available calib readers: %s" % ",".join(result.keys()))
        return result


    def load_recordsource(self, config, recordsource_sname):
        log.info("Load Datasource: %s" % recordsource_sname)
        ds_cfg = config[recordsource_sname]
        title = ds_cfg.get('title')

        data_directory = os.path.join(self.working_directory, ds_cfg.get("data_directory"))

        # parse schema
        reference_column = ds_cfg["reference"]
        columns = ds_cfg["columns"]

        fields = []

        for k, v in columns.items():
            fs = StreamFileSpec(fieldname=k,
                                filename=os.path.join(data_directory, v['filename']).strip(),
                                datatype=v['datatype'].strip().lower(),
                                is_array=v.get('is_array', "false").lower() == "true",
                                )
            f = FieldInterpolator(filespec=fs,
                                  is_reference=bool(k == reference_column),
                                  selector=v.get('selector', 'matching').strip().lower(),
                                  latency=float(v.get('latency', 0.0)))
            fields.append(f)

        return RecordSource(name=recordsource_sname,
                            title=title,
                            fieldspec=fields)

    def load_dataset(self, config, dataset_sname):
        log.info("Load Dataset: %s" % dataset_sname)
        ds_cfg = config[dataset_sname]
        title = ds_cfg['title']
        calib_directory = os.path.join(self.working_directory, ds_cfg.get("calib_directory"))

        recordsource = self.recordsources.get(ds_cfg['recordsource'])
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

        return DataSet(name=dataset_sname, title=title, recordsource=recordsource,
                       processor_factory=processor_factory, attributes=attributes)


    def _default_recordsources(self):
        """default recordsources"""
        config = self.config
        result = {}
        if config is not None and "recordsources" in config:
            if not isinstance(config["recordsources"], dict):
                raise ValueError("Invalid recordsources configuration")
            all_recordsources = config["recordsources"].keys()
            for recordsource_sname in sorted(all_recordsources):
                try:
                    data = self.load_recordsource(config["recordsources"], recordsource_sname)
                    result[recordsource_sname] = data
                except Exception, e:
                    log.error("Error while loading recordsource: %s" % recordsource_sname)
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


    def _default_recordsource_names(self):
        return sorted(self.recordsources.keys())

    def _default_dataset_names(self):
        return sorted(self.datasets.keys())
