"""
Check a config.yml against the keys TopoPyScale reads, listed in config_reference.yml (the example of the
documentation). Keys of older versions are reported with what replaced them, unknown keys with the closest
documented key, as a misspelled key would otherwise be read as empty without notice.

S. Filhol, J. Fiddes and contributors
"""
import difflib
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REFERENCE = Path(__file__).with_name('config_reference.yml')
# values that are data (lists, dictionaries of data), not settings
DATA_KEYS = {'project.extent', 'project.authors', 'sampling.toposub.clustering_features'}
OLD_KEYS = {
    'project.parallelization.downscaling_method': 'not read: downscaling runs CPU_cores threads over blocks of points',
    'project.parallelization.setting.dask': 'not read: downscaling runs CPU_cores threads over blocks of points',
    'project.split': 'removed: long periods are processed by blocks of time steps (toposcale.time_block)',
    'outputs.file.downscaled_pt': 'removed: the points are written to outputs.file.zarr_store',
    'outputs.file.landform': 'not read: the landform map is outputs/landform.tif',
    'sampling.points.epsg': 'not read: x, y must be in the DEM projection',
    'sampling.points.ID_col': 'renamed sampling.points.name_column',
    'clean_up.delete_tmp_dirs': 'renamed clean_up.rm_tmp_dirs',
    'climate.era5.rm_daily': 'not read: ERA5 is written to the zarr store, without daily files',
    'climate.era5.cds_output_format': 'not read: netcdf is downloaded',
    'toposcale.pt_sampling_method': 'not read: the horizontal interpolation is toposcale.interpolation_method',
    'dem.dem_resol': 'not read: the resolution is read from the DEM file',
    'project.CPU_cores': 'not read: the number of cores is project.parallelization.setting.multicore.CPU_cores',
}


def flatten(d, prefix=''):
    """Dotted keys of a nested dictionary, stopping at DATA_KEYS."""
    for k, v in (d or {}).items():
        key = f'{prefix}{k}'
        if isinstance(v, dict) and key not in DATA_KEYS:
            yield from flatten(v, key + '.')
        else:
            yield key


def reference_keys():
    """Keys of config_reference.yml."""
    with open(REFERENCE) as f:
        return set(flatten(yaml.safe_load(f)))


def check_config(config):
    """
    Messages about the keys of a configuration that TopoPyScale does not read.

    Args:
        config (dict): configuration as read from config.yml

    Returns:
        list: one message per key: renamed or removed key, or unknown key with the closest documented key
    """
    known = reference_keys()
    source = (config.get('project') or {}).get('climate') or 'era5'
    messages, reported = [], set()
    climate = config.get('climate') or {}
    if source not in climate:
        messages.append(f'project.climate is {source}, but there is no climate.{source} section')
    for key in flatten(config):
        parts = key.split('.')
        if parts[0] == 'climate' and len(parts) > 2 and isinstance(climate.get(parts[1]), dict):
            if parts[1] != source:
                continue   # settings of another climate source, not used by this run
            key_ref = '.'.join(['climate', 'era5'] + parts[2:])
        else:
            key_ref = key
        if key_ref in known:
            continue
        old = [k for k in OLD_KEYS if key_ref == k or key_ref.startswith(k + '.')]
        if old:
            if old[0] not in reported:
                reported.add(old[0])
                messages.append(f'{old[0]}: {OLD_KEYS[old[0]]}')
            continue
        close = difflib.get_close_matches(key_ref, known, n=1, cutoff=0.8)
        if not close:   # misspelled and/or misplaced: compare the last part of the key
            last = {k.split('.')[-1]: k for k in sorted(known)}
            near = difflib.get_close_matches(parts[-1], list(last), n=1, cutoff=0.8)
            close = [last[near[0]]] if near else []
        hint = f' Did you mean {close[0]}?' if close else ''
        messages.append(f'{key} is not a TopoPyScale setting and is not read.{hint}')
    return messages


def warn_config(config):
    """Log check_config() messages as warnings. Returns the messages."""
    messages = check_config(config)
    for m in messages:
        logger.warning(f'config: {m}')
    return messages
