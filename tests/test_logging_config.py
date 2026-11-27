"""Messages through logging (TopoPyScale.log) and the check of config.yml against the documented keys."""
import contextlib
import io
import logging
import warnings

import pytest
import yaml

import TopoPyScale
from TopoPyScale import config_check as cc
from TopoPyScale import log


@pytest.fixture(autouse=True)
def restore_logging():
    yield
    log.setup()


def messages(verbose):
    out = io.StringIO()
    log.setup(verbose=verbose)
    lg = log.get_logger('topo_export')
    with contextlib.redirect_stdout(out):
        lg.debug('detail')
        lg.info('---> progress')
        lg.warning('something odd')
    return out.getvalue()


def test_verbose_levels():
    assert messages(True) == '---> progress\nWARNING: something odd\n'
    assert messages(False) == 'WARNING: something odd\n'
    assert messages('debug').startswith('detail\n')
    with pytest.raises(ValueError, match='verbose'):
        log.setup(verbose='loud')


def test_log_file_and_warnings(tmp_path):
    f = log.setup(verbose=False, logfile=tmp_path / 'logs' / 'run.log', capture_warnings=True)
    with contextlib.redirect_stdout(io.StringIO()):
        log.get_logger('topoclass').info('---> step done')
        warnings.warn('a warning of a dependency')
    text = f.read_text()
    assert 'INFO    TopoPyScale.topoclass: ---> step done' in text and 'a warning of a dependency' in text
    assert text[:4].isdigit()   # time stamp


def test_importing_topopyscale_keeps_warnings_visible():
    import TopoPyScale.topoclass  # noqa: F401 (imports every module)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('default')
        warnings.warn('still visible', UserWarning)
    assert [str(x.message) for x in w] == ['still visible']
    assert not any(f[0] == 'ignore' and f[2] in (Warning, UserWarning) and f[1] is None
                   for f in warnings.filters)


def test_config_check():
    cfg = {'project': {'climate': 'era5', 'CPU_core': 4, 'split': {'IO': True, 'time': {'nb': 2}}},
           'climate': {'era5': {'timestep': '1h', 'plevel': [700]}, 'other_source': {'anything': 1}},
           'toposcale': {'surface_effect': {'methd': 'dreamit'}}, 'clean_up': {'delete_tmp_dirs': True}}
    msgs = cc.check_config(cfg)
    assert msgs == [
        'project.CPU_core is not a TopoPyScale setting and is not read. Did you mean '
        'project.parallelization.setting.multicore.CPU_cores?',
        'project.split: removed: long periods are processed by blocks of time steps (toposcale.time_block)',
        'climate.era5.plevel is not a TopoPyScale setting and is not read. Did you mean climate.era5.plevels?',
        'toposcale.surface_effect.methd is not a TopoPyScale setting and is not read. Did you mean '
        'toposcale.surface_effect.method?',
        'clean_up.delete_tmp_dirs: renamed clean_up.rm_tmp_dirs']
    assert cc.check_config({'project': {'climate': 'era6'}, 'climate': {'era5': {}}}) == [
        'project.climate is era6, but there is no climate.era6 section']
    with open(cc.REFERENCE) as f:
        assert cc.check_config(yaml.safe_load(f)) == []


def test_missing_config_file(tmp_path):
    from TopoPyScale.topoclass import Topoclass
    with pytest.raises(FileNotFoundError, match='nothere.yml'):
        Topoclass(tmp_path / 'nothere.yml')


def test_package_logger_shows_progress_by_default():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        logging.getLogger('TopoPyScale.fetch_era5').info('---> downloading')
    assert out.getvalue() == '---> downloading\n'
    assert TopoPyScale.log.LOGGER.propagate is False
