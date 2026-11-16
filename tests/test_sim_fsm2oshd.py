"""Tests of the FSM2oshd simulation helpers (sim_fsm2oshd.py), with a stand-in FSM executable."""
import contextlib
import io
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from TopoPyScale import sim_fsm2oshd as sim


def quiet(fun, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fun(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
#  FSM2oshd simulation helpers, with a stand-in executable
# ---------------------------------------------------------------------------------------------------------------------
FAKE_FSM = r'''
import os, re, sys
nam = open(sys.argv[1]).read()
out = re.search(r"out_file\s*=\s*'([^']+)'", nam).group(1)
met = re.search(r"met_file\s*=\s*'([^']+)'", nam).group(1)
mode = re.search(r"CTILE\s*=\s*'(\w+)'", nam).group(1)
if 'fail' in os.path.basename(out):
    sys.exit(3)
base = 1.0 if mode == 'forest' else 3.0
with open(out, 'w') as f:
    for k, line in enumerate(l for l in open(met) if l.strip()):
        y, m, d, h = line.split()[:4]
        f.write(f"{y} {m} {d} {h} {base + 0.01 * k:.3f} {10 * base} {100 * base + k} -1.5 -0.5 0.5 1.0 1.5\n")
'''


def _write_case(sim_dir, n_points=3, fail_point=None):
    sim_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range('2021-01-01', periods=48, freq='1h')
    for i in range(n_points):
        met = sim_dir / f'fsm__met_{i:03d}.txt'
        pd.DataFrame({'y': times.year, 'm': times.month, 'd': times.day, 'h': times.hour}).assign(
            **{c: 1.0 for c in sim.FSM_MET_COLUMNS[4:]}).to_csv(met, sep=' ', header=False, index=False)
        for mode in ['forest', 'open']:
            out = sim_dir / f'fsm__outputs_{mode}_{i:03d}.txt'
            if fail_point == i and mode == 'open':
                out = sim_dir / f'fsm__outputs_fail_{i:03d}.txt'
            (sim_dir / f'fsm__{mode}_{i:03d}.nam').write_text(
                "\n&nam_grid\n  NNx = 1,\n/\n&nam_driving\n"
                f"  met_file = '{met}',\n  out_file = '{out}',\n/\n&nam_modconf\n  CTILE = '{mode}',\n/\n")
    pd.DataFrame({'proportion_with_forest': [0.05, 0.5, 1.0][:n_points]}).to_pickle(sim_dir / 'df_forest.pckl')


def test_fsm2oshd_chain(tmp_path):
    sim_dir = tmp_path / 'fsm_sim'
    _write_case(sim_dir)
    fake = tmp_path / 'fake_fsm.py'
    fake.write_text(FAKE_FSM)
    failed = quiet(sim.fsm2oshd_sim_parallel, simulation_path=sim_dir, fsm_exec=f'{sys.executable} {fake}', n_cores=2)
    assert failed == []
    assert not list(sim_dir.glob('fsm__outputs_*.txt'))  # text outputs converted and removed
    assert len(list(sim_dir.glob('fsm__outputs_*.nc'))) == 6

    out = quiet(sim.aggregate_all_open_forest, fname_df_forest=sim_dir / 'df_forest.pckl',
                fname_forest_outputs=str(sim_dir / 'fsm__outputs_forest_*.nc'),
                fname_open_outputs=str(sim_dir / 'fsm__outputs_open_*.nc'),
                fout=str(sim_dir / 'fsm_outputs'), n_cores=2, remove_tmp_outputs=False)
    assert len(out) == 3
    ds = quiet(sim.read_fsm_ds_with_mask, str(sim_dir / 'fsm_outputs_*.nc'))
    assert list(ds.point_ind.values) == [-9999, 0, 1, 2]
    # forest proportion 0.05 -> open only; 0.5 -> mean; 1 -> forest only
    k = np.arange(48) * 0.01
    np.testing.assert_allclose(ds.sd.sel(point_ind=0).values, 3 + k, atol=1e-3)
    np.testing.assert_allclose(ds.sd.sel(point_ind=1).values, 2 + k, atol=1e-3)
    np.testing.assert_allclose(ds.sd.sel(point_ind=2).values, 1 + k, atol=1e-3)
    np.testing.assert_allclose(ds.tsoil_1.sel(point_ind=1).values, -0.5, atol=1e-3)  # constant variable


def test_fsm2oshd_reports_failures_and_missing_executable(tmp_path):
    sim_dir = tmp_path / 'fsm_sim'
    _write_case(sim_dir, n_points=2, fail_point=1)
    fake = tmp_path / 'fake_fsm.py'
    fake.write_text(FAKE_FSM)
    with pytest.warns(UserWarning, match='1 FSM2oshd simulations failed'):
        failed = sim.fsm2oshd_sim_parallel(simulation_path=sim_dir, fsm_exec=f'{sys.executable} {fake}', n_cores=1)
    assert [p.endswith('fsm__open_001.nam') for p in failed] == [True]
    with pytest.raises(FileNotFoundError):
        sim.fsm2oshd_sim_parallel(simulation_path=sim_dir, fsm_exec=str(tmp_path / 'no_such_FSM'))


def test_output_to_ds_and_readers(tmp_path):
    sim_dir = tmp_path / 'fsm_sim'
    _write_case(sim_dir, n_points=2)
    fake = tmp_path / 'fake_fsm.py'
    fake.write_text(FAKE_FSM)
    quiet(sim.fsm2oshd_sim_parallel, simulation_path=sim_dir, fsm_exec=f'{sys.executable} {fake}', n_cores=1,
          fsm_open_nam=None, keep_txt=True)
    df = sim.read_pt_fsm2oshd(sim_dir / 'fsm__outputs_forest_000.txt')
    assert df.index[0] == pd.Timestamp('2021-01-01 00:00') and len(df) == 48
    assert len(sim.read_pt_met_fsm2oshd(sim_dir / 'fsm__met_000.txt')) == 48
    out = quiet(sim.output_to_ds, str(sim_dir / 'fsm__outputs_forest_*.nc'), fout=str(tmp_path / 'renamed'))
    assert sorted(p.name for p in tmp_path.glob('renamed_*.nc')) == ['renamed_000.nc', 'renamed_001.nc']
    assert len(out) == 2
