import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from pathlib import Path
import pytest

from trough import config, _trough, scripts, get_data
from trough._config import TroughIdParams


def test_preprocess_interval():
    T = 10
    nlat = 100
    nlon = 100
    bg_est_shape = (1, 21, 21)
    times = np.datetime64("2000") + np.arange(T) * np.timedelta64(1, 'h')
    background = np.sin(np.linspace(0, 2 * np.pi, nlon))[None, None, :] * np.ones((T, nlat, nlon))
    signal = np.sin(10 * np.linspace(0, 2 * np.pi, nlon))[None, None, :] * np.ones((T, nlat, nlon))
    log_tec = background + signal
    tec = 10 ** log_tec
    tec[1, 20, 20] = 200
    tec[2, 40, 40] = -5
    coords = {'time': times, 'mlat': np.arange(nlat), 'mlt': np.arange(nlon)}
    data = xr.DataArray(tec, coords=coords, dims=['time', 'mlat', 'mlt']).to_dataset(name='tec')
    _trough.preprocess_interval(data, bg_est_shape=bg_est_shape)
    assert 'x' in data
    det_log_tec = data['x'].values
    assert np.nanmean(abs(det_log_tec - signal)) < .1


def test_model_artificial_example():
    mlt_grid, mlat_grid = np.meshgrid(config.get_mlt_vals(), config.get_mlat_vals())
    times = np.datetime64("2000") + np.arange(4) * np.timedelta64(1, 'h')
    # nominal labels
    labels1 = abs(mlat_grid - 65) <= 2
    # partial labels
    labels2 = (abs(mlat_grid - 65) <= 2) * (abs(mlt_grid) < 3)
    # high labels - reject
    labels3 = (mlat_grid > 83) * (abs(mlt_grid) < 3)
    # low labels - reject
    labels4 = (mlat_grid < 35) * (abs(mlt_grid) < 2)
    # convert to det log tec
    basis = _trough.get_rbf_matrix(mlat_grid.shape)
    det_log_tec = -.2 * (basis @ np.column_stack((labels1.ravel(), labels2.ravel(), labels3.ravel(), labels4.ravel()))).T
    shp = (4,) + mlat_grid.shape
    det_log_tec = det_log_tec.reshape(shp)
    det_log_tec += np.random.randn(*shp) * .1
    coords = {'time': times, 'mlat': config.get_mlat_vals(), 'mlt': config.get_mlt_vals()}
    data = xr.Dataset({
        'x': xr.DataArray(
            det_log_tec,
            coords=coords,
            dims=['time', 'mlat', 'mlt']
        ),
        'model': xr.DataArray(
            np.ones((times.shape[0], mlt_grid.shape[1])) * 65,
            coords={'time': times, 'mlt': config.get_mlt_vals()},
            dims=['time', 'mlt']
        ),
    })
    args = _trough.get_optimization_args(data, 30, 1, 2, 1, .15, .06)
    output = np.stack([_trough.run_single(*a).reshape(mlat_grid.shape) for a in args], axis=0)
    print(output[0][labels1].mean(), output[1][labels2].mean(), output[2][labels3].mean(), output[3][labels4].mean())
    assert output[0][labels1].mean() > 1
    assert output[1][labels2].mean() > 1
    assert output[2][labels3].mean() < 1
    assert output[3][labels4].mean() < 1


def test_postprocess():
    """verify that small troughs are rejected, verify that troughs that wrap around the border are not incorrectly
    rejected
    """
    mlt_grid, mlat_grid = np.meshgrid(config.get_mlt_vals(), config.get_mlat_vals())
    good_labels = (abs(mlat_grid - 65) < 3) * (abs(mlt_grid) < 2)
    small_reject = (abs(mlat_grid - 52) <= 1) * (abs(mlt_grid - 4) <= .5)
    boundary_good_labels = (abs(mlat_grid - 65) < 3) * (abs(mlt_grid) >= 10.2)
    boundary_bad_labels = (abs(mlat_grid - 52) < 3) * (abs(mlt_grid) >= 11.5)
    weird_good_labels = (abs(mlat_grid - 40) < 5) * (abs(mlt_grid - 9) <= 2.5)
    weird_good_labels += (abs(mlat_grid - 34) <= 2) * (abs(mlt_grid) > 11.3)
    weird_good_labels += (abs(mlat_grid - 44) <= 2) * (abs(mlt_grid) > 11.3)
    high_labels = (abs(mlat_grid - 80) < 2) * (abs(mlt_grid + 6) <= 3)
    arb = np.ones((1, 180)) * 70
    initial_labels = good_labels + small_reject + boundary_good_labels + boundary_bad_labels + weird_good_labels + high_labels
    coords = {'time': [0], 'mlat': config.get_mlat_vals(), 'mlt': config.get_mlt_vals()}
    data = xr.Dataset({
        'labels': xr.DataArray(
            initial_labels[None],
            coords=coords,
            dims=['time', 'mlat', 'mlt']
        ),
        'arb': xr.DataArray(
            arb,
            coords={'time': [0], 'mlt': config.get_mlt_vals()},
            dims=['time', 'mlt']
        ),
    })
    _trough.postprocess(data, 'north', perimeter_th=50)
    labels = data['labels'].values[0]
    assert labels[good_labels].all()
    assert not labels[small_reject].any()
    assert labels[boundary_good_labels].all()
    assert not labels[boundary_bad_labels].any()
    assert labels[weird_good_labels].all()
    assert not labels[high_labels].any()


def test_get_optimization_args():
    """Verify that get_optimization_args properly handles various outputs
    """
    T = 3
    D = 10
    x = np.random.randn(T, D, D)
    x[0, :2, :2] = np.nan
    times = np.datetime64("2000") + np.arange(T) * np.timedelta64(1, 'h')
    mlt_vals = np.arange(D)
    mlat_grid = np.arange(D)[:, None] * np.ones((1, D))
    arb = np.ones((T, D)) * 7.5 - 3

    coords = {'time': times, 'mlat': np.arange(D), 'mlt': mlt_vals}
    data = xr.Dataset({
        'x': xr.DataArray(
            x,
            coords=coords,
            dims=['time', 'mlat', 'mlt']
        ),
        'model': xr.DataArray(
            arb,
            coords={'time': times, 'mlt': mlt_vals},
            dims=['time', 'mlt']
        ),
    })
    args = _trough.get_optimization_args(data, 10, .5, .5, .5, .5, .5)
    var, basis, x_out, tv, l2 = args[0]
    assert len(args) == T
    assert basis.shape == (D ** 2 - 4, D ** 2)
    assert np.all(x_out == x[0][np.isfinite(x[0])])
    assert np.all(np.diag(tv.toarray()) == 1)
    assert tv.shape == (D ** 2, D ** 2)
    assert l2.min() == .5
    assert l2.max() == 5
    l2reg = abs(mlat_grid - arb[0]).ravel()
    l2reg -= l2reg.min()
    l2reg = (10 - 1) * l2reg / l2reg.max() + 1
    l2reg *= .5
    assert np.all(l2reg == args[0][4])


def test_get_tec_troughs():
    """Verify that get_tec_troughs can detect an actual trough, verify that high troughs are rejected using auroral
    boundary data
    """
    start_date = datetime(2015, 10, 7, 6, 0, 0)
    end_date = start_date + timedelta(hours=12)
    params = TroughIdParams(bg_est_shape=(1, 19, 19), model_weight_max=5, l2_weight=.1, tv_weight=.05, tv_hw=2)
    with config.temp_config(trough_id_params=params):
        scripts.download_all(start_date, end_date)
        scripts.process_all(start_date, end_date)
        data_north = _trough.label_trough_interval(
            start_date, end_date, config.trough_id_params, 'north',
            config.processed_tec_dir, config.processed_arb_dir, config.processed_omni_file
        )
        data_south = _trough.label_trough_interval(
            start_date, end_date, config.trough_id_params, 'south',
            config.processed_tec_dir, config.processed_arb_dir, config.processed_omni_file
        )

    labels = data_north['labels'].values
    assert labels.shape == (12, 60, 180)
    assert labels[1, 20:30, 60:120].mean() > .5
    for i in range(12):
        assert labels[i][(data_north.mlat > data_north['arb'][i] + 3).values].sum() == 0
        assert labels[i][(data_south.mlat < data_north['arb'][i] - 3).values].sum() == 0


@pytest.mark.parametrize('dates',
                         [
                             [datetime(2021, 1, 3, 6, 0, 0), datetime(2021, 1, 3, 12, 0, 0)],
                             [datetime(2020, 12, 31, 20, 0, 0), datetime(2021, 1, 1, 4, 0, 0)]
                         ])
def test_process_trough_interval(dates):
    start_date, end_date = dates
    n_times = (end_date - start_date) / timedelta(hours=1)
    scripts.download_all(start_date, end_date)
    scripts.process_all(start_date, end_date)
    data = _trough.label_trough_interval(
        start_date, end_date, config.trough_id_params, 'north',
        config.processed_tec_dir, config.processed_arb_dir, config.processed_omni_file
    )
    assert 'labels' in data
    assert 'tec' in data
    assert data.time.shape[0] == n_times
    assert data.mlat.shape[0] == 60
    assert data.mlt.shape[0] == 180
    assert np.nanmean(data['tec'].values[data['labels'].values]) < np.nanmean(data['tec'].values[~data['labels'].values])


@pytest.mark.parametrize('dates',
                         [
                             [datetime(2021, 1, 3, 6, 0, 0), datetime(2021, 1, 3, 12, 0, 0)],
                             [datetime(2020, 12, 31, 20, 0, 0), datetime(2021, 1, 1, 4, 0, 0)]
                         ])
def test_script(dates):
    start_date, end_date = dates
    n_times = (end_date - start_date) / timedelta(hours=1)
    with TemporaryDirectory() as tempdir:
        with config.temp_config(base_dir=tempdir):
            scripts.full_run(*dates)
            n_files = len([p for p in Path(config.processed_labels_dir).glob('labels*.nc')])
            assert n_files == (end_date.year - start_date.year + 1)
            data = get_data(start_date, end_date, 'north')
            data.load()
            assert data.time.shape[0] == n_times
            data.close()
