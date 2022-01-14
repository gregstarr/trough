from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
import itertools
from datetime import datetime, timedelta

from trough import config, scripts
from trough._tec import process_interval, get_tec_data, _get_downloaded_tec_data, get_tec_paths, calculate_bins
from trough._download import MadrigalTecDownloader
from trough.exceptions import InvalidProcessDates


def test_calculate_bins():
    mlat = np.arange(10)[:, None] * np.ones((1, 10))
    mlt = np.arange(10)[None, None, :] * np.ones((1, 10, 1))
    tec = np.zeros((1, 10, 10))
    tec[0, 0, 0] = 10
    tec[0, 0, -1] = 20
    tec[0, -1, 0] = 30
    times = np.ones(1) * np.nan
    data = xr.DataArray(
        tec,
        coords={
            'time': times,
            'x': np.arange(10),
            'y': np.arange(10),
            'mlat': (('x', 'y'), mlat),
            'mlt': (('time', 'x', 'y'), mlt)
        },
        dims=['time', 'x', 'y']
    )
    be = np.array([-.5, 4.5, 9.5])
    out_tec = calculate_bins(data, be, be)
    assert out_tec.shape == (2, 2)
    assert out_tec[0, 0] == 10 / 25
    assert out_tec[0, 1] == 20 / 25
    assert out_tec[1, 0] == 30 / 25
    assert out_tec[1, 1] == 0


@pytest.fixture(scope='module')
def download_dir():
    with TemporaryDirectory() as t:
        yield t


@pytest.fixture(scope='module')
def process_dir():
    with TemporaryDirectory() as t:
        yield t


def test_download_tec(test_dates, download_dir):
    start, end = test_dates
    downloader = MadrigalTecDownloader(download_dir, 'gstarr', 'gstarr@bu.edu', 'BU')
    downloader.download(*test_dates)
    tec_files = list(Path(download_dir).glob('*'))
    assert len(tec_files) > 0
    data = _get_downloaded_tec_data(*test_dates, download_dir)
    assert data.time.values[0] < np.datetime64(start, 's')
    assert data.time.values[-1] > np.datetime64(end, 's')


map_periods = [np.timedelta64(10, 'm'), np.timedelta64(30, 'm'), np.timedelta64(1, 'h'), np.timedelta64(2, 'h')]
@pytest.mark.parametrize(['dt', 'mlt_bins', 'mlat_bins'],
                         itertools.product(
                             map_periods,
                             [config.get_mlt_bins(), np.arange(10)],
                             [config.get_mlat_bins(), np.arange(10)]
                         ))
def test_process_tec(download_dir, process_dir, test_dates, dt, mlt_bins, mlat_bins):
    start, end = test_dates
    correct_times = np.arange(np.datetime64(start, 's'), np.datetime64(end, 's'), dt)
    mlt_vals = (mlt_bins[:-1] + mlt_bins[1:]) / 2
    mlat_vals = (mlat_bins[:-1] + mlat_bins[1:]) / 2
    processed_file = Path(process_dir) / 'tec_test.nc'
    process_interval(start, end, processed_file, download_dir, dt, mlat_bins, mlt_bins)
    assert processed_file.exists()
    data = xr.open_dataarray(processed_file)
    assert data.shape == (correct_times.shape[0], mlat_vals.shape[0], mlt_vals.shape[0])
    assert (data.mlt == mlt_vals).all().item()
    assert (data.mlat == mlat_vals).all().item()
    assert (data.time == correct_times).all().item()


def test_process_tec_out_of_range(download_dir, process_dir, test_dates):
    dt = np.timedelta64(1, 'h')
    start, end = [date - timedelta(days=100) for date in test_dates]
    processed_file = Path(process_dir) / 'tec_test.nc'
    with pytest.raises(InvalidProcessDates):
        process_interval(start, end, processed_file, download_dir, dt, config.get_mlat_bins(), config.get_mlt_bins())


def test_get_tec_data(download_dir, process_dir, test_dates):
    start, end = test_dates
    dt = np.timedelta64(1, 'h')
    mlt_bins = config.get_mlt_vals()
    mlat_bins = config.get_mlat_vals()
    mlt_vals = (mlt_bins[:-1] + mlt_bins[1:]) / 2
    mlat_vals = (mlat_bins[:-1] + mlat_bins[1:]) / 2
    correct_times = np.arange(np.datetime64(start), np.datetime64(end), dt)
    processed_file = get_tec_paths(start, end, process_dir)[0]
    process_interval(start, end, processed_file, download_dir, dt, mlat_bins, mlt_bins)
    data = get_tec_data(start, end, process_dir)
    assert data.shape == (correct_times.shape[0], mlat_vals.shape[0], mlt_vals.shape[0])
    assert (data.mlt == mlt_vals).all().item()
    assert (data.mlat == mlat_vals).all().item()
    assert (data.time == correct_times).all().item()


def test_scripts(test_dates):
    with TemporaryDirectory() as base_dir:
        with config.temp_config(base_dir=base_dir) as cfg:
            scripts.download_tec(*test_dates)
            tec_files = list(Path(cfg.download_tec_dir).glob('*'))
            assert len(tec_files) > 0
            data = _get_downloaded_tec_data(*test_dates, cfg.download_tec_dir)
            assert data.time.values[0] < np.datetime64(test_dates[0], 's')
            assert data.time.values[-1] > np.datetime64(test_dates[-1], 's')
            scripts.process_tec(*test_dates)
            data = get_tec_data(*test_dates, cfg.processed_tec_dir)
            data.load()
            dt = np.timedelta64(1, 'h')
            mlt_vals = config.get_mlt_vals()
            mlat_vals = config.get_mlat_vals()
            correct_times = np.arange(np.datetime64(test_dates[0]), np.datetime64(test_dates[-1]), dt)
            assert data.shape == (correct_times.shape[0], mlat_vals.shape[0], mlt_vals.shape[0])
            assert (data.mlt == mlt_vals).all().item()
            assert (data.mlat == mlat_vals).all().item()
            assert (data.time == correct_times).all().item()