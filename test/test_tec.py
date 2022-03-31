from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
import itertools
from datetime import datetime, timedelta
import json

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
    out_tec = calculate_bins(data, be, be, 'north')
    assert out_tec.shape == (2, 2)
    assert out_tec[0, 0] == 10 / 25
    assert out_tec[0, 1] == 20 / 25
    assert out_tec[1, 0] == 30 / 25
    assert out_tec[1, 1] == 0


def test_calculate_bins_south():
    mlat = np.arange(10)[:, None] * np.ones((1, 10)) * -1
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
    out_tec = calculate_bins(data, be, be, 'south')
    assert out_tec.shape == (2, 2)
    assert out_tec[0, 0] == 10 / 25
    assert out_tec[0, 1] == 20 / 25
    assert out_tec[1, 0] == 30 / 25
    assert out_tec[1, 1] == 0


def test_file_list():
    start_date = datetime(2001, 1, 1, 12, 0, 0)
    end_date = datetime(2001, 1, 2, 12, 0, 0)
    with TemporaryDirectory() as tempdir:
        cache_fn = Path(tempdir) / "file_list.json"
        cache = {'100139613': 'file_1', '100139351': 'file_2'}
        with open(cache_fn, 'w') as f:
            json.dump(cache, f)
        downloader = MadrigalTecDownloader(tempdir, 'gstarr', 'gstarr@bu.edu', 'bu')
        download_dict = downloader._get_file_list(start_date, end_date)
        assert cache == download_dict


def test_verify_download():
    with TemporaryDirectory() as tempdir:
        server_files = ["/opt/cedar3/experiments3/2009/gps/31dec09/gps091231g.001.hdf5"]
        downloader = MadrigalTecDownloader(tempdir, 'gstarr', 'gstarr@bu.edu', 'bu')
        local_files = downloader._download_files(server_files)
        bad_server_files = downloader._verify_files(local_files, server_files)
        assert len(bad_server_files) == 0
        with open(local_files[0], 'rb+') as f:
            f.write('random extra stuff'.encode())
        bad_server_files = downloader._verify_files(local_files, server_files)
        assert bad_server_files == server_files


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


@pytest.mark.parametrize(
    ['dt', 'mlt_bins', 'mlat_bins'],
    itertools.product(
        [np.timedelta64(10, 'm'), np.timedelta64(30, 'm'), np.timedelta64(1, 'h'), np.timedelta64(2, 'h')],
        [config.get_mlt_bins(), np.arange(10)],
        [config.get_mlat_bins(), np.arange(10)]
    )
)
def test_process_tec(download_dir, process_dir, test_dates, dt, mlt_bins, mlat_bins):
    start, end = test_dates
    correct_times = np.arange(np.datetime64(start, 's'), np.datetime64(end, 's') + dt, dt)
    mlt_vals = (mlt_bins[:-1] + mlt_bins[1:]) / 2
    mlat_vals = (mlat_bins[:-1] + mlat_bins[1:]) / 2
    processed_file = Path(process_dir) / 'tec_test.nc'
    for hemisphere in ['north', 'south']:
        process_interval(start, end, hemisphere, processed_file, download_dir, dt, mlat_bins, mlt_bins)
        assert processed_file.exists()
        data = xr.open_dataarray(processed_file)
        data.load()
        h = 1 if hemisphere == 'north' else -1
        try:
            assert data.shape == (correct_times.shape[0], mlat_vals.shape[0], mlt_vals.shape[0])
            assert (data.mlt == mlt_vals).all().item()
            assert (data.mlat == h * mlat_vals).all().item()
            assert (data.time == correct_times).all().item()
        finally:
            data.close()
            processed_file.unlink()


def test_process_tec_out_of_range(download_dir, process_dir, test_dates):
    dt = np.timedelta64(1, 'h')
    start, end = [date - timedelta(days=100) for date in test_dates]
    processed_file = Path(process_dir) / 'tec_test.nc'
    with pytest.raises(InvalidProcessDates):
        process_interval(start, end, 'north', processed_file, download_dir, dt, config.get_mlat_bins(), config.get_mlt_bins())


def test_get_tec_data(download_dir, process_dir, test_dates):
    start, end = test_dates
    dt = np.timedelta64(1, 'h')
    mlt_bins = config.get_mlt_bins()
    mlat_bins = config.get_mlat_bins()
    mlt_vals = (mlt_bins[:-1] + mlt_bins[1:]) / 2
    mlat_vals = (mlat_bins[:-1] + mlat_bins[1:]) / 2
    correct_times = np.arange(np.datetime64(start), np.datetime64(end) + dt, dt)
    for hemisphere in ['north', 'south']:
        processed_file = get_tec_paths(start, end, hemisphere, process_dir)[0]
        process_interval(start, end, hemisphere, processed_file, download_dir, dt, mlat_bins, mlt_bins)
        data = get_tec_data(start, end, hemisphere, process_dir)
        h = 1 if hemisphere == 'north' else -1
        assert data.shape == (correct_times.shape[0], mlat_vals.shape[0], mlt_vals.shape[0])
        assert (data.mlt == mlt_vals).all().item()
        assert (data.mlat == h * mlat_vals).all().item()
        assert (data.time == correct_times).all().item()


def test_scripts(test_dates):
    start, end = test_dates
    with TemporaryDirectory() as base_dir:
        with config.temp_config(base_dir=base_dir) as cfg:
            scripts.download_tec(start, end)
            tec_files = list(Path(cfg.download_tec_dir).glob('*'))
            assert len(tec_files) > 0
            data = _get_downloaded_tec_data(start, end, cfg.download_tec_dir)
            assert data.time.values[0] < np.datetime64(test_dates[0], 's')
            assert data.time.values[-1] > np.datetime64(test_dates[-1], 's')
            scripts.process_tec(start, end)

            for hemisphere in ['north', 'south']:
                data = get_tec_data(start, end, hemisphere, cfg.processed_tec_dir)
                data.load()
                dt = np.timedelta64(1, 'h')
                mlt_vals = config.get_mlt_vals()
                mlat_vals = config.get_mlat_vals()
                correct_times = np.arange(np.datetime64(test_dates[0]), np.datetime64(test_dates[-1]) + dt, dt)
                h = 1 if hemisphere == 'north' else -1
                assert data.shape == (correct_times.shape[0], mlat_vals.shape[0], mlt_vals.shape[0])
                assert (data.mlt == mlt_vals).all().item()
                assert (data.mlat == h * mlat_vals).all().item()
                assert (data.time == correct_times).all().item()
