from tempfile import TemporaryDirectory
from pathlib import Path
import xarray as xr
import numpy as np
import pytest
import itertools
from datetime import datetime, timedelta

from trough import config, scripts
from trough._arb import process_interval, get_arb_data, get_arb_paths, _get_downloaded_arb_data
from trough._download import ArbDownloader
from trough.exceptions import InvalidProcessDates


def test_download_arb_ftp(skip_ftp, test_dates):
    if skip_ftp:
        pytest.skip("Skipping because 'skip_ftp' set to True")
    with TemporaryDirectory() as tempdir:
        downloader = ArbDownloader(tempdir, 'ftp')
        downloader.download(*test_dates)
        arb_files = list(Path(tempdir).glob('*'))
        assert len(arb_files) > 0
        data, times = _get_downloaded_arb_data(*test_dates, tempdir)
        assert min(times) < test_dates[0]
        assert max(times) > test_dates[-1]


@pytest.fixture(scope='module')
def download_dir():
    with TemporaryDirectory() as t:
        yield t


@pytest.fixture(scope='module')
def processed_dir():
    with TemporaryDirectory() as t:
        yield t


def test_download_arb_http(test_dates, download_dir):
    downloader = ArbDownloader(download_dir, 'http')
    downloader.download(*test_dates)
    arb_files = list(Path(download_dir).glob('*'))
    assert len(arb_files) > 0
    data, times = _get_downloaded_arb_data(*test_dates, download_dir)
    assert min(times) < test_dates[0]
    assert max(times) > test_dates[-1]


@pytest.mark.parametrize(['dt', 'mlt_vals'], itertools.product([np.timedelta64(30, 'm'), np.timedelta64(1, 'h'), np.timedelta64(2, 'h')], [config.get_mlt_vals(), np.arange(10)]))
def test_process_arb(download_dir, processed_dir, test_dates, dt, mlt_vals):
    start, end = test_dates
    correct_times = np.arange(np.datetime64(start, 's'), np.datetime64(end, 's'), dt)
    processed_file = Path(processed_dir) / 'arb_test.nc'
    process_interval(start, end, processed_file, download_dir, mlt_vals, dt)
    assert processed_file.exists()
    data = xr.open_dataarray(processed_file)
    assert data.shape == (correct_times.shape[0], mlt_vals.shape[0])
    assert (data.mlt == mlt_vals).all().item()
    assert (data.time == correct_times).all().item()


def test_process_arb_out_of_range(download_dir, processed_dir, test_dates):
    dt = np.timedelta64(1, 'h')
    start, end = [date - timedelta(days=100) for date in test_dates]
    processed_file = Path(processed_dir) / 'arb_test.nc'
    with pytest.raises(InvalidProcessDates):
        process_interval(start, end, processed_file, download_dir, config.get_mlt_vals(), dt)


def test_get_arb_data(download_dir, processed_dir, test_dates):
    start, end = test_dates
    dt = np.timedelta64(1, 'h')
    mlt = config.get_mlt_vals()
    correct_times = np.arange(np.datetime64(start), np.datetime64(end), dt)
    processed_file = get_arb_paths(start, end, processed_dir)[0]
    process_interval(start, end, processed_file, download_dir, mlt, dt)
    data = get_arb_data(start, end, processed_dir)
    assert data.shape == (correct_times.shape[0], mlt.shape[0])
    assert (data.mlt == mlt).all().item()
    assert (data.time == correct_times).all().item()


def test_scripts(test_dates):
    with TemporaryDirectory() as base_dir:
        with config.temp_config(base_dir=base_dir) as cfg:
            scripts.download_arb(*test_dates)
            arb_files = list(Path(cfg.download_arb_dir).glob('*'))
            assert len(arb_files) > 0
            data, times = _get_downloaded_arb_data(*test_dates, cfg.download_arb_dir)
            assert min(times) < test_dates[0]
            assert max(times) > test_dates[-1]
            scripts.process_arb(*test_dates)
            data = get_arb_data(*test_dates, cfg.processed_arb_dir)
            data.load()
            dt = np.timedelta64(1, 'h')
            mlt = config.get_mlt_vals()
            correct_times = np.arange(np.datetime64(test_dates[0]), np.datetime64(test_dates[-1]), dt)
            assert data.shape == (correct_times.shape[0], mlt.shape[0])
            assert (data.mlt == mlt).all().item()
            assert (data.time == correct_times).all().item()
