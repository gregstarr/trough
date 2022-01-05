from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import trough


def test_process_omni(test_dates):
    start_date, end_date = test_dates
    with TemporaryDirectory() as tempdir:
        with trough.config.temp_config(processed_omni_dir=tempdir) as cfg:
            trough.scripts.process_omni(start_date, end_date)
            processed_file = Path(cfg.processed_omni_dir) / 'omni.h5'
            assert processed_file.exists()
            data = trough.get_omni_data(datetime(2021, 1, 1), datetime(2021, 1, 2))
            assert data.index[0] == start_date
            assert data.index[-1] == end_date


def test_process_arb():
    pytest.skip("not finished")
    start_date = datetime(2021, 1, 1, 6, 0, 0)
    end_date = datetime(2021, 1, 1, 18, 0, 0)
    with TemporaryDirectory() as tempdir:
        with trough.config.temp_config(processed_arb_dir=tempdir) as cfg:
            trough.scripts.process_arb(start_date, end_date)
            processed_file = Path(cfg.processed_omni_dir) / 'omni.h5'
            assert processed_file.exists()
            data = trough.get_omni_data(datetime(2021, 1, 1), datetime(2021, 1, 2))
            assert data.index[0] == start_date
            assert data.index[-1] == end_date


def test_process_tec():
    pytest.skip("not finished")
    start_date = datetime(2021, 1, 1, 6, 0, 0)
    end_date = datetime(2021, 1, 1, 18, 0, 0)
    with TemporaryDirectory() as tempdir:
        with trough.config.temp_config(processed_omni_dir=tempdir) as cfg:
            trough.scripts.process_omni(start_date, end_date)
            processed_file = Path(cfg.processed_omni_dir) / 'omni.h5'
            assert processed_file.exists()
            data = trough.get_omni_data(datetime(2021, 1, 1), datetime(2021, 1, 2))
            assert data.index[0] == start_date
            assert data.index[-1] == end_date
