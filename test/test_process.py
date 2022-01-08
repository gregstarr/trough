from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

import trough


def test_process_omni(test_dates, download_omni_data):
    start_date, end_date = test_dates
    with TemporaryDirectory() as tempdir:
        with trough.config.temp_config(processed_omni_dir=tempdir) as cfg:
            trough.scripts.process_omni(start_date, end_date)
            processed_file = Path(cfg.processed_omni_dir) / 'omni.h5'
            assert processed_file.exists()


def test_process_arb(test_dates, download_arb_data):
    start_date, end_date = test_dates
    with TemporaryDirectory() as tempdir:
        with trough.config.temp_config(processed_arb_dir=tempdir) as cfg:
            trough.scripts.process_arb(start_date, end_date)
            processed_file = Path(cfg.processed_arb_dir) / 'arb_2021.h5'
            assert processed_file.exists()


def test_process_tec(test_dates, download_tec_data):
    start_date, end_date = test_dates
    with TemporaryDirectory() as tempdir:
        with trough.config.temp_config(processed_tec_dir=tempdir) as cfg:
            trough.scripts.process_tec(start_date, end_date)
            processed_file = Path(cfg.processed_tec_dir) / 'tec_2021_01.h5'
            assert processed_file.exists()


def test_label_trough(test_dates, process_all_data):
    # pytest.skip("edfiogua")
    start_date, end_date = test_dates
    with TemporaryDirectory() as tempdir:
        with trough.config.temp_config(processed_labels_dir=tempdir) as cfg:
            trough.scripts.label_trough(start_date, end_date)
            processed_file = Path(cfg.processed_labels_dir) / 'tec_2021_01.h5'
            assert processed_file.exists()
