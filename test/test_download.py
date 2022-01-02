from datetime import datetime
from pathlib import Path
import tempfile
import h5py

import trough


def test_download_tec():
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 1, 1)
    user_data = {'madrigal_user_name': 'gstarr', 'madrigal_user_email': 'gstarr@bu.edu', 'madrigal_user_affil': 'BU'}
    with tempfile.TemporaryDirectory() as tempdir:
        with trough.config.temp_config(base_dir=tempdir, **user_data) as cfg:
            trough.scripts.download_tec(start_date, end_date)
            tec_files = list(Path(cfg.download_tec_dir).glob('*'))
            tec_file_names = [file.stem[:-4] for file in Path(cfg.download_tec_dir).glob('*')]
        assert len(tec_file_names) == 2
        assert 'gps201231g' in tec_file_names
        assert 'gps210101g' in tec_file_names
        for tec_file in tec_files:
            with h5py.File(tec_file) as f:
                assert 'tec' in f['Data/Array Layout/2D Parameters']


def test_download_omni():
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 1, 1)
    with tempfile.TemporaryDirectory() as tempdir:
        with trough.config.temp_config(base_dir=tempdir) as cfg:
            trough.scripts.download_omni(start_date, end_date)
            omni_files = list(Path(cfg.download_omni_dir).glob('*'))
        print()


def test_download_arb():
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 1, 1)
    with tempfile.TemporaryDirectory() as tempdir:
        with trough.config.temp_config(base_dir=tempdir) as cfg:
            trough.scripts.download_arb(start_date, end_date)
            arb_files = list(Path(cfg.download_arb_dir).glob('*'))
        print()

