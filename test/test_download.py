from datetime import datetime
from pathlib import Path
import tempfile
import h5py
import pytest

import trough


def test_download_tec(test_dates):
    user_data = {'madrigal_user_name': 'gstarr', 'madrigal_user_email': 'gstarr@bu.edu', 'madrigal_user_affil': 'BU'}
    with tempfile.TemporaryDirectory() as tempdir:
        with trough.config.temp_config(base_dir=tempdir, **user_data) as cfg:
            trough.scripts.download_tec(*test_dates)
            tec_files = list(Path(cfg.download_tec_dir).glob('*'))
            tec_file_names = [file.stem[:-4] for file in Path(cfg.download_tec_dir).glob('*')]
        assert len(tec_file_names) == 2
        assert 'gps201231g' in tec_file_names
        assert 'gps210101g' in tec_file_names
        for tec_file in tec_files:
            with h5py.File(tec_file) as f:
                assert 'tec' in f['Data/Array Layout/2D Parameters']


@pytest.mark.parametrize('method', ['ftp', 'http'])
def test_download_omni(skip_ftp, method, test_dates):
    if method == 'ftp' and skip_ftp:
        pytest.skip("Skipping because 'skip_ftp' set to True")
    with tempfile.TemporaryDirectory() as tempdir:
        with trough.config.temp_config(base_dir=tempdir, nasa_spdf_download_method=method) as cfg:
            trough.scripts.download_omni(*test_dates)
            omni_files = list(Path(cfg.download_omni_dir).glob('*'))
        assert len(omni_files) == 1
        assert omni_files[0].name == 'omni2_2021.dat'


@pytest.mark.parametrize('method', ['ftp', 'http'])
def test_download_arb(skip_ftp, method, test_dates):
    if method == 'ftp' and skip_ftp:
        pytest.skip("Skipping because 'skip_ftp' set to True")
    with tempfile.TemporaryDirectory() as tempdir:
        with trough.config.temp_config(base_dir=tempdir, nasa_spdf_download_method=method) as cfg:
            trough.scripts.download_arb(*test_dates)
            arb_files = list(Path(cfg.download_arb_dir).glob('*'))
        assert len(arb_files) > 0
        for file in arb_files:
            assert file.suffix == '.nc'
            with h5py.File(file) as f:
                assert 'MODEL_NORTH_GEOGRAPHIC_LATITUDE' in f.keys()
