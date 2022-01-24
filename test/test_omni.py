from tempfile import TemporaryDirectory
from pathlib import Path
import xarray as xr
import numpy as np
import pytest

from trough._omni import open_downloaded_omni_file, process_omni_dataset
from trough._download import OmniDownloader


def test_verify_download():
    with TemporaryDirectory() as tempdir:
        server_files = ["/pub/data/omni/low_res_omni/omni2_2009.dat"]
        downloader = OmniDownloader(tempdir)
        local_files = downloader._download_files(server_files)
        bad_server_files = downloader._verify_files(local_files, server_files)
        assert len(bad_server_files) == 0
        with open(local_files[0], 'w') as f:
            f.write('random')
        bad_server_files = downloader._verify_files(local_files, server_files)
        assert bad_server_files == server_files


def test_download_omni_ftp(skip_ftp, test_dates):
    if skip_ftp:
        pytest.skip("Skipping because 'skip_ftp' set to True")
    with TemporaryDirectory() as tempdir:
        downloader = OmniDownloader(tempdir, 'ftp')
        downloader.download(*test_dates)
        omni_files = list(Path(tempdir).glob('*.dat'))
        assert len(omni_files) > 0
        data = []
        for file in omni_files:
            data.append(open_downloaded_omni_file(file))
        assert min([d.time.values[0] for d in data]) < np.datetime64(test_dates[0], 's')
        assert max([d.time.values[-1] for d in data]) > np.datetime64(test_dates[-1], 's')


@pytest.fixture(scope='module')
def tempdir():
    with TemporaryDirectory() as t:
        yield t


def test_download_omni_http(test_dates, tempdir):
    downloader = OmniDownloader(tempdir, 'http')
    downloader.download(*test_dates)
    omni_files = list(Path(tempdir).glob('*.dat'))
    assert len(omni_files) > 0
    data = []
    for file in omni_files:
        data.append(open_downloaded_omni_file(file))
    assert min([d.time.values[0] for d in data]) < np.datetime64(test_dates[0], 's')
    assert max([d.time.values[-1] for d in data]) > np.datetime64(test_dates[-1], 's')


def test_process_omni(tempdir, test_dates):
    start, end = test_dates
    processed_file = Path(tempdir) / 'omni_test.nc'
    process_omni_dataset(tempdir, processed_file)
    assert processed_file.exists()
    omni_data = xr.open_dataset(processed_file)
    assert omni_data.time.values[0] < np.datetime64(start, 's')
    assert omni_data.time.values[-1] > np.datetime64(end, 's')
