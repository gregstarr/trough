import pytest
import logging
from tempfile import TemporaryDirectory
from datetime import datetime

from trough import config


logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--skip-ftp", action='store_true', default=False)


@pytest.fixture(scope='session', autouse=True)
def skip_ftp(request):
    return request.config.getoption("--skip-ftp")


@pytest.fixture(scope='session', autouse=True)
def setup_cfg(skip_ftp):
    logger.info("setting up config")
    with TemporaryDirectory() as tempdir:
        config.set_base_dir(tempdir)
        config.madrigal_user_affil = 'bu'
        config.madrigal_user_email = 'gstarr@bu.edu'
        config.madrigal_user_name = 'gregstarr'
        if skip_ftp:
            config.nasa_spdf_download_method = 'http'
        logger.info(f"base directory: {tempdir}")
        yield
    logger.info(f'resetting config: {config.download_omni_dir}')


@pytest.fixture(scope='session')
def test_dates():
    start_date = datetime(2021, 1, 3, 6, 0, 0)
    end_date = datetime(2021, 1, 3, 12, 0, 0)
    yield start_date, end_date
