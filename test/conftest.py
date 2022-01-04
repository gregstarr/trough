import pytest

import trough


def pytest_addoption(parser):
    parser.addoption("--skip-ftp", action='store_true', default=False)


@pytest.fixture(autouse=True)
def skip_ftp(request):
    return request.config.getoption("--skip-ftp")