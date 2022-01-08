import numpy as np
import pytest

import trough._tec as trough_tec


map_periods = [np.timedelta64(10, 'm'), np.timedelta64(30, 'm'), np.timedelta64(1, 'h'), np.timedelta64(2, 'h')]


@pytest.fixture
def times():
    yield np.datetime64('2010-01-01T00:00:00') + np.arange(100) * np.timedelta64(5, 'm')


@pytest.mark.parametrize('map_period', map_periods)
def test_assemble_args(times, map_period):
    mlat = np.arange(10)
    mlt = np.arange(10)
    mlt, mlat = np.meshgrid(mlt, mlat)
    mlat = mlat[None, :, :] * np.ones((times.shape[0], 1, 1))
    mlt = mlt[None, :, :] * np.ones((times.shape[0], 1, 1))
    tec = np.random.rand(*mlat.shape)
    args = trough_tec.assemble_binning_args(mlat, mlt, tec, times, map_period)
    assert len(args) == np.ceil((times[-1] - times[0]) / map_period)
    assert args[0][3][0] == times[0]
    assert args[-1][3][0] + map_period >= times[-1]
    assert args[-1][3][0] < times[-1]
    assert args[-1][3][-1] == times[-1]
    for i in range(len(args) - 1):
        assert args[i][3][-1] == args[i + 1][3][0] - np.timedelta64(5, 'm')


def test_calculate_bins():
    mlat = np.arange(10)[None, :, None] * np.ones((1, 1, 10))
    mlt = np.arange(10)[None, None, :] * np.ones((1, 10, 1))
    tec = np.zeros((1, 10, 10))
    tec[0, 0, 0] = 10
    tec[0, 0, -1] = 20
    tec[0, -1, 0] = 30
    times = np.ones(1) * np.nan
    be = np.array([-.5, 4.5, 9.5])
    bins = [be, be]
    out_t, out_tec = trough_tec.calculate_bins(mlat.ravel(), mlt.ravel(), tec.ravel(), times, bins)
    assert np.isnan(out_t)
    assert out_tec.shape == (2, 2)
    assert out_tec[0, 0] == 10 / 25
    assert out_tec[0, 1] == 20 / 25
    assert out_tec[1, 0] == 30 / 25
    assert out_tec[1, 1] == 0
