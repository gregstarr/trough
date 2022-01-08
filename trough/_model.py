import numpy as np
import apexpy
from scipy.interpolate import interp1d
import datetime

import trough._aux_data as trough_data


def get_model(ut, mlt_vals):
    """Get magnetic latitudes of the trough according to the model in Deminov 2017
    for a specific time and set of magnetic local times.

    Parameters
    ----------
    ut, mlt: numpy.ndarray
    Returns
    -------
    mlat: numpy.ndarray (n_ut, n_mlt)
        model evaluated at the given magnetic local times
    """
    if np.issubdtype(ut.dtype, np.datetime64):
        ut = ut.copy().astype('datetime64[s]').astype(int)
    kp = _get_weighted_kp(ut)
    apex = apexpy.Apex(date=datetime.datetime.fromtimestamp(ut[0]))
    mlat = 65.5 * np.ones((ut.shape[0], mlt_vals.shape[0]))
    for i in range(10):
        glat, glon = apex.convert(mlat, mlt_vals[None, :], 'mlt', 'geo', 350, ut[:, None])
        mlat = _model_subroutine_lat(mlt_vals[None, :], glon, kp[:, None])
    return mlat


def _model_subroutine_lat(mlt, glon, kp):
    """Get's model output mlat given MLT, geographic lon and weighted kp

    Parameters
    ----------
    mlt: numpy.ndarray (n_mlt, )
    glon: numpy.ndarray (n_mlt, )
    kp: float

    Returns
    -------
    mlat: numpy.ndarray (n_t, n_mlt)
    """
    phi_t = 3.16 - 5.6 * np.cos(np.deg2rad(15 * (mlt - 2.4))) + 1.4 * np.cos(np.deg2rad(15 * (2 * mlt - .8)))
    phi_lon = .85 * np.cos(np.deg2rad(glon + 63)) - .52 * np.cos(np.deg2rad(2 * glon + 5))
    return 65.5 - 2.4 * kp + phi_t + phi_lon * np.exp(-.3 * kp)


def _get_weighted_kp(ut, tau=.6, T=10):
    """Get a weighed sum of kp values over time. See paper for details.
    """
    omni = trough_data.get_omni_data(ut[0].astype('datetime64[s]'), ut[-1].astype('datetime64[s]'))
    ap = omni['ap'].values
    times = np.array(omni.index.values.astype(float) / 1e9, dtype=int)
    prehistory = np.column_stack([ap[T - i - 1:ap.shape[0] - i] for i in range(T)])
    weight_factors = tau ** np.arange(T)
    ap_tau = np.sum((1 - tau) * prehistory * weight_factors, axis=1)
    kp_tau = 2.1 * np.log(.2 * ap_tau + 1)
    times = times[T - 1:]
    kp = interp1d(times, kp_tau, kind='previous')
    return kp(ut)
