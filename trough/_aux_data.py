import numpy as np
import pandas

from trough import config

_omni_names = ['year', 'decimal_day', 'hour', 'bartels_rotation_number', 'id_imf', 'id_sw_plasma', 'imf_n_pts',
               'plasma_n_pts', 'avg_b_mag', 'mag_avg_b', 'lat_avg_b', 'lon_avg_b', 'bx_gse', 'by_gse', 'bz_gse',
               'by_gsm', 'bz_gsm', 'sigma_mag_b', 'sigma_b', 'sigma_bx', 'sigma_by', 'sigma_bz', 'proton_temperature',
               'proton_density', 'plasma_speed', 'plasma_flow_lon', 'plasma_flow_lat', 'na/np', 'flow_pressure',
               'sigma_t', 'sigma_n', 'sigma_v', 'sigma_phi_v', 'sigma_theta_v', 'sigma_na/np', 'electric_field',
               'plasma_beta', 'alfven_mach_number', 'kp', 'r', 'dst', 'ae', 'proton_flux_1', 'proton_flux_2',
               'proton_flux_4', 'proton_flux_10', 'proton_flux_30', 'proton_flux_60', 'flag', 'ap', 'f107', 'pc', 'al',
               'au', 'mach_number']
_omni_formats = ['i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'i', 'i',
                 'i', 'i', 'f', 'f', 'f', 'f', 'f', 'f', 'i', 'i', 'f', 'f', 'i', 'i', 'f']


def open_downloaded_omni_file(fn):
    data = np.loadtxt(fn, dtype={'names': _omni_names, 'formats': _omni_formats})
    dates = (data['year'] - 1970).astype('datetime64[Y]') + \
            (data['decimal_day'] - 1).astype('timedelta64[D]') + \
            data['hour'].astype('timedelta64[h]')
    data = pandas.DataFrame(data)
    data.index = dates
    return data.drop(columns=['year', 'hour', 'decimal_day'])


def get_arb_data(start_date, end_date, dt=np.timedelta64(1, 'h'), data_dir=None):
    """Gets auroral boundary mlat and timestamps

    Parameters
    ----------
    start_date, end_date: np.datetime64
    data_dir: str

    Returns
    -------
    arb_mlat, times: numpy.ndarray
    """
    if data_dir is None:
        data_dir = config.arb_dir
    dt_sec = dt.astype('timedelta64[s]').astype(int)
    start_date = (np.ceil(start_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    end_date = (np.ceil(end_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[s]').astype(int)
    arb_mlat = []
    uts = []
    file_dates = np.unique(ref_times.astype('datetime64[M]'))
    file_dates = utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        fn = os.path.join(data_dir, f"{y:04d}_{m:02d}_arb.h5")
        mlat, ut = open_arb_file(fn)
        arb_mlat.append(mlat)
        uts.append(ut)
    uts = np.concatenate(uts)
    arb_mlat = np.concatenate(arb_mlat, axis=0)
    int_arb_mlat = np.empty((ref_times.shape[0], config.mlt_vals.shape[0]))
    for i in range(config.mlt_vals.shape[0]):
        int_arb_mlat[:, i] = np.interp(ref_times_ut, uts, arb_mlat[:, i])
    return int_arb_mlat, ref_times


def open_arb_file(fn):
    """Open a monthly auroral boundary file, return its data

    Parameters
    ----------
    fn: str

    Returns
    -------
    arb_mlat, times: numpy.ndarray
    """
    with h5py.File(fn, 'r') as f:
        arb_mlat = f['mlat'][()]
        times = f['times'][()]
    print(f"Opened ARB file: {fn}, size: {arb_mlat.shape}")
    return arb_mlat, times


def prepare_auroral_boundary_dataset():
    ...
