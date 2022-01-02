

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