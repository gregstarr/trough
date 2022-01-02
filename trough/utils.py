import numpy as np
import datetime


def datetime64_to_timestamp(dt64):
    """Convert single / array of numpy.datetime64 to timestamps (seconds since epoch)

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    timestamp: numpy.ndarray[float]
    """
    return (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')


def datetime64_to_datetime(dt64):
    """Convert single datetime64 to datetime

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    list[datetime]
    """
    ts = datetime64_to_timestamp(dt64)
    if isinstance(ts, np.ndarray):
        return [datetime.datetime.utcfromtimestamp(t) for t in ts]
    return datetime.datetime.utcfromtimestamp(ts)


def decompose_datetime64(dt64):
    """Convert array of np.datetime64 to an array (N x 3) of year, month (jan=1), day (1 index)

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    idx: numpy.ndarray (N x 3)
    """
    year_floor = dt64.astype('datetime64[Y]')
    month_floor = dt64.astype('datetime64[M]')

    year = year_floor.astype(int) + 1970
    month = (dt64.astype('datetime64[M]') - year_floor).astype(int) + 1
    day = (dt64.astype('datetime64[D]') - month_floor).astype(int) + 1

    return np.column_stack((year, month, day))


def no_ext_fn(fn):
    """return name of file with no path or extension

    Parameters
    ----------
    fn: str

    Returns
    -------
    str
    """
    return os.path.splitext(os.path.basename(fn))[0]


def centered_bn_func(func, arr, window_diameter, pad=False, **kwargs):
    """Call a centered bottleneck moving window function on an array, optionally padding with the edge values to keep
    the same shape. Window moves through axis 0.

    Parameters
    ----------
    func: bottleneck moving window function
    arr: numpy.ndarray
    window_diameter: int
        odd number window width
    pad: bool
        whether to pad to keep same shape or not
    kwargs
        passed to func

    Returns
    -------
    numpy.ndarray
    """
    window_radius = window_diameter // 2
    assert (2 * window_radius + 1) == window_diameter, "window_diameter must be odd"
    if pad:
        pad_tuple = ((window_radius, window_radius), ) + ((0, 0), ) * (arr.ndim - 1)
        arr = np.pad(arr, pad_tuple, mode='edge')
    return func(arr, window_diameter, **kwargs)[2 * window_radius:]


def moving_func_trim(window_diameter, *arrays):
    """Trim any number of arrays to valid dimension after calling a centered bottleneck moving window function

    Parameters
    ----------
    window_diameter: int
        odd number window width
    arrays: 1 or more numpy.ndarray

    Returns
    -------
    tuple of numpy.ndarrays
    """
    window_radius = window_diameter // 2
    assert (2 * window_radius + 1) == window_diameter, "window_diameter must be odd"
    if window_radius == 0:
        return (array for array in arrays)
    return (array[window_radius:-window_radius] for array in arrays)


def extract_patches(arr, patch_shape, step=1):
    """Assuming `arr` is 3D (time, lat, lon). `arr` will be padded, then have patches extracted using
    `skimage.util.view_as_windows`. The padding will be "edge" for lat, and "wrap" for lon, with no padding for
    time. Returned array will have same lat and lon dimension length as input and a different time dimension length
    depending on `patch_shape`.

    Parameters
    ----------
    arr: numpy.ndarray
        must be 3 dimensional
    patch_shape: tuple
        must be length 3
    step: int
    Returns
    -------
    patches view of padded array
        shape (arr.shape[0] - patch_shape[0] + 1, arr.shape[1], arr.shape[2]) + patch_shape
    """
    assert arr.ndim == 3 and len(patch_shape) == 3, "Invalid input args"
    # lat padding
    padded = np.pad(arr, ((0, 0), (patch_shape[1] // 2, patch_shape[1] // 2), (0, 0)), 'edge')
    # lon padding
    padded = np.pad(padded, ((0, 0), (0, 0), (patch_shape[2] // 2, patch_shape[2] // 2)), 'wrap')
    patches = view_as_windows(padded, patch_shape, step)
    return patches


def write_h5(fn, **kwargs):
    """Writes an h5 file with data specified by kwargs.

    Parameters
    ----------
    fn: str
        file path to write
    **kwargs
    """
    with h5py.File(fn, 'w') as f:
        for key, value in kwargs.items():
            f.create_dataset(key, data=value)
