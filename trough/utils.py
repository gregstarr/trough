import numpy as np
import datetime
import warnings
import logging
import xarray as xr
try:
    import h5py
    from skimage.util import view_as_windows
except ImportError as imp_err:
    warnings.warn(f"Packages required for recreating dataset not installed: {imp_err}")


logger = logging.getLogger(__name__)


def datetime64_to_datetime(dt64):
    """Convert single datetime64 to datetime

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    list[datetime]
    """
    ts = dt64.astype('datetime64[s]').astype(float)
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
    """
    with h5py.File(fn, 'w') as f:
        for key, value in kwargs.items():
            f.create_dataset(key, data=value)


def get_data_checker(data_getter):

    def check(start, end, dt, hemisphere, processed_file):
        times = np.arange(np.datetime64(start), np.datetime64(end), dt).astype('datetime64[s]')
        if processed_file.exists():
            logger.info(f"processed file already exists {processed_file=}, checking...")
            try:
                data_check = data_getter(start, end, hemisphere, processed_file.parent)
                data_check = data_check.sel(time=times)
                has_missing_data = data_check.isnull().all(axis=[i for i in range(1, data_check.ndim)]).any()
                if not has_missing_data:
                    logger.info(f"downloaded data already processed {processed_file=}, checking...")
                    return False
            except KeyError:
                logger.info(f"processed file doesn't have the requested data")
            except Exception as e:
                logger.info(f"error reading processed file {processed_file=}: {e}, removing and reprocessing")
                processed_file.unlink()
        return True

    return check


def read_netcdfs(files, dim):
    """https://xarray.pydata.org/en/stable/user-guide/io.html#reading-multi-file-datasets
    """
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataarray(path) as ds:
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(files)
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined
