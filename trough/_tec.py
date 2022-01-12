import numpy as np
import apexpy
from scipy.stats import binned_statistic_2d
from multiprocessing import Pool
from pathlib import Path
import h5py
import logging
import functools
from datetime import datetime

import trough.utils as trough_utils
from trough import config


logger = logging.getLogger(__name__)


def get_madrigal_data(start_date, end_date):
    """Gets madrigal TEC and timestamps assuming regular sampling. Fills in missing time steps.
    """
    mad_lon, mad_lat = np.arange(-180, 180), np.arange(-90, 90)
    dt = np.timedelta64(5, 'm')
    dt_sec = dt.astype('timedelta64[s]').astype(int)
    start_date = (np.ceil(start_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    end_date = (np.ceil(end_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[s]').astype(int)
    tec = np.ones((mad_lat.shape[0], mad_lon.shape[0], ref_times_ut.shape[0])) * np.nan
    file_dates = np.unique(ref_times.astype('datetime64[D]'))
    file_dates = trough_utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        d = file_dates[i, 2]
        try:
            fn = list(Path(config.download_tec_dir).glob(f"gps{y - 2000:02d}{m:02d}{d:02d}g.*.hdf5"))[-1]
        except IndexError:
            logger.warning(f"{y}-{m}-{d} madrigal file doesn't exist")
            continue
        t, ut, lat, lon = open_madrigal_file(fn)
        month_time_mask = np.in1d(ref_times_ut, ut)
        day_time_mask = np.in1d(ut, ref_times_ut)
        if not (np.all(lat == mad_lat) and np.all(lon == mad_lon)):
            logger.warning(f"THIS FILE HAS MISSING DATA!!!!!!! {fn}")
            lat_ind = np.argwhere(np.in1d(mad_lat, lat))[:, 0]
            lon_ind = np.argwhere(np.in1d(mad_lon, lon))[:, 0]
            time_ind = np.argwhere(month_time_mask)[:, 0]
            lat_grid_ind, lon_grid_ind, time_grid_ind = np.meshgrid(lat_ind, lon_ind, time_ind)
            tec[lat_grid_ind.ravel(), lon_grid_ind.ravel(), time_grid_ind.ravel()] = t[:, :, day_time_mask].ravel()
        else:
            # assume ut is increasing and has no repeating entries, basically that it is a subset of ref_times_ut
            tec[:, :, month_time_mask] = t[:, :, day_time_mask]
    return np.moveaxis(tec, -1, 0), ref_times


def open_madrigal_file(fn):
    """Open a madrigal file, return its data
    """
    with h5py.File(fn, 'r') as f:
        tec = f['Data']['Array Layout']['2D Parameters']['tec'][()]
        timestamps = f['Data']['Array Layout']['timestamps'][()]
        lat = f['Data']['Array Layout']['gdlat'][()]
        lon = f['Data']['Array Layout']['glon'][()]
    print(f"Opened madrigal file: {fn}, size: {tec.shape}")
    return tec, timestamps, lat, lon


def get_tec_data(start_date, end_date):
    """Gets TEC and timestamps
    """
    tec = []
    ut = []
    file_dates = np.arange(
        start_date.astype('datetime64[M]'),
        end_date.astype('datetime64[M]') + 1,
        np.timedelta64(1, 'M')
    )
    file_dates = trough_utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        fn = Path(config.processed_tec_dir) / f"tec_{y:04d}_{m:02d}.h5"
        with h5py.File(fn, 'r') as f:
            tec.append(f['tec'][()])
            ut.append(f['times'][()])
    tec = np.concatenate(tec, axis=0)
    times = np.concatenate(ut, axis=0).astype('datetime64[s]')
    mask = (times >= start_date) & (times < end_date)
    return tec[mask], times[mask]


def assemble_binning_args(mlat, mlt, tec, times, map_period=np.timedelta64(1, 'h')):
    """Creates a list of tuple arguments to be passed to `calculate_bins`. `calculate_bins` is called by the process
    pool manager using each tuple in the list returned by this function as arguments. Each set of arguments corresponds
    to one processed TEC map and should span a time period specified by `map_period`. `map_period` should evenly divide
    24h, and probably should be a multiple of 5 min.

    Parameters
    ----------
    mlat, mlt, tec, times: numpy.ndarray[float]
    map_period: {np.timedelta64, int}

    Returns
    -------
    list[tuple]
        each tuple in the list is passed to `calculate_bins`
    """
    if isinstance(map_period, np.timedelta64):
        map_period = map_period.astype('timedelta64[s]').astype(int)
    args = []
    current_time = times[0]
    while current_time < times[-1]:
        start = np.argmax(times >= current_time)
        end = np.argmax(times >= current_time + map_period)
        if end == 0:
            end = times.shape[0]
        time_slice = slice(start, end)
        fin_mask = np.isfinite(tec[time_slice])
        mlat_r = mlat[time_slice][fin_mask].copy()
        mlt_r = mlt[time_slice][fin_mask].copy()
        tec_r = tec[time_slice][fin_mask].copy()
        args.append((mlat_r, mlt_r, tec_r, times[time_slice]))
        current_time += map_period
    return args


def calculate_bins(mlat, mlt, tec, times, bins):
    """Calculates TEC in MLAT - MLT bins. Executed in process pool.

    Parameters
    ----------
    mlat, mlt, tec, times, ssmlon: numpy.ndarray[float] (N, )
    bins: list[numpy.ndarray[float] (X + 1, ), numpy.ndarray[float] (Y + 1, )]

    Returns
    -------
    tuple
        time: int or float
        final_tec, final_tec_n
    """
    if tec.size == 0:
        final_tec = np.ones((bins[0].shape[0] - 1, bins[1].shape[0] - 1)) * np.nan
    else:
        final_tec = binned_statistic_2d(mlat, mlt, tec, 'mean', bins).statistic
    return times[0], final_tec


def get_mag_grid(converter):
    lon_grid, lat_grid = np.meshgrid(np.arange(-180, 180), np.arange(-90, 90))
    mlat, mlon = converter.convert(lat_grid.ravel(), lon_grid.ravel(), 'geo', 'apex', height=350)
    mlat = mlat.reshape(lat_grid.shape)
    mlon = mlon.reshape(lat_grid.shape)
    return mlat, mlon


def process_interval(start_date, end_date):
    """Processes an interval of madrigal data and writes to files.
    """
    logger.info(f"Processing interval: {start_date, end_date}")
    apex = apexpy.Apex(trough_utils.datetime64_to_datetime(start_date))
    mlat, mlon = get_mag_grid(apex)
    tec, ts = get_madrigal_data(start_date, end_date)
    logger.info("Converting coordinates")
    mlt = apex.mlon2mlt(mlon[None, :, :], ts[:, None, None])
    mlat = mlat[None, :, :] * np.ones((ts.shape[0], 1, 1))
    mlt[mlt > 12] -= 24
    logger.info("Setting up for binning")
    args = assemble_binning_args(mlat, mlt, tec, ts)
    logger.info(f"Calculating bins for {len(args)} time steps")
    calc_bins = functools.partial(calculate_bins, bins=[config.get_mlat_bins(), config.get_mlt_bins()])
    with Pool() as p:
        pool_result = p.starmap(calc_bins, args)
    logger.info("Calculated bins")
    times = np.array([r[0] for r in pool_result])
    tec = np.array([r[1] for r in pool_result])

    dd = trough_utils.decompose_datetime64(start_date)
    output_fn = Path(config.processed_tec_dir) / f"tec_{dd[0, 0]:04d}_{dd[0, 1]:02d}.h5"
    trough_utils.write_h5(output_fn, times=times.astype('datetime64[s]').astype(int), tec=tec)


def _parse_madrigal_fn(path):
    date = datetime.strptime(path.name[3:9], "%y%m%d")
    return date


def process_tec_dataset():
    madrigal_dates = [_parse_madrigal_fn(path) for path in Path(config.download_tec_dir).glob("*.hdf5")]
    min_date = np.datetime64(min(madrigal_dates))
    max_date = np.datetime64(max(madrigal_dates))
    Path(config.processed_tec_dir).mkdir(exist_ok=True, parents=True)
    start_date = min_date.astype('datetime64[M]')
    end_date = max_date.astype('datetime64[M]')
    months = np.arange(start_date, end_date + 1, np.timedelta64(1, 'M'))
    for month in months:
        interval_start = max(min_date, month)
        interval_end = max(max_date, month + 1)
        process_interval(interval_start, interval_end)
