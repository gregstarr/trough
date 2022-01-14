import numpy as np
import xarray as xr
from apexpy import Apex
from scipy.stats import binned_statistic_2d
from multiprocessing import Pool
from pathlib import Path
import h5py
import logging
import functools
from datetime import datetime, timedelta

from trough import config, utils
from trough.exceptions import InvalidProcessDates


logger = logging.getLogger(__name__)


def _parse_madrigal_fn(path):
    date = datetime.strptime(path.name[3:9], "%y%m%d")
    return date


def _get_downloaded_tec_data(start_date, end_date, input_dir):
    start_date -= timedelta(hours=3)
    end_date += timedelta(hours=3)
    data = []
    for path in Path(input_dir).glob('*.hdf5'):
        date1 = _parse_madrigal_fn(path)
        date2 = date1 + timedelta(days=1)
        if start_date > date2 or end_date < date1:
            continue
        data.append(open_madrigal_file(path))
        logger.info(f"arb file: {path}, info: [{date1}, {date2}], tec: {data[-1]}")
    if len(data) == 0:
        return xr.DataArray()
    return xr.concat(data, 'time')


def open_madrigal_file(fn):
    """Open a madrigal file, return its data
    """
    with h5py.File(fn, 'r') as f:
        tec = f['Data']['Array Layout']['2D Parameters']['tec'][()]
        timestamps = f['Data']['Array Layout']['timestamps'][()]
        lat = f['Data']['Array Layout']['gdlat'][()]
        lon = f['Data']['Array Layout']['glon'][()]
    logger.info(f"Opened madrigal file: {fn}, size: {tec.shape}")
    return xr.DataArray(
        np.moveaxis(tec, -1, 0),
        coords={
            'time': timestamps.astype('datetime64[s]'),
            'glat': lat,
            'glon': lon,
        },
        dims=['time', 'glat', 'glon']
    )


def get_tec_paths(start_date, end_date, processed_dir):
    file_dates = np.arange(
        np.datetime64(start_date, 'M'),
        np.datetime64(end_date, 'M') + 1,
        np.timedelta64(1, 'M')
    )
    file_dates = utils.decompose_datetime64(file_dates)
    return [Path(processed_dir) / f"tec_{d[0]:04d}_{d[1]:02d}.nc" for d in file_dates]


def get_tec_data(start_date, end_date, processed_dir):
    data = xr.concat([xr.open_dataarray(file) for file in get_tec_paths(start_date, end_date, processed_dir)], 'time')
    return data.sel(time=slice(start_date, end_date))


def calculate_bins(data, mlat_bins, mlt_bins):
    """Calculates TEC in MLAT - MLT bins. Executed in process pool.
    """
    if data.shape == ():
        tec = np.ones((mlat_bins.shape[0] - 1, mlt_bins.shape[0] - 1)) * np.nan
    else:
        mask = np.isfinite(data.values)
        tec = binned_statistic_2d(
            np.broadcast_to(data.mlat, data.shape)[mask],
            data.mlt.values[mask],
            data.values[mask],
            statistic='mean',
            bins=[mlat_bins, mlt_bins]
        ).statistic
    return tec


def get_mag_coords(apex, mad_data):
    mlat, mlt = apex.convert(mad_data.glat, mad_data.glon, 'geo', 'mlt', height=350,
                             datetime=mad_data.time.values[:, None, None])
    mlt[mlt > 12] -= 24
    return xr.DataArray(
        mad_data.values,
        coords={
            'time': mad_data.time,
            'glat': mad_data.glat,
            'glon': mad_data.glon,
            'mlat': xr.DataArray(mlat, coords={'glat': mad_data.glat, 'glon': mad_data.glon}),
            'mlt': xr.DataArray(mlt, coords={'time': mad_data.time, 'glat': mad_data.glat, 'glon': mad_data.glon}),
        },
        dims=['time', 'glat', 'glon']
    )


def process_interval(start_date, end_date, output_fn, input_dir, sample_dt, mlat_bins, mlt_bins):
    """Processes an interval of madrigal data and writes to files.
    """
    logger.info(f"processing tec data for {start_date, end_date}")
    calc_bins = functools.partial(calculate_bins, mlat_bins=mlat_bins, mlt_bins=mlt_bins)
    ref_times = np.arange(np.datetime64(start_date, 's'), np.datetime64(end_date, 's'), sample_dt)
    mad_data = _get_downloaded_tec_data(start_date, end_date, input_dir)
    if mad_data.shape == () or min(mad_data.time.values) > ref_times[0] or max(mad_data.time.values) < ref_times[-1]:
        logger.error(f"mad_data shape: {mad_data.shape}")
        if mad_data.shape != ():
            logger.error(f"times: {min(mad_data.time.values)} - {max(mad_data.time.values)}")
        raise InvalidProcessDates(f"Need to download full data range before processing")
    mad_data = mad_data.sel(time=slice(ref_times[0], ref_times[-1] + sample_dt))

    logger.info("Converting coordinates")
    apex = Apex(date=start_date)
    mad_data = get_mag_coords(apex, mad_data)

    logger.info("Setting up for binning")
    time_bins = np.arange(mad_data.time.values[0], mad_data.time.values[-1] + sample_dt, sample_dt)
    data_groups = mad_data.groupby_bins('time', bins=time_bins, right=False)
    data = [_data for _interval, _data in data_groups]
    logger.info("Calculated bins")
    tec = xr.DataArray(
        np.array([result for result in map(calc_bins, data)]),
        coords={
            'time': np.array([_interval.left for _interval, _data in data_groups]),
            'mlat': (mlat_bins[:-1] + mlat_bins[1:]) / 2,
            'mlt': (mlt_bins[:-1] + mlt_bins[1:]) / 2,
        },
        dims=['time', 'mlat', 'mlt']
    )
    tec.to_netcdf(output_fn)


def process_tec_dataset(start_date, end_date, download_dir=None, process_dir=None, dt=None, mlat_bins=None,
                        mlt_bins=None):
    if download_dir is None:
        download_dir = config.download_tec_dir
    if process_dir is None:
        process_dir = config.processed_tec_dir
    if mlt_bins is None:
        mlt_bins = config.get_mlt_bins()
    if mlat_bins is None:
        mlat_bins = config.get_mlat_bins()
    if dt is None:
        dt = np.timedelta64(1, 'h')
    Path(process_dir).mkdir(exist_ok=True, parents=True)

    for year in range(start_date.year, end_date.year + 1):
        for month in range(1, 13):
            start = datetime(year, month, 1)
            end = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
            if start > end_date or end < start_date:
                continue
            start = max(start_date, start)
            end = min(end_date, end)
            process_interval(start, end, Path(process_dir) / f"tec_{year:04d}_{month:02d}.nc", download_dir, dt,
                             mlat_bins, mlt_bins)
