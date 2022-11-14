import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from apexpy import Apex
from scipy.interpolate import interp1d
import logging
import warnings

try:
    import h5py
except ImportError as imp_err:
    warnings.warn(f"Packages required for recreating dataset not installed: {imp_err}")

from trough import config, utils
from trough.exceptions import InvalidProcessDates

_arb_fields = [
    'YEAR', 'DOY', 'TIME', 'ALTITUDE',
    'MODEL_NORTH_GEOGRAPHIC_LATITUDE', 'MODEL_NORTH_GEOGRAPHIC_LONGITUDE',
    'MODEL_SOUTH_GEOGRAPHIC_LATITUDE', 'MODEL_SOUTH_GEOGRAPHIC_LONGITUDE'
]
logger = logging.getLogger(__name__)


def get_arb_paths(start_date, end_date, hemisphere, processed_dir):
    file_dates = np.arange(
        np.datetime64(start_date, 'Y'),
        (np.datetime64(end_date, 's')).astype('datetime64[Y]') + 1,
        np.timedelta64(1, 'Y')
    )
    file_dates = utils.decompose_datetime64(file_dates)
    return [Path(processed_dir) / f"arb_{hemisphere}_{d[0]:04d}.nc" for d in file_dates]


def get_arb_data(start_date, end_date, hemisphere, processed_dir=None):
    if processed_dir is None:
        processed_dir = config.processed_arb_dir
    data = utils.read_netcdfs(get_arb_paths(start_date, end_date, hemisphere, processed_dir), 'time')
    return data.sel(time=slice(start_date, end_date))


def parse_arb_fn(path):
    sat_name = path.name[36:39]
    date = datetime.strptime(path.name[67:75], "%Y%m%d")
    return sat_name, date


def _get_downloaded_arb_data(start_date, end_date, input_dir):
    start_date -= timedelta(days=1)
    end_date += timedelta(days=1)
    data = {field: [] for field in _arb_fields}
    data['sat'] = []
    for path in Path(input_dir).glob('*.NC'):
        sat_name, date1 = parse_arb_fn(path)
        date2 = date1 + timedelta(days=1)
        if start_date > date2 or end_date < date1:
            continue
        data['sat'].append(sat_name)
        with h5py.File(path, 'r') as f:
            for field in _arb_fields:
                data[field].append(f[field][()])
        logger.info(f"arb file: {path}, info: [{sat_name}, {date1}, {date2}], n_pts: {len(data['ALTITUDE'][-1])}")
    years = np.array(data['YEAR']) - 1970
    doys = np.array(data['DOY']) - 1
    seconds = np.array(data['TIME'])
    times = years.astype('datetime64[Y]') + doys.astype('timedelta64[D]') + seconds.astype('timedelta64[s]')
    return data, times


def process_interval(start_date, end_date, hemisphere, output_fn, input_dir, mlt_vals, sample_dt):
    logger.info(f"processing arb data for {start_date, end_date}")
    ref_times = np.arange(np.datetime64(start_date, 's'), np.datetime64(end_date, 's') + sample_dt, sample_dt)
    apex = Apex(date=start_date)
    arb_data, times = _get_downloaded_arb_data(start_date, end_date, input_dir)
    if times.size == 0 or min(times) > ref_times[0] or max(times) < ref_times[-1]:
        logger.error(f"times size: {times.size}")
        if len(times) > 0:
            logger.error(f"{min(times)=} {ref_times[0]=} {max(times)=} {ref_times[-1]=}")
        raise InvalidProcessDates("Need to download full data range before processing")
    logger.info(f"{times.shape[0]} time points")
    sort_idx = np.argsort(times)

    mlat = np.empty((times.shape[0], mlt_vals.shape[0]))
    for i, idx in enumerate(sort_idx):
        height = np.mean(arb_data['ALTITUDE'][idx])
        lat = arb_data[f'MODEL_{hemisphere.upper()}_GEOGRAPHIC_LATITUDE'][idx]
        lon = arb_data[f'MODEL_{hemisphere.upper()}_GEOGRAPHIC_LONGITUDE'][idx]
        apx_lat, mlt = apex.convert(lat, lon, 'geo', 'mlt', height, utils.datetime64_to_datetime(times[idx]))
        mlat[i] = np.interp(mlt_vals, mlt, apx_lat, period=24)
    good_mask = np.mean(abs(mlat - np.median(mlat, axis=0, keepdims=True)), axis=1) < 1
    interpolator = interp1d(
        times.astype('datetime64[s]').astype(float)[sort_idx][good_mask],
        mlat[good_mask],
        axis=0, bounds_error=False
    )
    mlat = interpolator(ref_times.astype(float))
    data = xr.DataArray(
        mlat,
        coords={'time': ref_times, 'mlt': mlt_vals},
        dims=['time', 'mlt']
    )
    logger.info(f"ref times: [{ref_times[0]}, {ref_times[-1]}]")
    data.to_netcdf(output_fn)


check_processed_data_interval = utils.get_data_checker(get_arb_data)


def process_auroral_boundary_dataset(start_date, end_date, download_dir=None, process_dir=None, mlt_vals=None, dt=None):
    if download_dir is None:
        download_dir = config.download_arb_dir
    if process_dir is None:
        process_dir = config.processed_arb_dir
    if mlt_vals is None:
        mlt_vals = config.mlt_vals
    if dt is None:
        dt = config.sample_dt
    Path(process_dir).mkdir(exist_ok=True, parents=True)

    logger.info(f"processing arb dataset over interval {start_date=} {end_date=}")
    for year in range(start_date.year, end_date.year + 1):
        logger.info(f"arb year {year=}")
        start = max(start_date, datetime(year, 1, 1))
        end = utils.datetime64_to_datetime(np.datetime64(datetime(year + 1, 1, 1)) - dt)
        end = min(end_date, end)
        if end - start <= timedelta(hours=1):
            continue
        for hemisphere in ['north', 'south']:
            output_file = Path(process_dir) / f"arb_{hemisphere}_{year:04d}.nc"
            if check_processed_data_interval(start, end, dt, hemisphere, output_file):
                process_interval(start, end, hemisphere, output_file, download_dir, mlt_vals, dt)
