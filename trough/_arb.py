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
except ImportError as e:
    warnings.warn(f"Packages required for recreating dataset not installed: {e}")

from trough import config, utils
from trough.exceptions import InvalidProcessDates

_arb_fields = [
    'YEAR', 'DOY', 'TIME', 'ALTITUDE',
    'MODEL_NORTH_GEOGRAPHIC_LATITUDE', 'MODEL_NORTH_GEOGRAPHIC_LONGITUDE',
    'MODEL_SOUTH_GEOGRAPHIC_LATITUDE', 'MODEL_SOUTH_GEOGRAPHIC_LONGITUDE'
]
logger = logging.getLogger(__name__)


def get_arb_paths(start_date, end_date, processed_dir):
    file_dates = np.arange(
        np.datetime64(start_date, 'Y'),
        (np.datetime64(end_date, 's') - np.timedelta64(1, 'h')).astype('datetime64[Y]') + 1,
        np.timedelta64(1, 'Y')
    )
    file_dates = utils.decompose_datetime64(file_dates)
    return [Path(processed_dir) / f"arb_{d[0]:04d}.nc" for d in file_dates]


def get_arb_data(start_date, end_date, processed_dir=None):
    if processed_dir is None:
        processed_dir = config.processed_arb_dir
    data = xr.concat([xr.open_dataset(file) for file in get_arb_paths(start_date, end_date, processed_dir)], 'time')
    return data.sel(time=slice(start_date, end_date))


def parse_arb_fn(path):
    sat_name = path.name[36:39]
    date = datetime.strptime(path.name[67:75], "%Y%m%d")
    return sat_name, date


def _get_downloaded_arb_data(start_date, end_date, input_dir):
    start_date -= timedelta(hours=3)
    end_date += timedelta(hours=3)
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


def process_interval(start_date, end_date, output_fn, input_dir, mlt_vals, sample_dt):
    logger.info(f"processing arb data for {start_date, end_date}")
    ref_times = np.arange(np.datetime64(start_date, 's'), np.datetime64(end_date, 's'), sample_dt)
    apex = Apex(date=start_date)
    arb_data, times = _get_downloaded_arb_data(start_date, end_date, input_dir)
    if times.size == 0 or min(times) > ref_times[0] or max(times) < ref_times[-1]:
        logger.error(f"times size: {times.size}")
        if len(times) > 0:
            logger.error(f"{min(times)=} {ref_times[0]=} {max(times)=} {ref_times[-1]=}")
        raise InvalidProcessDates("Need to download full data range before processing")
    logger.info(f"{times.shape[0]} time points")
    sort_idx = np.argsort(times)

    data = {}
    for hemi in ['NORTH', 'SOUTH']:
        mlat = np.empty((times.shape[0], mlt_vals.shape[0]))
        for i, idx in enumerate(sort_idx):
            height = np.mean(arb_data['ALTITUDE'][idx])
            lat = arb_data[f'MODEL_{hemi}_GEOGRAPHIC_LATITUDE'][idx]
            lon = arb_data[f'MODEL_{hemi}_GEOGRAPHIC_LONGITUDE'][idx]
            apx_lat, mlt = apex.convert(lat, lon, 'geo', 'mlt', height, utils.datetime64_to_datetime(times[idx]))
            mlat[i] = np.interp(mlt_vals, mlt, apx_lat, period=24)
        good_mask = np.mean(abs(mlat - np.median(mlat, axis=0, keepdims=True)), axis=1) < 1
        interpolator = interp1d(
            times.astype('datetime64[s]').astype(float)[sort_idx][good_mask],
            mlat[good_mask],
            axis=0, bounds_error=False
        )
        mlat = interpolator(ref_times.astype(float))
        data[f'arb_{hemi.lower()}'] = xr.DataArray(
            mlat,
            coords={'time': ref_times, 'mlt': mlt_vals},
            dims=['time', 'mlt']
        )

    logger.info(f"ref times: [{ref_times[0]}, {ref_times[-1]}]")

    data = xr.Dataset(data)
    data.to_netcdf(output_fn)


def check_processed_data_interval(start, end, processed_file):
    if processed_file.exists():
        logger.info(f"processed file already exists {processed_file=}, checking...")
        try:
            data_check = get_arb_data(start, end, processed_file.parent)
            if not data_check.isnull().all(dim=['mlt']).any().item():
                logger.info(f"downloaded data already processed {processed_file=}, checking...")
                return False
        except Exception as e:
            logger.info(f"error reading processed file {processed_file=}: {e}, removing and reprocessing")
            processed_file.unlink()
    return True


def process_auroral_boundary_dataset(start_date, end_date, download_dir=None, process_dir=None, mlt_vals=None, dt=None):
    if download_dir is None:
        download_dir = config.download_arb_dir
    if process_dir is None:
        process_dir = config.processed_arb_dir
    if mlt_vals is None:
        mlt_vals = config.get_mlt_vals()
    if dt is None:
        dt = np.timedelta64(1, 'h')
    Path(process_dir).mkdir(exist_ok=True, parents=True)

    for year in range(start_date.year, end_date.year + 1):
        output_file = Path(process_dir) / f"arb_{year:04d}.nc"
        start = max(start_date, datetime(year, 1, 1))
        end = min(end_date, datetime(year + 1, 1, 1))
        if end - start <= timedelta(hours=1):
            continue
        if check_processed_data_interval(start, end, output_file):
            process_interval(start, end, output_file, download_dir, mlt_vals, dt)
