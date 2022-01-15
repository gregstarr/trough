import h5py
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from apexpy import Apex
from scipy.interpolate import interp1d
import logging

from trough import config, utils
from trough.exceptions import InvalidProcessDates


_arb_fields = ['YEAR', 'DOY', 'TIME', 'ALTITUDE', 'MODEL_NORTH_GEOGRAPHIC_LATITUDE', 'MODEL_NORTH_GEOGRAPHIC_LONGITUDE']
logger = logging.getLogger(__name__)


def get_arb_paths(start_date, end_date, processed_dir):
    file_dates = np.arange(
        np.datetime64(start_date, 'Y'),
        np.datetime64(end_date, 'Y') + 1,
        np.timedelta64(1, 'Y')
    )
    file_dates = utils.decompose_datetime64(file_dates)
    return [Path(processed_dir) / f"arb_{d[0]:04d}.nc" for d in file_dates]


def get_arb_data(start_date, end_date, processed_dir=None):
    if processed_dir is None:
        processed_dir = config.processed_arb_dir
    data = xr.concat([xr.open_dataarray(file) for file in get_arb_paths(start_date, end_date, processed_dir)], 'time')
    return data.sel(time=slice(start_date, end_date))


def _parse_arb_fn(path):
    sat_name = path.name[:7]
    date1 = datetime.strptime(path.name[25:39], "%Y%jT%H%M%S")
    date2 = datetime.strptime(path.name[40:54], "%Y%jT%H%M%S")
    return sat_name, date1, date2


def _get_downloaded_arb_data(start_date, end_date, input_dir):
    start_date -= timedelta(hours=3)
    end_date += timedelta(hours=3)
    data = {field: [] for field in _arb_fields}
    data['sat'] = []
    for path in Path(input_dir).glob('*.nc'):
        sat_name, date1, date2 = _parse_arb_fn(path)
        if start_date > date2 or end_date < date1:
            continue
        data['sat'].append(sat_name)
        with h5py.File(path) as f:
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
    data, times = _get_downloaded_arb_data(start_date, end_date, input_dir)
    if times.size == 0 or min(times) > ref_times[0] or max(times) < ref_times[-1]:
        logger.error(f"times size: {times.size}")
        if len(times) > 0:
            logger.error(f"times: {min(times)} - {max(times)}")
        raise InvalidProcessDates(f"Need to download full data range before processing")
    assert times.shape == np.unique(times).shape, "Non unique times, time to fix"
    logger.info(f"{times.shape[0]} time points")
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    mlat = np.empty((times.shape[0], mlt_vals.shape[0]))
    for i, idx in enumerate(sort_idx):
        lat = data['MODEL_NORTH_GEOGRAPHIC_LATITUDE'][idx]
        lon = data['MODEL_NORTH_GEOGRAPHIC_LONGITUDE'][idx]
        height = np.mean(data['ALTITUDE'][idx])
        apx_lat, mlt = apex.convert(lat, lon, 'geo', 'mlt', height, utils.datetime64_to_datetime(times[idx]))
        mlat[i] = np.interp(mlt_vals, mlt, apx_lat, period=24)
    logger.info(f"ref times: [{ref_times[0]}, {ref_times[-1]}]")
    interpolator = interp1d(times.astype('datetime64[s]').astype(float), mlat, axis=0, bounds_error=False)
    mlat = interpolator(ref_times.astype(float))
    data = xr.DataArray(mlat, coords={'time': ref_times, 'mlt': mlt_vals}, dims=['time', 'mlt'])
    data.to_netcdf(output_fn)


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
        start = max(start_date, datetime(year, 1, 1))
        end = min(end_date, datetime(year + 1, 1, 1))
        process_interval(start, end, Path(process_dir) / f"arb_{year:04d}.nc", download_dir, mlt_vals, dt)
