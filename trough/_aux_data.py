import h5py
import numpy as np
import pandas
from pathlib import Path
from datetime import datetime
from apexpy import Apex
from scipy.interpolate import interp1d
import logging

import trough.utils as trough_utils
from trough import config


logger = logging.getLogger(__name__)

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
_arb_fields = ['YEAR', 'DOY', 'TIME', 'ALTITUDE', 'MODEL_NORTH_GEOGRAPHIC_LATITUDE', 'MODEL_NORTH_GEOGRAPHIC_LONGITUDE']


def open_downloaded_omni_file(fn):
    data = np.loadtxt(fn, dtype={'names': _omni_names, 'formats': _omni_formats})
    dates = (data['year'] - 1970).astype('datetime64[Y]') + \
            (data['decimal_day'] - 1).astype('timedelta64[D]') + \
            data['hour'].astype('timedelta64[h]')
    data = pandas.DataFrame(data)
    data.index = dates
    return data.drop(columns=['year', 'hour', 'decimal_day'])


def get_omni_data(start_date, end_date):
    path = Path(config.processed_omni_dir) / 'omni.h5'
    data = pandas.read_hdf(path, key='data')
    return data[start_date:end_date]


def get_arb_data(start_date, end_date):
    """Gets auroral boundary mlat and timestamps

    """
    arb = []
    ut = []
    file_dates = np.arange(
        start_date.astype('datetime64[Y]'),
        end_date.astype('datetime64[Y]') + 1,
        np.timedelta64(1, 'Y')
    )
    file_dates = trough_utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        fn = Path(config.processed_arb_dir) / f"arb_{y:04d}.h5"
        with h5py.File(fn, 'r') as f:
            arb.append(f['mlat'][()])
            ut.append(f['times'][()])
    arb = np.concatenate(arb, axis=0)
    times = np.concatenate(ut, axis=0).astype('datetime64[s]')
    mask = (times >= start_date) & (times < end_date)
    return arb[mask], times[mask]


def _parse_arb_fn(path):
    sat_name = path.name[:7]
    date1 = datetime.strptime(path.name[25:39], "%Y%jT%H%M%S")
    date2 = datetime.strptime(path.name[40:54], "%Y%jT%H%M%S")
    return sat_name, date1, date2


def _process_auroral_boundary_dataset_year(year):
    logger.info(f"processing arb data for {year}")
    output_path = Path(config.processed_arb_dir) / f"arb_{year}.h5"

    start_date = datetime(year - 1, 12, 31)
    end_date = datetime(year + 1, 1, 2)
    apex = Apex(date=start_date)

    data = {field: [] for field in _arb_fields}
    data['sat'] = []
    for path in Path(config.download_arb_dir).glob('*.nc'):
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
    assert times.shape == np.unique(times).shape, "Non unique times, time to fix"
    logger.info(f"{times.shape[0]} time points")
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    mlt_vals = config.get_mlt_vals()
    mlat = np.empty((times.shape[0], mlt_vals.shape[0]))
    for i, idx in enumerate(sort_idx):
        lat = data['MODEL_NORTH_GEOGRAPHIC_LATITUDE'][idx]
        lon = data['MODEL_NORTH_GEOGRAPHIC_LONGITUDE'][idx]
        height = np.mean(data['ALTITUDE'][idx])
        apx_lat, mlt = apex.convert(lat, lon, 'geo', 'mlt', height, trough_utils.datetime64_to_datetime(times[idx]))
        mlat[i] = np.interp(mlt_vals, mlt, apx_lat, period=24)
    ref_times = np.arange(
        np.datetime64(f"{year}-01-01T00:00:00"),
        np.datetime64(f"{year + 1}-01-01T00:00:00"),
        np.timedelta64(1, 'h')
    )
    logger.info(f"ref times for year: [{ref_times[0]}, {ref_times[-1]}]")
    ref_times = ref_times[(ref_times >= times[0]) & (ref_times <= times[-1])]
    logger.info(f"reduced ref times: [{ref_times[0]}, {ref_times[-1]}]")
    ref_times = ref_times.astype(float)
    interpolator = interp1d(times.astype('datetime64[s]').astype(float), mlat, axis=0)
    mlat = interpolator(ref_times)
    trough_utils.write_h5(output_path, times=ref_times, mlat=mlat)


def process_auroral_boundary_dataset():
    arb_file_info = [_parse_arb_fn(path) for path in Path(config.download_arb_dir).glob("*.nc")]
    min_date = min([start_date for sat_name, start_date, end_date in arb_file_info])
    max_date = max([end_date for sat_name, start_date, end_date in arb_file_info])
    Path(config.processed_arb_dir).mkdir(exist_ok=True, parents=True)
    for year in range(min_date.year, max_date.year + 1):
        _process_auroral_boundary_dataset_year(year)


def process_omni_dataset():
    output_path = Path(config.processed_omni_dir) / 'omni.h5'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    data = []
    for path in Path(config.download_omni_dir).glob('*.dat'):
        data.append(open_downloaded_omni_file(path))
    data = pandas.concat(data)
    data.to_hdf(output_path, 'data')
