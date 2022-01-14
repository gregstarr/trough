import numpy as np
import xarray as xr
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
# https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
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
    data = xr.Dataset({key: xr.DataArray(data[key], coords={'time': dates}, dims=['time']) for key in _omni_names})
    data = data.drop_vars(['year', 'decimal_day', 'hour', 'bartels_rotation_number'])
    return data


def process_omni_dataset(input_dir, output_fn):
    output_path = Path(output_fn)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    data = xr.combine_by_coords([open_downloaded_omni_file(path) for path in Path(input_dir).glob('*.dat')])
    data.to_netcdf(output_fn)
