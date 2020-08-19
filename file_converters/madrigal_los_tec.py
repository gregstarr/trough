import numpy as np
import h5py
import xarray as xr
import glob
import os
import apexpy
import datetime

from trough import convert

input_folder = "E:\\los_tec"
output_file = "E:\\los_tec\\2018_los_tec.nc"

variables = ["los_tec", "tec", "azm", "elm"]
satellites = list(range(1, 32))
dataset = {var: None for var in variables}

files = glob.glob(os.path.join(input_folder, '*.h5'))
converter = apexpy.Apex(datetime.datetime(2018, 10, 1))
for file in files:
    print(file)
    with h5py.File(file, 'r') as f:
        data = f['Data/Table Layout'][()]

    el_mask = data['elm'] >= 40
    zero_mask = data['tec'] > 0
    lat_mask = data['gdlat'] > 45
    lon_mask = (data['glon'] > -170) * (data['glon'] < -10)
    mask = el_mask * zero_mask * lat_mask
    data = data[mask]

    sat = data['sat_id']
    rx = data['gps_site']
    times = data['ut1_unix'] * np.timedelta64(1, 's') + np.datetime64("1970-01-01T00:00:00")

    unique_sat = np.unique(sat)
    unique_rx = np.unique(rx)
    unique_times = np.unique(times)

    d = np.empty_like(data, shape=unique_times.shape + unique_rx.shape + unique_sat.shape)

    for var in variables:
        array = xr.DataArray(data[var][:, None], dims=['time', 'satellite'], coords={'time': times, 'satellite': [sat]})
        dataset[var][sat].append(array)
    mlat, mlon = converter.convert(data['gdlat'], data['glon'], 'geo', 'apex')
    mlat = xr.DataArray(mlat[:, None], dims=['time', 'satellite'], coords={'time': times, 'satellite': [sat]})
    dataset['mlat'][sat].append(mlat)
    mlon = xr.DataArray(mlon[:, None], dims=['time', 'satellite'], coords={'time': times, 'satellite': [sat]})
    dataset['mlon'][sat].append(mlon)
    _, mlt = convert.geo_to_mlt(data['gdlat'], data['glon'], mlat.time, converter=converter)
    mlt = xr.DataArray(mlt[:, None], dims=['time', 'satellite'], coords={'time': times, 'satellite': [sat]})
    dataset['mlt'][sat].append(mlt)

for var in dataset.keys():
    for sat in satellites:
        dataset[var][sat] = xr.concat(dataset[var][sat], dim='time')
    dataset[var] = xr.concat(dataset[var].values(), dim='satellite')

dataset = xr.Dataset(dataset)
dataset.to_netcdf(output_file)

d = xr.open_dataset(output_file)
print(d)