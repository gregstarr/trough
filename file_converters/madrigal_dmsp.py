import numpy as np
import h5py
import xarray as xr
import glob
import os
import apexpy
import datetime

from trough import convert

input_folder = "E:\\dmsp"
output_file = "E:\\dmsp\\2018_dmsp.nc"

variables = ['gdlat', 'glon', 'gdalt', 'ne', 'hor_ion_v', 'vert_ion_v']
satellites = [15, 16, 17, 18]
dataset = {var: {sat: [] for sat in satellites} for var in variables}
dataset['mlat'] = {sat: [] for sat in satellites}
dataset['mlon'] = {sat: [] for sat in satellites}
dataset['mlt'] = {sat: [] for sat in satellites}

files = glob.glob(os.path.join(input_folder, '*.hdf5'))
converter = apexpy.Apex(datetime.datetime(2018, 10, 1))
for file in files:
    print(file)
    with h5py.File(file, 'r') as f:
        data = f['Data/Table Layout'][()]
    sat = data['sat_id'][0]
    times = data['ut1_unix'] * np.timedelta64(1, 's') + np.datetime64("1970-01-01T00:00:00")
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