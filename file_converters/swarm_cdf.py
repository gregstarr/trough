import numpy as np
import pysatCDF
import xarray as xr
import glob
import os
import apexpy
import datetime

from trough import convert

input_folder = "E:\\swarm"
output_file = "E:\\swarm\\2018_swarm.nc"

variables = ['Latitude', 'Longitude', 'Vixh', 'Viy', 'Viz', 'Quality_flags', 'MLT']
satellites = ['A', 'B', 'C']
dataset = {var: {sat: [] for sat in satellites} for var in variables}
dataset['mlat'] = {sat: [] for sat in satellites}
dataset['mlon'] = {sat: [] for sat in satellites}
dataset['mlt'] = {sat: [] for sat in satellites}

files = glob.glob(os.path.join(input_folder, '*.cdf'))
converter = apexpy.Apex(datetime.datetime(2018, 10, 1))
for file in files:
    print(file)
    with pysatCDF.CDF(file) as f:
        data = f.data
    sat = os.path.basename(file)[11]
    times = data['Timestamp']
    for var in variables:
        array = xr.DataArray(data[var][:, None], dims=['time', 'satellite'], coords={'time': times, 'satellite': [sat]})
        dataset[var][sat].append(array)
    mlat, mlon = converter.convert(data['Latitude'], data['Longitude'], 'geo', 'apex')
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
