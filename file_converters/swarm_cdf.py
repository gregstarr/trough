import numpy as np
import pysatCDF
import xarray as xr
import glob
import os
import apexpy
import datetime

from trough import convert

year = 2018
input_folder = "E:\\swarm"
variables = ['Latitude', 'Longitude', 'n', 'T_elec']
# variables = ['Latitude', 'Longitude', 'Vixh', 'Viy', 'Viz', 'Quality_flags', 'MLT', 'VsatN', 'VsatE', 'VsatC', 'Ehx',
#              'Ehy', 'Ehz', 'Evx', 'Evy', 'Evz', 'Bx', 'By', 'Bz', 'Vicrx', 'Vicry', 'Vicrz', 'QDLatitude']
satellites = ['A', 'B', 'C']

for month in range(1, 13):
    output_file = f"E:\\swarm\\{year}_{month}_swarm.nc"
    dataset = {var: {sat: [] for sat in satellites} for var in variables}
    dataset['mlat'] = {sat: [] for sat in satellites}
    dataset['mlon'] = {sat: [] for sat in satellites}
    dataset['mlt'] = {sat: [] for sat in satellites}

    files = glob.glob(os.path.join(input_folder, f'*LP_HM_{year:04d}{month:02d}*.cdf'))
    converter = apexpy.Apex(datetime.datetime(year, 1, 1))
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
        _, mlt = convert.geo_to_mlt(data['Latitude'], data['Longitude'], mlat.time, converter=converter)
        mlt[mlt > 12] -= 24
        mlt = xr.DataArray(mlt[:, None], dims=['time', 'satellite'], coords={'time': times, 'satellite': [sat]})
        dataset['mlt'][sat].append(mlt)

    output_dset = {}
    for var in dataset.keys():
        var_array = []
        for sat in satellites:
            if len(dataset[var][sat]) == 0:
                continue
            v = xr.concat(dataset[var][sat], dim='time')
            _, index = np.unique(v.time.values, return_index=True)
            var_array.append(v.isel(time=index))
        if len(var_array) == 0:
            continue
        output_dset[var] = xr.concat(var_array, dim='satellite')

    output_dset = xr.Dataset(output_dset)
    output_dset['n'] *= 100 ** 3
    output_dset.to_netcdf(output_file)

    d = xr.open_dataset(output_file)
    print(d)
