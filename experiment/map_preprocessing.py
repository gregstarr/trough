import numpy as np
import xarray as xr
import os
import apexpy
import matplotlib.pyplot as plt
from cartopy import crs

from trough import utils, convert, plotting

# define paths, open dataset
tec_paths = "E:\\tec_data\\data\\dataset 1\\major\\*.nc"
save_dir = "E:\\tec_data\\data\\dataset 1"
madrigal = xr.open_mfdataset(tec_paths, combine='by_coords')['tec']

# nothing
madrigal = madrigal.sel(time='2018-10')

# prepare conversion
converter = apexpy.Apex(date=utils.datetime64_to_datetime(madrigal.time.values[0]))
x_vals = np.arange(-120, 20)
y_vals = np.arange(20, 80)
x, y = np.meshgrid(x_vals, y_vals)
glat, glon = converter.convert(y.ravel(), x.ravel(), 'apex', 'geo')
glat_grid = glat.reshape(y.shape)
glon_grid = glon.reshape(y.shape)
lat = xr.DataArray(glat_grid, dims=["y", "x"], coords={"x": x_vals, "y": y_vals})
lon = xr.DataArray(glon_grid, dims=["y", "x"], coords={"x": x_vals, "y": y_vals})

# coarsen/fill -> time average
binned = madrigal.coarsen(longitude=4, latitude=4).mean()
interpolated = binned.interp(longitude=madrigal.longitude, latitude=madrigal.latitude)
cf = madrigal.where(madrigal.notnull(), interpolated)
cf1 = cf.interp(longitude=lon, latitude=lat, method='nearest')
cf1 = cf1.coarsen(time=3).mean()

cf2 = cf.coarsen(time=3).mean()
cf2 = cf2.interp(longitude=lon, latitude=lat, method='nearest')

fig, ax = plt.subplots(1, 2, figsize=(18, 9))

cf1.sel(time='2018-10-10T01:20:00').plot(ax=ax[0], x='x', y='y', vmin=0, vmax=12, add_colorbar=False)
cf2.sel(time='2018-10-10T01:20:00').plot(ax=ax[1], x='x', y='y', vmin=0, vmax=12, add_colorbar=False)

plt.show()
