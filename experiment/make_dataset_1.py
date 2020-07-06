"""
Make dataset 1:
    - each sample is (X, Y, UT) = ([-120, 20], [20, 80], MLT->[-6, 6] minus the window width
    - TODO: expand the MLT range in particular cases so that all examples have the same number of time steps
"""
import numpy as np
import xarray as xr
import os
import apexpy

from trough import utils, convert

# define paths, open dataset
tec_paths = "E:\\tec_data\\data\\madrigal\\*.nc"
save_dir = "E:\\tec_data\\data\\dataset 1"
madrigal = xr.open_mfdataset(tec_paths, combine='by_coords')['tec']

madrigal = madrigal.sel(time='2018-10')

# interpolation
madrigal = madrigal.coarsen(time=4).mean()
madrigal = madrigal.rolling(time=3, center=True, min_periods=1).mean()
# binned = madrigal.coarsen(longitude=2, latitude=2).mean()
# interpolated = binned.interp(longitude=madrigal.longitude, latitude=madrigal.latitude)
# madrigal = madrigal.where(madrigal.notnull(), interpolated)

# convert to mlon
converter = apexpy.Apex(date=utils.datetime64_to_datetime(madrigal.time.values[0]))
x_vals = np.arange(-120, 20)
y_vals = np.arange(20, 80)
x, y = np.meshgrid(x_vals, y_vals)
glat, glon = converter.convert(y.ravel(), x.ravel(), 'apex', 'geo')
glat_grid = glat.reshape(y.shape)
glon_grid = glon.reshape(y.shape)
lat = xr.DataArray(glat_grid, dims=["y", "x"], coords={"x": x_vals, "y": y_vals})
lon = xr.DataArray(glon_grid, dims=["y", "x"], coords={"x": x_vals, "y": y_vals})
madrigal = madrigal.interp(longitude=lon, latitude=lat)

# interpolation
# madrigal = madrigal.coarsen(time=4).mean()
# binned = madrigal.coarsen(x=2, y=2).mean()
# interpolated = binned.interp(x=madrigal.x, y=madrigal.y)
# madrigal = madrigal.where(madrigal.notnull(), interpolated)

# restrict to full view within [-6, 6] MLT
mlt_at_edges = convert.mlon_to_mlt(x_vals[[0, -1]], madrigal.time, converter)
mlt0 = xr.DataArray(mlt_at_edges[:, 0], dims=['time'], coords={'time': madrigal.time.values})
madrigal = madrigal.assign_coords({'mlt0': mlt0})
mlt_at_edges[mlt_at_edges > 12] -= 24
mlt_mask = np.all(mlt_at_edges >= -8, axis=1) * np.all(mlt_at_edges < 8, axis=1) * (mlt_at_edges[:, 0] < mlt_at_edges[:, 1])
madrigal = madrigal[mlt_mask]

# separate into days
dt = madrigal.time.dt
example_n = dt.dayofyear + (dt.hour + dt.minute / 60 + dt.second / (60 * 60)) / 24
example_n -= example_n.min()
example_n //= 1
madrigal = madrigal.assign_coords({'example': example_n})

madrigal.load()
data = []
times = []
for ex_n in np.unique(madrigal.example.values.astype(int)):
    print(ex_n)
    ex = madrigal[madrigal.example == ex_n]
    i = np.arange(ex.shape[0])
    mlt0 = xr.DataArray(ex.mlt0.values, dims=['i'], coords={'i': i})
    t = xr.DataArray(ex.time.values[None, ...], dims=['example', 'i'], coords={'i': i, 'example': [ex_n]})
    ex = xr.DataArray(ex.values[None, ...], dims=['example', 'i', 'mlat', 'mlon'],
                      coords={'i': i,
                              'mlat': y_vals,
                              'mlon': x_vals,
                              'example': [ex_n],
                              'mlt0': mlt0})
    data.append(ex)
    times.append(t)
data = xr.concat(data, dim='example')
times = xr.concat(times, dim='example')
dataset = xr.Dataset({'data': data, 'time': times})
dataset.to_netcdf(os.path.join(save_dir, "test.nc"))

# import matplotlib.pyplot as plt
# for i in range(3):
#     plt.figure()
#     madrigal[i].plot()
# plt.show()
