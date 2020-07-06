"""
Plotting script:
    - single image
    - average TEC in MLT bins over entire (downloaded) Madrigal dataset for quiet, medium and stormy conditions
"""
import numpy as np
import xarray as xr
import time
import os
import matplotlib.pyplot as plt

from trough import utils, convert

tec_paths = "E:\\tec_data\\data\\madrigal\\*.nc"
save_dir = "E:\\tec_data\\data\\madrigal\\mag"
print("Loading dataset")
tec_data = xr.open_mfdataset(tec_paths, combine='by_coords')['tec'].sel(longitude=slice(-170, -51), latitude=slice(0, 89))

mlat_vals = np.arange(20, 80)
mlt_vals = np.arange(-6, 6, 24/360)
mlt, mlat = np.meshgrid(mlt_vals, mlat_vals)

times = tec_data.time.values
batch_size = 1000
total_time = 0
print(f"Calculating, {times.shape[0] / batch_size} batches")
for i in range(int(np.ceil(times.shape[0] / batch_size))):
    t0 = time.time()
    index_slice = slice(i*batch_size, (i+1)*batch_size)
    current_times = times[index_slice]
    current_data = tec_data.isel(time=index_slice)
    current_data.load()

    smoothed = current_data.rolling(time=3, center=True, min_periods=1).mean()
    binned = smoothed.coarsen(longitude=2, latitude=2).mean()
    interpolated = binned.interp(longitude=current_data.longitude, latitude=current_data.latitude, method='nearest')
    current_data = current_data.where(current_data.notnull(), interpolated)

    glat, glon = convert.mlt_to_geo(mlat.ravel(), mlt.ravel(), current_data.time)
    glat_grid = glat.reshape((current_times.shape[0], ) + mlat.shape)
    glon_grid = glon.reshape((current_times.shape[0], ) + mlat.shape)
    lat = xr.DataArray(glat_grid, dims=["time", "mlat", "mlt"],
                       coords={"mlt": mlt_vals, "mlat": mlat_vals, "time": current_times})
    lon = xr.DataArray(glon_grid, dims=["time", "mlat", "mlt"],
                       coords={"mlt": mlt_vals, "mlat": mlat_vals, "time": current_times})
    remapped = current_data.interp(longitude=lon, latitude=lat)
    remapped.to_netcdf(os.path.join(save_dir, f"mag_tec_{i:03d}.nc"))
    total_time += time.time() - t0
    t_sec = total_time * times.shape[0] / ((i+1) * batch_size)
    print(f"{total_time / 3600} / {t_sec / 3600}")
