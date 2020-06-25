import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import apexpy
import os

import trough

tec_paths = "E:\\tec_data\\data\\*.nc"
index_path = "E:\\indices.nc"
plot_directory = "E:\\tec_data\\plots"
print("Loading dataset")
tec_data = xr.open_mfdataset(tec_paths, combine='by_coords')['tec']
tec_data = tec_data.sel(longitude=slice(-90, -60), latitude=slice(25, 80))
index_data = xr.open_dataset(index_path)
kp_data = index_data['kp']

mlt = np.mod(np.arange(0, 12, 12/180) + 19, 24)
mlt[mlt > 12] -= 24
mlat = np.arange(40, 75, 1)
mlt_grid, mlat_grid = np.meshgrid(mlt, mlat)

tec_agg = {
    'quiet': {
        'sum': np.zeros_like(mlt_grid),
        'n': np.zeros_like(mlt_grid),
    },
    'moderate': {
        'sum': np.zeros_like(mlt_grid),
        'n': np.zeros_like(mlt_grid),
    },
    'strong': {
        'sum': np.zeros_like(mlt_grid),
        'n': np.zeros_like(mlt_grid),
    },
}

for date in tec_data.time:
    data = tec_data.sel(time=date)
    print(date)
    datetime = trough.utils.datetime64_to_datetime(date)
    apex_converter = apexpy.Apex(date=datetime)
    glat, glon = apex_converter.convert(mlat_grid.ravel(), mlt_grid.ravel(), 'mlt', 'geo', datetime=datetime, precision=-1)
    glat_grid = glat.reshape(mlt_grid.shape)
    glon_grid = glon.reshape(mlt_grid.shape)
    lat = xr.DataArray(glat_grid, dims=["mlat", "mlt"], coords={"mlt": mlt, "mlat": mlat})
    lon = xr.DataArray(glon_grid, dims=["mlat", "mlt"], coords={"mlt": mlt, "mlat": mlat})
    remapped = data.interp(longitude=lon, latitude=lat)
    kp = kp_data.interp(time=date, method='nearest').item()
    if kp <= 2:
        condition = 'quiet'
    elif kp <= 4:
        condition = 'moderate'
    else:
        condition = 'strong'
    tec_agg[condition]['sum'] += remapped.fillna(0).values
    tec_agg[condition]['n'] += remapped.notnull().values

fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
vmax = 0
for i, condition in enumerate(tec_agg):
    np.save(f"E:\\tec_data\\{condition}_sum.npy", tec_agg[condition]['sum'])
    np.save(f"E:\\tec_data\\{condition}_n.npy", tec_agg[condition]['n'])
    tec_agg[condition]['average'] = tec_agg[condition]['sum'] / tec_agg[condition]['n']
    try:
        vmax = max(vmax, tec_agg[condition]['average'][np.isfinite(tec_agg[condition]['average'])].max())
    except:
        pass

for i, condition in enumerate(tec_agg):
    a = ax[i]
    pcm = a.pcolormesh(mlt_grid, mlat_grid, tec_agg[condition]['average'], vmin=0, vmax=vmax)
    a.set_title(condition)

fig.colorbar(pcm, ax=a)
plt.tight_layout()
plt.savefig(os.path.join(plot_directory, "average.png"))
