"""
experiment:
    - subtract a running mean from madrigal, look for the satellite signatures
    - could the tec map be optimized to uncorrelate the residuals?
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools

from trough import plotting

# define paths, open dataset
tec_paths = "E:\\tec_data\\data\\madrigal\\*.nc"
madrigal = xr.open_mfdataset(tec_paths, combine='by_coords')['tec']

madrigal = madrigal.sel(time='2018-10').sel(longitude=slice(-170, -50), latitude=slice(0, 70))

# interpolation
smoothed = madrigal.rolling(time=12, center=True).mean()

madrigal = madrigal.interp(time=smoothed.time.values, method='nearest')

res = madrigal - smoothed
res = res
res.load()

# fig, ax = plt.subplots(1, 1, figsize=(20, 12))
# for i in range(res.time.size):
#     time_str = str(res[i].time.item()).replace('-', '').replace(':', '')[:15]
#     print(time_str)
#     res[i].plot(x='longitude', y='latitude', add_colorbar=False, vmin=-5, vmax=5, cmap='coolwarm')
#     ax.grid()
#     plt.savefig(f"E:\\tec_data\\plots\\artifact residual\\{time_str}.png", fig=fig)
#     ax.clear()

p1 = res.sel(longitude=-110, latitude=40)
p2 = res.sel(longitude=-110, latitude=45)
mask = p1.notnull() * p2.notnull()
p1 = p1[mask]
p2 = p2[mask]
ccf = stattools.ccf(p1, p2)

plt.xcorr(p1, p2, normed=True, maxlags=int(24 * 60 / 5))
plt.show()
