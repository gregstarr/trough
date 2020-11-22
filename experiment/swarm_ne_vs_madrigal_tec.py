import numpy as np
import xarray as xr
import h5py
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
from matplotlib import colors


swarm = xr.open_mfdataset("E:\\swarm\\2018*swarm.nc", concat_dim="time", combine="by_coords")
print("SWARM DATA:")
print(swarm)
swarm_timestamps = np.int64((swarm.time.values - np.datetime64("1970-01-01T00:00:00")).astype(float)/1e9)
swarm_ne = swarm['n'][:, 0].values
fin_mask = np.isfinite(swarm_ne)
swarm_ne = swarm_ne[fin_mask] / 100**3
swarm_mlt = swarm['mlt'][fin_mask, 0].values
swarm_mlt[swarm_mlt < 0] += 24
swarm_mlat = swarm['mlat'][fin_mask, 0].values
swarm_timestamps = swarm_timestamps[fin_mask]

with h5py.File("E:\\tec_data\\grid.h5", 'r') as f:
    mlt = f['mlt'][()]
    mlat = f['mlat'][()]

mlt_bins = np.concatenate((mlt - (mlt[1] - mlt[0])/2, [mlt[-1] + (mlt[1] - mlt[0])/2]))
mlat_bins = np.concatenate((mlat - (mlat[1] - mlat[0])/2, [mlat[-1] + (mlat[1] - mlat[0])/2]))

tec_vals = []
ne_vals = []

print("BINNING SWARM DATA")
for month in range(1, 13):
    print(month)
    tec_file = f"E:\\tec_data\\2018_{month:02d}_tec.h5"
    with h5py.File(tec_file, 'r') as f:
        tec = f['tec'][()]
        tec_timestamps = f['start_time'][()]
    dt = (tec_timestamps[1] - tec_timestamps[0]) / 2
    time_bins = np.concatenate((tec_timestamps - dt, [tec_timestamps[-1] + dt]))
    bstat = binned_statistic_dd(np.column_stack((swarm_mlat, swarm_mlt, swarm_timestamps)), swarm_ne,
                                bins=[mlat_bins, mlt_bins, time_bins])
    binned_swarm = bstat.statistic
    fin = np.isfinite(binned_swarm) * (binned_swarm < 1e16)
    tec_vals.append(tec[fin])
    ne_vals.append(binned_swarm[fin])

tec_vals = np.concatenate(tec_vals)
ne_vals = np.concatenate(ne_vals)
fin = np.isfinite(tec_vals) * np.isfinite(ne_vals)
tec_vals = tec_vals[fin]
ne_vals = ne_vals[fin]

print("PLOTTING")
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 12), tight_layout=True)
ax[0].scatter(tec_vals, ne_vals, s=1)
ax[1].hist2d(tec_vals, ne_vals, bins=200)
ax[2].hist2d(tec_vals, ne_vals, bins=200, norm=colors.LogNorm())

plt.show()
