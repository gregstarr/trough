import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt

from trough import plotting
from trough.satellite import find_troughs_swarm
from trough import tec_map

plot_directory = "E:\\tec_data\\plots\\poleward transition zone"
dataset_path = "E:\\tec_data\\data\\dataset 1\\*madrigal.nc"
data = xr.open_mfdataset(dataset_path, concat_dim="time", combine="by_coords")['tec']
features = tec_map.get_features(data)
times = data.time

swarm = xr.open_mfdataset("E:\\swarm\\2018*swarm.nc", concat_dim="time", combine="by_coords")
# swarm_troughs = find_troughs_swarm(swarm)
swarm_troughs = {}
for sat in ['A', 'B', 'C']:
    swarm_troughs[sat] = xr.open_dataset(f"E:\\swarm\\2018_swarm_troughs_{sat}.nc")

lr, dbg = tec_map.get_lr(features, swarm_troughs)

tec_troughs = tec_map.get_troughs(features, lr)

# PLOT PARAMETERS
dt = 1.2 * (times.values[1] - times.values[0])  # time between TEC maps
TEC_VLIM = dict(vmin=0, vmax=12)
FEATURE_VLIM = dict(vmin=-3, vmax=3)
PROB_VLIM = dict(vmin=0, vmax=1)

# setup figure
fig, ax = plt.subplots(1, 2, sharex='all', sharey='all',
                       gridspec_kw=dict(hspace=.1, wspace=.01),
                       subplot_kw=dict(polar=True))
ax = ax.flatten()
t = 3000


# event handler
def keypress(event):
    global t
    if event.key == 'right':
        t += 1
    elif event.key == 'left':
        if t == 0:
            print("BEGINNING")
            return
        else:
            t -= 1
    elif event.key == 'z':
        time_str = str(times.values[t]).replace('-', '').replace(':', '')[:15]
        print(time_str)
        plt.savefig(f"E:\\plots\\poleward transition zone\\{time_str}.png", fig=fig)
    plot(t)


# plot updater
def plot(t):
    for a in ax.flatten():
        a.clear()
    date = times.values[t]

    # ax[0]: TEC map, section, 1/0
    plotting.plot_tec_map_mag(ax[0], date, data[t], **TEC_VLIM)
    plotting.plot_coastline_mag(ax[0], date, coord_sys='mlt', alpha=.5)
    # plotting.plot_sat_location(ax[0], date, swarm, dt)
    # plotting.plot_trough_locations(ax[0], date, swarm_troughs, dt)
    # plotting.plot_solar_terminator(ax[0], date, altitude=100, linestyle='--', color='y')
    # plotting.plot_lr_debug(ax[0], date, dbg)
    # ax[1]: d2lat, ax[2]: low val, ax[3]: pwall, ax[4]: ewall
    # plotting.plot_feature_maps(ax[1:5], date, features[t], lr.coef_, **FEATURE_VLIM, cmap='coolwarm')
    # ax[5]: probabilities / detections
    # plotting.plot_lr_predictions(ax[5], date, features[t], lr, **PROB_VLIM, cmap='Blues')
    plotting.plot_tec_trough(ax[1], date, tec_troughs[t], cmap="Blues")

    time_str = np.datetime_as_string(times.values[t], 's')
    fig.suptitle(time_str)

    titles = ['TEC', 'd2lat', 'low val', 'pwall', 'ewall', 'prob']
    for i, a in enumerate(ax.flatten()):
        plotting.format_polar_mag_ax(a)
        a.set_title(titles[i])
    plt.draw()


plot(t)
cid = fig.canvas.mpl_connect('key_press_event', keypress)
plt.show()
