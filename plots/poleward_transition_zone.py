"""
Plot:
    - plot latitude profiles at a few different MLTs
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import xarray as xr
import os
import matplotlib.pyplot as plt

from trough import plotting

plot_directory = "E:\\tec_data\\plots\\poleward transition zone"
dataset_path = "E:\\tec_data\\data\\dataset 1\\test.nc"
dataset = xr.open_dataset(dataset_path)
data = dataset['data']
times = dataset['time']
# sites = xr.load_dataset("C:\\Users\\Greg\\Downloads\\site_20181001.001.nc")
# sat_pos = xr.open_dataarray("E:\\tec_data\\data\\nav\\sat_pos_1min.nc")
dmsp = xr.open_dataset("E:\\dmsp\\2018_dmsp.nc")
dmsp.load()
swarm = xr.open_dataset("E:\\swarm\\2018_swarm.nc")
swarm.load()

mlon_limits = (-120, 20)
mlat_limits = (20, 80)

smoothed = data.rolling(mlat=4, center=True).mean()
mask = smoothed.notnull().sum(dim='mlat') > 10
tmin = smoothed.mlat[smoothed.where(mask, 99).argmin(dim='mlat')]
x1, x2 = np.meshgrid(tmin.i.values*10, tmin.mlon.values)
x = np.column_stack((x1.ravel(), x2.ravel()))
poly = PolynomialFeatures(2)
xp = poly.fit_transform(x)
models = []
for ex in range(tmin.shape[0]):
    tm = tmin[ex].values.ravel()
    mask = (tm <= 70) * (tm >= 50)
    tm = tm[mask]
    xe = xp[mask]
    lin = Ridge()
    lin.fit(xe, tm)
    models.append(lin)


fig = plt.figure()
gs = fig.add_gridspec(6, 1)
ax = fig.add_subplot(gs[:4])
ax.set_facecolor('grey')
ax1 = fig.add_subplot(gs[4:])
# ax2 = ax1.twinx()
ax.grid()
ax1.grid()
ex = 0
i = 0


def keypress(event):
    global i, ex
    if event.key == 'right':
        if i == data.shape[1] - 1:
            if ex == data.shape[0] - 1:
                print("END")
                return
            ex += 1
            i = 0
        else:
            i += 1

    elif event.key == 'left':
        if i == 0:
            if ex == 0:
                print("BEGINNING")
                return
            ex -= 1
            i = data.shape[1] - 1
        else:
            i -= 1
    elif event.key == 'z':
        time_str = str(times.values[ex, i]).replace('-', '').replace(':', '')[:15]
        print(time_str)
        plt.savefig(f"E:\\tec_data\\plots\\poleward transition zone\\{time_str}.png", fig=fig)
    plot(ex, i)


def plot(ex, i):
    ax.clear()
    ax1.clear()
    # ax2.clear()
    data[ex].isel(i=i).plot(ax=ax, vmin=0, vmax=10, add_colorbar=False)
    plotting.plot_dmsp_location(ax, times[ex, i], dmsp)
    plotting.plot_swarm_location_mag(ax, times[ex, i], swarm)
    plotting.plot_dmsp_hor_ion_v_timeseries(ax1, times[ex, i], dmsp)
    plotting.plot_swarm_timeseries(ax1, times[ex, i], swarm)
    plotting.plot_mlt_lines_mag(ax, times[ex, i])
    plotting.plot_coastline_mag(ax, times[ex, i])
    # ax.plot(data.mlon.values, models[ex].predict(xp).reshape(x1.shape)[:, i], 'r')
    # plotting.plot_rx_locations_mag(ax, sites)
    # plotting.plot_sv_ipp_mag(ax, times[ex, i], np.array((37.44, -120.23, 0.)), sat_pos.isel(sv=7),
    #                          dt=np.timedelta64(60, 'm'), min_el=20, color='white')
    ax1.set_ylabel("Horizontal Ion Velocity")
    ax1.set_xlabel("MLAT")
    ax.set_title('')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax1.grid()
    ax1.legend()
    time_str = np.datetime_as_string(times.values[ex, i], 's')
    fig.suptitle(time_str)
    plt.draw()


plot(ex, i)
cid = fig.canvas.mpl_connect('key_press_event', keypress)
plt.show()
