"""
Plotting script:
    - 3 subplots
    - first:
        - apex coordinates
        - TEC map
        - Mahali IPPs for SV G07 and G30
    - second:
        - time series line plot of G30 TEC
    - third:
        - time series line plot of G07 TEC
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import apexpy

import trough
from trough.gps import ReceiverArray

data_paths = "E:\\tec_data\\data\\madrigal\\201510*.nc"
plot_directory = "E:\\tec_data\\plots\\mahali"
print("Loading dataset")
rx_array = ReceiverArray()
madrigal = xr.open_mfdataset(data_paths, combine='by_coords', parallel=True)['tec'].sel(time=rx_array.obs_time_range)
madrigal.load()
g30_tec = rx_array.align_with_tec_map(madrigal, 'G30')
g30_ipp = rx_array.pierce_points.sel(sv='G30')
g07_tec = rx_array.align_with_tec_map(madrigal, 'G07')
g07_ipp = rx_array.pierce_points.sel(sv='G07')

mlon = np.arange(-120, -69)
mlat = np.arange(50, 81)
mlon_grid, mlat_grid = np.meshgrid(mlon, mlat)
apex_converter = apexpy.Apex(date=trough.utils.datetime64_to_datetime(rx_array.obs_time_range.start))
print("Converting")
glat, glon = apex_converter.convert(mlat_grid.ravel(), mlon_grid.ravel(), 'apex', 'geo')
glat_grid = glat.reshape(mlat_grid.shape)
glon_grid = glon.reshape(mlat_grid.shape)
lat = xr.DataArray(glat_grid, dims=["mlat", "mlon"], coords={"mlon": mlon, "mlat": mlat})
lon = xr.DataArray(glon_grid, dims=["mlat", "mlon"], coords={"mlon": mlon, "mlat": mlat})

cmap = plt.cm.get_cmap('tab10', len(rx_array.pierce_points.rx.values))
time = rx_array.obs.time.values[0]
average_range = np.timedelta64(30, 'm')
dt = np.timedelta64(5, 'm')
while time < rx_array.obs.time.values[-1]:
    time_str = str(time).replace('-', '').replace(':', '')[:15]
    print(time_str)
    time_slice = slice(time, time + average_range)
    time += dt
    if g30_tec.sel(time=time_slice).notnull().sum() < 10 and g07_tec.sel(time=time_slice).notnull().sum() < 10:
        continue
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(6, 1, wspace=.05, hspace=.05)
    ax1 = fig.add_subplot(gs[:4])
    # madrigal tec map
    remapped = madrigal.sel(time=time_slice).mean(dim='time').interp(longitude=lon, latitude=lat)
    remapped.plot.pcolormesh(ax=ax1, vmin=0, vmax=12)
    for i, rx in enumerate(g30_ipp.rx.values):
        # G30 GPS
        pp = g30_ipp.sel(time=time_slice, rx=rx)
        pp = pp[pp.notnull().all(dim='component')]
        mlat, mlon = apex_converter.convert(pp.sel(component='lat').values[::60], pp.sel(component='lon').values[::60], 'geo', 'apex')
        ax1.plot(mlon, mlat, '-', c=cmap(i))
        # G07 GPS
        pp = g07_ipp.sel(time=time_slice, rx=rx)
        pp = pp[pp.notnull().all(dim='component')]
        mlat, mlon = apex_converter.convert(pp.sel(component='lat').values[::60], pp.sel(component='lon').values[::60], 'geo', 'apex')
        ax1.plot(mlon, mlat, '-', c=cmap(i))
    ax1.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    ax1.set_xlabel('')
    ax1.grid()
    # tec line plot
    ax2 = fig.add_subplot(gs[4])
    ax3 = fig.add_subplot(gs[5], sharex=ax2)
    for i, rx in enumerate(g30_ipp.rx.values):
        if g30_tec.sel(time=time_slice, rx=rx).notnull().sum() > 10:
            g30_tec.sel(time=time_slice, rx=rx).plot.line(ax=ax2, x='time', c=cmap(i))
        if g07_tec.sel(time=time_slice, rx=rx).notnull().sum() > 10:
            g07_tec.sel(time=time_slice, rx=rx).plot.line(ax=ax3, x='time', c=cmap(i))
    ax2.set_ylim([0, 15])
    ax2.grid()
    ax2.set_title('')
    ax2.set_ylabel('G30')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax3.set_ylim([0, 15])
    ax3.grid()
    ax3.set_title('')
    ax3.set_ylabel('G07')
    ax3.set_xlabel('')
    ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.suptitle(time_str)
    plt.savefig(f"E:\\tec_data\\plots\\mahali\\{time_str}.png")
    plt.close(fig)
