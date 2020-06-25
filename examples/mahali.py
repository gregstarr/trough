import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs
import pymap3d as pm
import apexpy

import trough

data_paths = "E:\\tec_data\\data\\madrigal\\*.nc"
plot_directory = "E:\\tec_data\\plots\\maps"
rx_name = "MAH7"
rx_array = trough.gps.ReceiverArray()
rx = rx_array.receivers[rx_name]
print("Loading TEC data")
tec_data = xr.open_mfdataset(data_paths, combine='by_coords', parallel=True)['tec'].sel(time=rx_array.obs_time_range)

mlon = np.arange(-120, -69)
mlat = np.arange(50, 81)
mlon_grid, mlat_grid = np.meshgrid(mlon, mlat)
apex_converter = apexpy.Apex(date=trough.utils.datetime64_to_datetime(rx_array.obs_times[0]))
print("Converting")
glat, glon = apex_converter.convert(mlat_grid.ravel(), mlon_grid.ravel(), 'apex', 'geo')
glat_grid = glat.reshape(mlat_grid.shape)
glon_grid = glon.reshape(mlat_grid.shape)
lat = xr.DataArray(glat_grid, dims=["mlat", "mlon"], coords={"mlon": mlon, "mlat": mlat})
lon = xr.DataArray(glon_grid, dims=["mlat", "mlon"], coords={"mlon": mlon, "mlat": mlat})
print("interpolating")

rx_posg = rx.obs.position_geodetic
rx_pose = rx.position.values
rx_posm = apex_converter.convert(rx_posg[0], rx_posg[1], 'geo', 'apex')

time = rx.obs.time.values[0]
dt = np.timedelta64(30, 'm')
while time < rx.obs.time.values[-1]:
    time_str = str(time).replace('-', '').replace(':', '')[:15]
    print(time_str)
    time_slice = slice(time, time+dt)
    time += dt
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot()
    ax.set_title(time_str)
    remapped = tec_data.sel(time=time_slice).mean(dim='time').interp(longitude=lon, latitude=lat)
    remapped.plot.pcolormesh(ax=ax, vmin=0, vmax=12)
    ax.plot(rx_posm[1], rx_posm[0], 'rx')
    for sv in rx_array.svs:
        for alt in [200, 300, 400, 500, 600]:
            rx.update_pierce_points(rx_array.satellite_positions, alt)
            pp = rx.pierce_points[sv].sel(time=time_slice)
            pp = pp.where(pp.sel(component='el') > 30)
            if pp.size < 60:
                break
            mlat, mlon = apex_converter.convert(pp.sel(component='lat').values, pp.sel(component='lon').values, 'geo', 'apex')
            ax.plot(mlon, mlat, 'k.')
            ax.text(mlon[-1]+.1, mlat[-1], str(alt))
    plt.savefig(f"E:\\tec_data\\plots\\IPP\\{time_str}.png")
    plt.close(fig)
