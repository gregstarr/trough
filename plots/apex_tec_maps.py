"""
Plotting script:
    - single subplot
    - apex coordinates
    - TEC map
    - IPP trajectories showing TEC map artifacts
    - substorms*
    - moving MLT lines*
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from trough import plotting

madrigal_paths = "E:\\tec_data\\data\\madrigal\\*.nc"
print("Loading dataset")
madrigal = xr.open_mfdataset(madrigal_paths, combine='by_coords', parallel=True)['tec']
sat_pos = xr.open_dataarray("E:\\tec_data\\data\\nav\\sat_pos_1min.nc")

average_period = np.timedelta64(20, 'm')
time_step = np.timedelta64(10, 'm')

print("Plotting")
fig, ax1 = plt.subplots(1, 1, figsize=(20, 12))


date = madrigal.sel(time='2018').time.values[0]
for i in range(madrigal.time.size):
    time_str = str(date).replace('-', '').replace(':', '')[:15]
    print(time_str)
    plotting.plot_tec_map_mag(ax1, date, madrigal, y_arange_args=(20, 85), x_arange_args=(-120, 20), dt=average_period,
                              vmin=0, vmax=12)
    for i in range(32):
        plotting.plot_sv_ipp_mag(ax1, date, np.array((37.44, -120.23, 0.)), sat_pos.isel(sv=i), dt=average_period,
                                 min_el=30, color='white')
    plotting.plot_coastline_mag(ax1, date, color='black')
    date += time_step
    ax1.set_title(time_str)
    ax1.set_ylim(20, 85)
    ax1.set_xlim(-120, 20)
    ax1.grid()
    plt.savefig(f"E:\\tec_data\\plots\\artifact\\{time_str}.png", fig=fig)
    ax1.clear()
