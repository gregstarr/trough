import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import apexpy

import trough
from trough.gps import ReceiverArray

data_paths = "E:\\tec_data\\data\\madrigal\\201510*.nc"
print("Loading dataset")
rx_array = ReceiverArray()
madrigal = xr.open_mfdataset(data_paths, combine='by_coords', parallel=True)['tec'].sel(time=rx_array.obs_time_range)
madrigal.load()
rx_array.align_with_madrigal_tec_map(madrigal)

print("Plotting")
fig, ax = plt.subplots(1, 2, sharey=True)
rx_array.add_plot(ax[0], rx_array.plot_tec_ipp, np.timedelta64(24, 'h'), min_el=20, mag_coords='mlt', svs=['G30'])
rx_array.add_plot(ax[1], rx_array.plot_tec_ipp, np.timedelta64(24, 'h'), min_el=20, mag_coords='apex', svs=['G30'])
rx_array.plot_date(rx_array.obs.time.values[0])
plt.show()
