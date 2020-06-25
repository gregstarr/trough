import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from trough.gps import ReceiverArray
from trough.date_features import TimeAverageTecMap, GpsSatelliteFeature

data_paths = "E:\\tec_data\\data\\madrigal\\*.nc"
plot_directory = "E:\\tec_data\\plots\\maps"
print("Loading dataset")
tec_data = xr.open_mfdataset(data_paths, combine='by_coords')['tec']
rx_array = ReceiverArray()

mp = MapPlot()
tec_feature = TimeAverageTecMap(dict(vmax=18), tec_data)
mp.updates.append(tec_feature)

for sv in rx_array.svs:
    gps_feature = GpsSatelliteFeature(dict(sv=sv), rx_array)
    mp.updates.append(gps_feature)

date = np.datetime64(rx_array.obs_times[0], dtype='datetime64[s]')
dt = np.timedelta64(30, 'm')
print("Plotting...")
for i in range(1):#(2*24*10):
    print(date)
    date_range = slice(date, date + dt)
    mp.plot_date_range(date_range)
    mp.save_fig(plot_directory, date)
    date += dt
