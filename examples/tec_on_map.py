import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import trough

data_paths = "E:\\tec_data\\data\\*.nc"
plot_directory = "E:\\tec_data\\plots"
print("Loading dataset")
tec_data = xr.open_mfdataset(data_paths, combine='by_coords')['tec']

mp = trough.MapPlot()
tec_feature = trough.TimeAverageTecMap(dict(average=np.timedelta64(60, 'm'), vmax=25), tec_data)
mp.updates.append(tec_feature)

date = np.datetime64('2018-10-30T00:00:00')
dt = np.timedelta64(5, 'm')
print("Plotting...")
for i in range(288):
    print(date)
    mp.plot_date(date)
    mp.save_fig(plot_directory, date)
    date += dt
