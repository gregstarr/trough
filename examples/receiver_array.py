import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import trough
plt.style.use('ggplot')


data_paths = "E:\\tec_data\\data\\madrigal\\201510*.nc"

rx_array = trough.gps.ReceiverArray()
madrigal = xr.open_mfdataset(data_paths, combine='by_coords', parallel=True)['tec'].sel(time=rx_array.obs_time_range)
madrigal.load()
sv = 'G30'
pierce_points = rx_array.pierce_points.sel(sv=sv, component=['lat', 'lon'])
mapping = rx_array.mapping_func().sel(sv=sv)
mahali = rx_array.stec.sel(sv=sv)
cmap = plt.cm.get_cmap('tab10', len(pierce_points.rx.values))
for i, rx in enumerate(pierce_points.rx.values):
    print(rx)
    ipp = pierce_points.sel(rx=rx)
    mask = ipp.notnull().all(dim='component')
    ipp = ipp[mask]
    mah = mahali.sel(rx=rx)[mask]
    M = mapping.sel(rx=rx)[mask]
    interp_mad = madrigal.interp(time=ipp.time, latitude=ipp.sel(component='lat'), longitude=ipp.sel(component='lon'), method='nearest')
    interp_mad /= M
    mah += (interp_mad - mah).mean(dim='time')
    mah -= np.minimum(mah.min(dim='time'), 0)
    mah *= M
    mah.plot(x='time', c=cmap(i), label=rx)
plt.legend()
plt.title(f"VTEC, {sv}")
plt.show()
