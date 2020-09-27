import h5py
import numpy as np
from cartopy import crs
import matplotlib.pyplot as plt

# year month day hour min sec recno kindat kinst ut1_unix ut2_unix pierce_alt gps_site sat_id gdlatr gdlonr los_tec dlos_tec tec azm elm gdlat glon rec_bias drec_bias

with h5py.File("E:\\los_tec\\los_20181001.001.h5", 'r') as f:
    data = f['Data/Table Layout'][()]

lat = data['gdlatr']
lat, ind = np.unique(lat, return_index=True)
lon = data['gdlonr'][ind]

ax = plt.axes(projection=crs.PlateCarree())
ax.plot(lon, lat, 'r.', transform=crs.PlateCarree(), ms=2)
ax.coastlines()
ax.stock_img()
plt.show()
