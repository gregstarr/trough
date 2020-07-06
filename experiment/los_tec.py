import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File("C:\\Users\\Greg\\Downloads\\los_20181001.001.h5", 'r') as f:
    data = f['Data/Table Layout'][()]

# data = data[:100000]

ut = data['ut1_unix']
tec = data['tec']
el = data['elm']
lat = data['gdlatr']
lon = data['gdlonr']
lon[lon > 180] -= 360

unique_ut, unique_ut_inv = np.unique(ut, return_inverse=True)


times = ut.astype(int) * np.timedelta64(1, 's') + np.datetime64("1970-01-01T00:00:00")
el_mask = el >= 70
lat_mask = lat >= 20
lon_mask = (lon >= -170) * (lon <= -50)
zero_mask = tec > 0
dt = 600
i = ut[0] + dt
fig = plt.figure(figsize=(20, 12))
plt.grid()
plt.xlim([-170, -50])
plt.ylim([20, 70])
while i < ut.max():
    print(i)
    time_mask = (ut >= i - dt) * (ut <= i)
    i += dt
    mask = el_mask * lat_mask * lon_mask * time_mask * zero_mask
    plt.scatter(lon[mask], lat[mask], s=4, c=tec[mask], vmin=0, vmax=12)
    plt.savefig(f"E:\\tec_data\\plots\\high elevation madrigal\\{i}.png")
    # plt.close(fig)
