"""
go through whole dataset and plot how much data there is per year
    - average coverage per year
    - average coverage over longitude per year
    - full map of coverage per year
    - unprocessed and processed
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import glob
from teclab import config, utils
plt.style.use('ggplot')


download_dir = os.path.join("E:", "tec_data", "download")
pattern = "gps{year:02d}*.hdf5"
h5_path = os.path.join(download_dir, pattern)


n_finite = np.zeros((21, 180, 360))
n_total = np.zeros((21, 180, 360))
for year in range(21):
    print(year)
    files = glob.glob(h5_path.format(year=year))
    for fn in files[::10]:
        try:
            with h5py.File(fn, 'r') as f:
                tec = f['Data']['Array Layout']['2D Parameters']['tec'][()]
                lat = f['Data']['Array Layout']['gdlat'][()]
                lon = f['Data']['Array Layout']['glon'][()]
            lat_sl = slice(int(lat[0] + 90), int(lat[0] + 90 + lat.size))
            lon_sl = slice(int(lon[0] + 180), int(lon[0] + 180 + lon.size))
            for i in range(tec.shape[-1]//12):
                n_finite[year, lat_sl, lon_sl] += np.isfinite(tec[:, :, i * 12:(i + 1)*12]).any(axis=-1)
                n_total[year, lat_sl, lon_sl] += 1
        except:
            print('fail', fn)

coverage = n_finite / n_total
coverage_by_lon = n_finite[:, 120:].sum(axis=1) / n_total[:, 120:].sum(axis=1)

colors = plt.cm.Blues(np.linspace(.5, 1, 21))
for i in range(21):
    plt.plot(np.arange(-180, 180), coverage_by_lon[i], color=colors[i])
plt.xlabel('Longitude')
plt.ylabel('Coverage')
plt.title('Coverage by Year and Longitude')
plt.show()
