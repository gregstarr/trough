import numpy as np
import h5py
import os
import glob
import apexpy
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

from trough import utils, convert, plotting


def convert_average(tec, timestamps, lat, lon, bins, nmaps=12):
    long, latg = np.meshgrid(lon, lat)
    converter = apexpy.Apex(date=utils.timestamp_to_datetime(timestamps[0]))
    mag_tec = np.empty((len(bins[0]) - 1, len(bins[1]) - 1, tec.shape[-1] // nmaps))
    for m in range(tec.shape[-1] // nmaps):
        all_mlat = []
        all_mlt = []
        all_tec = []
        for i in range(nmaps):
            mask = np.isfinite(tec[:, :, m * nmaps + i].ravel())
            mlat, mlt = converter.convert(latg.ravel()[mask], long.ravel()[mask], 'geo', 'mlt', height=350,
                                          datetime=utils.timestamp_to_datetime(timestamps[m * nmaps + i]))
            all_mlat.append(mlat)
            all_mlt.append(mlt)
            all_tec.append(tec[:, :, m * nmaps + i].ravel()[mask])
        mlat = np.concatenate(all_mlat)
        mlt = np.concatenate(all_mlt)
        converted_tec = np.concatenate(all_tec)
        mag_tec[:, :, m] = binned_statistic_2d(mlat, mlt, converted_tec, bins=bins).statistic
    return mag_tec


def process_month(files, bins, output_pattern, nmaps=12):
    if not files:
        return
    mlat = (bins[0][:-1] + bins[0][1:]) / 2
    tec_maps = []
    times = []
    for file in files:
        print(file)
        with h5py.File(file, 'r') as f:
            tec = f['Data']['Array Layout']['2D Parameters']['tec'][()]
            timestamps = f['Data']['Array Layout']['timestamps'][()]
            lat = f['Data']['Array Layout']['gdlat'][()]
            lon = f['Data']['Array Layout']['glon'][()]
        if np.isnan(tec).all():
            continue
        coarsened_timestamps = timestamps[::nmaps]
        tec_maps_day = convert_average(tec, timestamps, lat, lon, bins, nmaps=nmaps)
        tec_maps.append(tec_maps_day[mlat >= 30])
        times.append(coarsened_timestamps)
    tec = np.concatenate(tec_maps, axis=-1)
    start_time = np.concatenate(times)
    time = utils.timestamp_to_datetime(start_time.mean())
    with h5py.File(output_pattern.format(year=time.year, month=time.month), 'w') as f:
        f.create_dataset('tec', data=tec)
        f.create_dataset('start_time', data=start_time)
        f.create_dataset('labels', data=np.zeros_like(tec, dtype=bool))


bins = [np.arange(-90.5, 90), np.arange(-12 / 360, 24, 24 / 360)]
mlat = (bins[0][:-1] + bins[0][1:]) / 2
mlt = (bins[1][:-1] + bins[1][1:]) / 2

download_dir = os.path.join("E:", "tec_data", "download")
pattern = "gps{year:02d}{month:02d}*.hdf5"
h5_path = os.path.join(download_dir, pattern)
output_dir = os.path.join("E:", "tec_data")
output_pattern = "{year:04d}_{month:02d}_tec.h5"
output_path = os.path.join(output_dir, output_pattern)

# make grid file
with h5py.File(os.path.join(output_dir, 'grid.h5'), 'w') as f:
    f.create_dataset('mlat', data=mlat[mlat >= 30])
    f.create_dataset('mlt', data=mlt)

# make dset
for year in range(19, 20):
    for month in range(1, 2):
        print(f"YEAR: {year}, MONTH: {month}")
        process_month(glob.glob(h5_path.format(year=year, month=month)), bins, output_path)
