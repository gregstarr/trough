import numpy as np
import xarray as xr
from scipy import interpolate
import traceback

from trough import utils


def find_troughs_swarm(data):
    """
        Aa 2020 - Statistical Analysis of the Main Ionospheric Trough Using Swarm in Situ Measurements

        Method:
            - 3 point median filter
            - cut into segments between 45 and 75
            - calculate background density as a 480 point rolling window average
            - detrend Ne using the background density
            - check detrended electron density for a negative peak that drops below 50% of background density
            - mark poleward and equatorward transitions back to background level as the walls
            - filter out troughs which are outside the width range [1, 18] degrees
            - choose the lower trough in case there are two

    Parameters
    ----------
    data: xr.DataSet
            each variable is time x satellite so (T x 3)

    Returns
    -------
    troughs: xr.DataArray
    """
    median_filtered = np.log(data['n'].rolling(time=3, center=True, min_periods=1).median())
    detrended = median_filtered - median_filtered.rolling(time=481, center=True, min_periods=50).mean()
    troughs = {}
    for sat in data.satellite.values:
        sat_trough_params = {key: [] for key in ['min', 'poleward', 'equatorward', 'mlon', 'mlt', 'trough']}
        sat_trough_time = []

        ne = detrended.sel(satellite=sat)
        sat_data = data.sel(satellite=sat).chunk({'time': -1}).interpolate_na(dim='time', method='linear')
        mlat = sat_data['mlat']
        mlon = sat_data['mlon']
        mlt = sat_data['mlt']

        entering, exiting = determine_segments(mlat)
        print("SEGMENTS: ", entering.shape[0])

        i = 0
        for start, end in zip(entering, exiting):
            i += 1
            try:
                segment_ne = ne[start:end].values
                fin_mask = np.isfinite(segment_ne)
                segment_ne = segment_ne[fin_mask]
                segment_mlat = mlat[start:end].values[fin_mask]
                segment_mlon = mlon[start:end].values[fin_mask]
                segment_mlt = mlt[start:end].values[fin_mask]
                segment_time = ne[start:end].time.values[fin_mask]
                trough_params = analyze_segment(segment_ne, segment_mlat, segment_time, segment_mlon, segment_mlt)
                sat_trough_time.append(trough_params['time'])
                for param in sat_trough_params:
                    sat_trough_params[param].append(trough_params[param])
            except Exception as e:
                print("ERROR")
                print("ERROR")
                print("ERROR")
                traceback.print_exc()
                continue
            if not i % (entering.shape[0] // 1000):
                print(100 * i / entering.shape[0], '%')

        sat_troughs = xr.Dataset({key: xr.DataArray(val, dims=['time'], coords={'time': sat_trough_time})
                                  for key, val in sat_trough_params.items()})
        troughs[sat] = sat_troughs
    return troughs


def determine_segments(mlat):
    in_region = (mlat >= 45) * (mlat <= 75)
    boundary = np.diff(in_region.values.astype(int))
    entering = np.argwhere(boundary == 1)[:, 0]
    exiting = np.argwhere(boundary == -1)[:, 0]
    segments = min(entering.shape[0], exiting.shape[0])
    assert np.all((exiting[:segments] - entering[:segments]) > 0)
    return entering[:segments], exiting[:segments]


def analyze_segment(ne, mlat, time, mlon, mlt):
    mlat, uidx = np.unique(mlat, return_index=True)
    ne = ne[uidx]
    time = time[uidx]
    sorter = np.argsort(mlat)
    mlat = mlat[sorter]
    ne = ne[sorter]
    time = time[sorter]
    spline = interpolate.UnivariateSpline(mlat, ne, s=.25)
    roots = spline.roots()
    for i in range(roots.shape[0] - 1):
        width = roots[i+1] - roots[i]
        if not 1 <= width <= 17:
            continue
        mask = (mlat >= roots[i]) * (mlat <= roots[i+1])
        if not mask.any():
            continue
        min_idx = ne[mask].argmin()
        min_ne = ne[mask][min_idx]
        if min_ne > -0.3:
            continue
        idx = np.argwhere(mask)[min_idx, 0]

        # plt.style.use('ggplot')
        # mlat_ticks = [45, 50, 55, 60, 65, 70, 75]
        # time_ticks = [utils.datetime64_to_datetime(time[np.argmin(abs(mlat - m))]).strftime("%H:%M:%S") for m in mlat_ticks]
        # tick_labels = [f"{str(m)}\n{str(t)}" for m, t in zip(mlat_ticks, time_ticks)]
        # plt.plot(mlat, ne)
        # plt.plot(mlat, spline(mlat))
        # plt.plot(roots, np.zeros_like(roots), 'g.')
        # plt.plot([roots[i], roots[i+1]], [0, 0], 'g', alpha=.25)
        # plt.plot(mlat[idx], min_ne, 'kx', ms=10)
        # plt.xticks(ticks=mlat_ticks, labels=tick_labels)
        # plt.hlines(-0.3, 50, 80, linestyles='dashed')
        # plt.ylabel("Detrended Ne")
        # plt.title("SWARM Trough Detection (Aa 2020)")
        # plt.show()

        return {
            'min': mlat[idx],
            'mlon': mlon[idx],
            'mlt': mlt[idx],
            'time': time[idx],
            'poleward': roots[i+1],
            'equatorward': roots[i],
            'trough': True,
        }
    return {
        'min': mlat.mean(),
        'mlon': mlon.mean(),
        'mlt': mlt.mean(),
        'time': time[time.shape[0]//2],
        'poleward': np.nan,
        'equatorward': np.nan,
        'trough': False,
    }


if __name__ == "__main__":
    import xarray as xr
    import matplotlib.pyplot as plt
    import os

    swarm_dir = "E:\\swarm"
    swarm = xr.open_mfdataset(os.path.join(swarm_dir, "2018*swarm.nc"), concat_dim="time", combine="by_coords")
    troughs = find_troughs_swarm(swarm)
    for sat, sat_troughs in troughs.items():
        sat_troughs.to_netcdf(os.path.join(swarm_dir, f"2018_swarm_troughs_{sat}.nc"))
