import numpy as np
import xarray as xr
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from skimage import filters


def get_troughs(features, lr, high_th=.7, low_th=.6):
    probabilty = lr.predict_proba(features.values.reshape((-1, 4)))[:, 1].reshape(features.shape[:-1])
    troughs = probabilty > high_th
    # troughs = MiniBatchKMeans(n_clusters=2).fit_predict(features.values.reshape((-1, 4))).reshape(features.shape[:-1])
    # troughs = np.zeros_like(probabilty)
    # for i in range(probabilty.shape[0]):
    #     troughs[i] = filters.apply_hysteresis_threshold(probabilty[i], high_th, low_th)
    troughs = xr.DataArray(troughs, dims=['time', 'mlat', 'mlon'],
                           coords={'time': features.time.values,
                                   'mlat': features.mlat.values,
                                   'mlon': features.mlon.values})
    return troughs


def get_lr(features, troughs, dt=np.timedelta64(60, 'm'), dmlon=5, mlat_extra=1):
    dmlt = dmlon * 24 / 360
    X = []
    y = []
    dbg = {}
    dbg_temp = {'low_mlon': [], 'high_mlon': [], 'pwall': [], 'ewall': [], 'trough': [], 'low_mlt': [], 'high_mlt': []}
    for sat, sat_troughs in troughs.items():
        for t in sat_troughs.time.values:
            trough = sat_troughs.sel(time=t)
            profile = features.sel(time=slice(t - dt, t + dt),
                                   mlon=slice(trough['mlon'].item() - dmlon, trough['mlon'].item() + dmlon))
            mask = np.zeros(profile.shape[:-1], dtype=bool)
            if trough['trough'].item():
                mlat_mask = (profile.mlat.values <= trough['poleward'].item() + mlat_extra) * (
                        profile.mlat.values >= trough['equatorward'].item() - mlat_extra)
                mask[:, mlat_mask] = True
            X.append(profile.values.reshape((-1, profile.shape[-1])))
            y.append(mask.ravel())
            for prof_time in profile.time.values:
                if prof_time not in dbg:
                    dbg[prof_time] = copy.deepcopy(dbg_temp)
                dbg[prof_time]['low_mlon'].append(trough['mlon'].item() - dmlon)
                dbg[prof_time]['high_mlon'].append(trough['mlon'].item() + dmlon)
                dbg[prof_time]['low_mlt'].append(trough['mlt'].item() - dmlt)
                dbg[prof_time]['high_mlt'].append(trough['mlt'].item() + dmlt)
                if trough['trough'].item():
                    dbg[prof_time]['pwall'].append(trough['poleward'].item() + mlat_extra)
                    dbg[prof_time]['ewall'].append(trough['equatorward'].item() - mlat_extra)
                    dbg[prof_time]['trough'].append(True)
                else:
                    dbg[prof_time]['pwall'].append(np.nan)
                    dbg[prof_time]['ewall'].append(np.nan)
                    dbg[prof_time]['trough'].append(False)
    dbg = xr.Dataset({f: xr.DataArray([d[f] for d in dbg.values()], dims=['time'],
                                      coords={'time': list(dbg.keys())}) for f in dbg_temp})
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    lr = LogisticRegression(C=5, class_weight='balanced')
    lr.fit(X, y)
    return lr, dbg


def get_features(tec_map):
    """
        - curvatuyre
        - low val
        - pwall
        - e wall
        - mlt*
        - lat*
        - doy*
        - other distances for walls

    Parameters
    ----------
    tec_map: xr.DataArray (time x lon x lat)
            unmodified tec map

    Returns
    -------
    features: xr.DataArray (time x lon x lat x channel)
    """
    centered = tec_map - tec_map.rolling(mlon=7, center=True, min_periods=True).mean().sel(mlat=slice(40, 80)).mean(dim='mlat')
    coarse = centered.coarsen(mlon=2, mlat=2).mean()
    smoothed = coarse.rolling(mlon=3, center=True, min_periods=1).mean().sel(mlat=slice(40, 80))
    # smoothed = smoothed.interp(mlon=tec_map.mlon.values, mlat=np.arange(40, 80), method='nearest')
    dlat = smoothed.differentiate('mlat')
    dlat_dlat = dlat.differentiate('mlat')
    dlat_dlat = np.where(np.isfinite(dlat_dlat), dlat_dlat, 0)
    low_val = np.maximum(0, -1 * smoothed.values)
    low_val = np.where(np.isfinite(low_val), low_val, 0)
    shift = 3
    p_edge = np.pad(dlat, ((0, 0), (0, shift), (0, 0)), mode='constant', constant_values=0)[:, shift:, :]
    p_edge = np.where(np.isfinite(p_edge), p_edge, 0)
    e_edge = -1 * np.pad(dlat, ((0, 0), (shift, 0), (0, 0)), mode='constant', constant_values=0)[:, :-shift, :]
    e_edge = np.where(np.isfinite(e_edge), e_edge, 0)
    feature_array = np.stack((dlat_dlat, low_val, p_edge, e_edge), axis=-1)
    features = xr.DataArray(feature_array, dims=['time', 'mlat', 'mlon', 'channel'],
                            coords={'time': tec_map.time.values,
                                    'mlat': smoothed.mlat.values,
                                    'mlon': smoothed.mlon.values,
                                    'channel': ['d2lat', 'low', 'pwall', 'ewall']})
    return features


if __name__ == "__main__":
    from trough.satellite import find_troughs_swarm
    tec_map = xr.open_mfdataset("E:\\tec_data\\data\\dataset 1\\*madrigal.nc", concat_dim="time", combine="by_coords")['tec']
    swarm = xr.open_mfdataset("E:\\swarm\\2018*swarm.nc", concat_dim="time", combine="by_coords")
    features = get_features(tec_map)
    swarm_troughs = find_troughs_swarm(swarm)
    lr, diag = get_lr(features, swarm_troughs)
    print(lr.coef_)
