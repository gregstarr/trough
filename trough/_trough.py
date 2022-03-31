import numpy as np
import xarray as xr
import multiprocessing
from scipy.sparse import csr_matrix
from pathlib import Path
from apexpy import Apex
from datetime import datetime, timedelta
import logging
import warnings
try:
    import cvxpy as cp
    import bottleneck as bn
    import pandas
    from skimage import measure, morphology
    from sklearn.metrics.pairwise import rbf_kernel
except ImportError as imp_err:
    warnings.warn(f"Packages required for recreating dataset not installed: {imp_err}")

from trough import config, utils, _tec, _arb


logger = logging.getLogger(__name__)


def get_model(tec_data, hemisphere, omni_file):
    """Get magnetic latitudes of the trough according to the model in Deminov 2017
    for a specific time and set of magnetic local times.
    """
    logger.info("getting model")
    omni_data = xr.open_dataset(omni_file)
    logger.info(f"{omni_data.time.values[0]=} {omni_data.time.values[-1]=}")
    kp = _get_weighted_kp(tec_data.time, omni_data)
    logger.info(f"{kp.shape=}")
    apex = Apex(date=utils.datetime64_to_datetime(tec_data.time.values[0]))
    mlat = 65.5 * np.ones((tec_data.time.shape[0], tec_data.mlt.shape[0]))
    if hemisphere == 'south':
        mlat = mlat * -1
    for i in range(10):
        glat, glon = apex.convert(mlat, tec_data.mlt.values[None, :], 'mlt', 'geo', 350, tec_data.time.values[:, None])
        mlat = _model_subroutine_lat(tec_data.mlt.values[None, :], glon, kp[:, None], hemisphere)
    tec_data['model'] = xr.DataArray(
        mlat,
        coords={'time': tec_data.time, 'mlt': tec_data.mlt},
        dims=['time', 'mlt']
    )


def _model_subroutine_lat(mlt, glon, kp, hemisphere):
    """Get's model output mlat given MLT, geographic lon and weighted kp

    Parameters
    ----------
    mlt: numpy.ndarray (n_mlt, )
    glon: numpy.ndarray (n_mlt, )
    kp: float

    Returns
    -------
    mlat: numpy.ndarray (n_t, n_mlt)
    """
    phi_t = 3.16 - 5.6 * np.cos(np.deg2rad(15 * (mlt - 2.4))) + 1.4 * np.cos(np.deg2rad(15 * (2 * mlt - .8)))
    if hemisphere == 'north':
        phi_lon = .85 * np.cos(np.deg2rad(glon + 63)) - .52 * np.cos(np.deg2rad(2 * glon + 5))
    elif hemisphere == 'south':
        phi_lon = 1.5 * np.cos(np.deg2rad(glon - 119))
    else:
        raise ValueError(f"Invalid hemisphere: {hemisphere}, valid = ['north', 'south']")
    return 65.5 - 2.4 * kp + phi_t + phi_lon * np.exp(-.3 * kp)


def _get_weighted_kp(times, omni_data, tau=.6, T=10):
    """Get a weighed sum of kp values over time. See paper for details.
    """
    logger.info(f"_get_weighted_kp {times[0]=} {times[-1]=}")
    ap = omni_data.sel(time=slice(times[0] - np.timedelta64(T, 'h'), times[-1]))['ap'].values
    prehistory = np.column_stack([ap[T - i:ap.shape[0] - i] for i in range(T)])
    weight_factors = tau ** np.arange(T)
    ap_tau = np.sum((1 - tau) * prehistory * weight_factors, axis=1)
    return 2.1 * np.log(.2 * ap_tau + 1)


def estimate_background(tec, patch_shape):
    """Use a moving average filter to estimate the background TEC value. `patch_shape` must contain odd numbers

    Parameters
    ----------
    tec: numpy.ndarray[float]
    patch_shape: tuple

    Returns
    -------
    numpy.ndarray[float]
    """
    assert all([2 * (p // 2) + 1 == p for p in patch_shape]), "patch_shape must be all odd numbers"
    patches = utils.extract_patches(tec, patch_shape)
    return bn.nanmean(patches.reshape((tec.shape[0] - patch_shape[0] + 1,) + tec.shape[1:] + (-1,)), axis=-1)


def preprocess_interval(data, min_val=0, max_val=100, bg_est_shape=(1, 15, 15)):
    logger.info("preprocessing interval")
    tec = data['tec'].values
    # throw away outlier values
    tec[tec > max_val] = np.nan
    tec[tec < min_val] = np.nan
    # change to log
    log_tec = np.log10(tec + .001)
    # estimate background
    bg = estimate_background(log_tec, bg_est_shape)
    # subtract background
    x = log_tec - bg
    coords = {'time': data.time, 'mlat': data.mlat, 'mlt': data.mlt}
    data['x'] = xr.DataArray(x, coords=coords, dims=['time', 'mlat', 'mlt'])
    data['tec'] = xr.DataArray(tec, coords=coords, dims=['time', 'mlat', 'mlt'])


def fix_boundaries(labels):
    fixed = labels.copy()
    while True:
        boundary_pairs = np.unique(fixed[:, [0, -1]], axis=0)
        if np.all((boundary_pairs[:, 0] == boundary_pairs[:, 1]) | np.any(boundary_pairs == 0, axis=1)):
            break
        for i in range(boundary_pairs.shape[0]):
            if np.any(boundary_pairs[i] == 0) or boundary_pairs[i, 0] == boundary_pairs[i, 1]:
                continue
            fixed[fixed == boundary_pairs[i, 1]] = boundary_pairs[i, 0]
            break
    return fixed


def remove_auroral(data, hemisphere, offset=3):
    if hemisphere == 'north':
        data['labels'] *= data.mlat < (data['arb'] + offset)
    elif hemisphere == 'south':
        data['labels'] *= data.mlat > (data['arb'] + offset)
    else:
        raise ValueError(f"Invalid hemisphere: {hemisphere}, valid = ['north', 'south']")


def postprocess(data, hemisphere, perimeter_th=50, area_th=1, closing_r=0):
    if closing_r > 0:
        selem = morphology.disk(closing_r, dtype=bool)[:, :, None]
        data['labels'] = np.pad(data['labels'], ((0, 0), (0, 0), (closing_r, closing_r)), 'wrap')
        data['labels'] = morphology.binary_closing(data['labels'], selem)[:, :, closing_r:-closing_r]
    remove_auroral(data, hemisphere)
    for t in range(data.time.shape[0]):
        tmap = data['labels'][t].values
        labeled = measure.label(tmap, connectivity=2)
        labeled = fix_boundaries(labeled)
        props = pandas.DataFrame(measure.regionprops_table(labeled, properties=('label', 'area', 'perimeter')))
        error_mask = (props['perimeter'] < perimeter_th) | (props['area'] < area_th)
        for i, r in props[error_mask].iterrows():
            tmap[labeled == r['label']] = 0
        data['labels'][t] = tmap


def get_rbf_matrix(shape, bandwidth=1):
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    gamma = np.log(2) / bandwidth ** 2
    basis = rbf_kernel(xy, gamma=gamma)
    basis[basis < .01] = 0
    return csr_matrix(basis)


def get_tv_matrix(im_shape, hw=1, vw=1):
    size = im_shape[0] * im_shape[1]

    right = np.eye(size)
    cols = im_shape[1] * (np.arange(size) // im_shape[1]) + (np.arange(size) + 1) % im_shape[1]
    right[np.arange(size), cols] = -1

    left = np.eye(size)
    cols = im_shape[1] * (np.arange(size) // im_shape[1]) + (np.arange(size) - 1) % im_shape[1]
    left[np.arange(size), cols] = -1

    up = np.eye(size)
    cols = np.arange(size) + im_shape[1]
    mask = cols < size
    cols = cols[mask]
    up[np.arange(size)[mask], cols] = -1

    down = np.eye(size)
    cols = np.arange(size) - im_shape[1]
    mask = cols >= 0
    cols = cols[mask]
    down[np.arange(size)[mask], cols] = -1
    return csr_matrix(hw * (right + left) + vw * (up + down))


def get_optimization_args(data, model_weight_max, rbf_bw, tv_hw, tv_vw, l2_weight, tv_weight):
    all_args = []
    # get rbf basis matrix
    basis = get_rbf_matrix((data.mlat.shape[0], data.mlt.shape[0]), rbf_bw)
    # get tv matrix
    tv = get_tv_matrix((data.mlat.shape[0], data.mlt.shape[0]), tv_hw, tv_vw) * tv_weight
    for i in range(data.time.shape[0]):
        # l2 norm cost away from model
        l2 = abs(data.mlat - data['model']).isel(time=i)
        l2 -= l2.min()
        l2 = (model_weight_max - 1) * l2 / l2.max() + 1
        l2 *= l2_weight
        fin_mask = np.isfinite(np.ravel(data['x'].isel(time=i)))
        if not fin_mask.any():
            raise Exception("WHY ALL NAN??")
        args = (cp.Variable(data.mlat.shape[0] * data.mlt.shape[0]), basis[fin_mask, :],
                np.ravel(data['x'].isel(time=i))[fin_mask], tv, np.ravel(l2))
        all_args.append(args)
    return all_args


def run_single(u, basis, x, tv, l2):
    main_cost = u.T @ basis.T @ x
    tv_cost = cp.norm1(tv @ u)
    l2_cost = l2 @ (u ** 2)
    total_cost = main_cost + tv_cost + l2_cost
    prob = cp.Problem(cp.Minimize(total_cost))
    try:
        prob.solve(solver=cp.GUROBI)
    except Exception as e:
        logger.info(f"FAILED, USING ECOS: {e}")
        prob.solve(solver=cp.ECOS)
    return u.value


def run_multiple(args, parallel=True):
    if parallel:
        with multiprocessing.Pool(processes=4) as p:
            results = p.starmap(run_single, args)
    else:
        results = []
        for arg in args:
            results.append(run_single(*arg))
    return np.stack(results, axis=0)


def label_trough_interval(start_date, end_date, params, hemisphere, tec_dir, arb_dir, omni_file):
    logger.info(f"labeling trough interval: {start_date=} {end_date=}")
    data = _tec.get_tec_data(start_date, end_date, hemisphere, tec_dir).to_dataset(name='tec')
    preprocess_interval(data, bg_est_shape=params.bg_est_shape)

    data = data.merge(_arb.get_arb_data(start_date, end_date, hemisphere, arb_dir))
    get_model(data, hemisphere, omni_file)
    args = get_optimization_args(data, params.model_weight_max, params.rbf_bw, params.tv_hw, params.tv_vw,
                                 params.l2_weight, params.tv_weight)
    logger.info("Running inversion optimization")
    data['score'] = xr.DataArray(
        run_multiple(args).reshape((data.time.shape[0], data.mlat.shape[0], data.mlt.shape[0])),
        coords={
            'time': data.time,
            'mlat': data.mlat,
            'mlt': data.mlt
        },
        dims=['time', 'mlat', 'mlt']
    )
    # threshold
    data['labels'] = data['score'] >= params.threshold
    # postprocess
    logger.info("Postprocessing inversion results")
    postprocess(data, hemisphere, params.perimeter_th, params.area_th, params.closing_rad)
    return data


def label_trough_dataset(start_date, end_date, params=None, tec_dir=None, arb_dir=None, omni_file=None,
                         output_dir=None):
    if params is None:
        params = config.trough_id_params
    if tec_dir is None:
        tec_dir = config.processed_tec_dir
    if arb_dir is None:
        arb_dir = config.processed_arb_dir
    if omni_file is None:
        omni_file = config.processed_omni_file
    if output_dir is None:
        output_dir = config.processed_labels_dir

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for year in range(start_date.year, end_date.year + 1):
        for hemisphere in ['north', 'south']:
            labels = []
            scores = []
            start = datetime(year, 1, 1, 0, 0)
            while start.year < year + 1:
                end = start + timedelta(days=1)
                if start >= end_date or end <= start_date:
                    start += timedelta(days=1)
                    continue
                start = max(start_date, start)
                end = min(end_date, end)
                data = label_trough_interval(start, end, params, hemisphere, tec_dir, arb_dir, omni_file)
                labels.append(data['labels'])
                scores.append(data['score'])
                start += timedelta(days=1)
            labels = xr.concat(labels, 'time')
            scores = xr.concat(scores, 'time')
            labels.to_netcdf(Path(output_dir) / f"labels_{hemisphere}_{year:04d}.nc")
            scores.to_netcdf(Path(output_dir) / f"scores_{hemisphere}_{year:04d}.nc")


def get_label_paths(start_date, end_date, hemisphere, processed_dir):
    file_dates = np.arange(
        np.datetime64(start_date, 'Y'),
        (np.datetime64(end_date, 's') - np.timedelta64(1, 'h')).astype('datetime64[Y]') + 1,
        np.timedelta64(1, 'Y')
    )
    file_dates = utils.decompose_datetime64(file_dates)
    return [Path(processed_dir) / f"labels_{hemisphere}_{d[0]:04d}.nc" for d in file_dates]


def get_trough_labels(start_date, end_date, hemisphere, labels_dir=None):
    if labels_dir is None:
        labels_dir = config.processed_labels_dir
    data = xr.concat([xr.open_dataarray(file) for file in get_label_paths(start_date, end_date, hemisphere, labels_dir)], 'time')
    return data.sel(time=slice(start_date, end_date))


def get_data(start_date, end_date, hemisphere, tec_dir=None, omni_file=None, labels_dir=None):
    if tec_dir is None:
        tec_dir = config.processed_tec_dir
    if omni_file is None:
        omni_file = config.processed_omni_file
    if labels_dir is None:
        labels_dir = config.processed_labels_dir
    data = xr.open_dataset(omni_file).sel(time=slice(start_date, end_date))
    data['tec'] = _tec.get_tec_data(start_date, end_date, hemisphere, tec_dir)
    data['labels'] = get_trough_labels(start_date, end_date, hemisphere, labels_dir)
    return data
