import numpy as np
import bottleneck as bn
import pandas
from skimage import measure, morphology
import cvxpy as cp
import multiprocessing
from scipy.sparse import csr_matrix
from pathlib import Path

from trough import config
import trough._tec as trough_tec
import trough._aux_data as trough_data
import trough._model as trough_model
import trough.utils as trough_utils


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


def preprocess_interval(tec, times, min_val=0, max_val=100, bg_est_shape=(3, 15, 15)):
    tec = tec.copy()
    # throw away outlier values
    tec[tec > max_val] = np.nan
    tec[tec < min_val] = np.nan
    # change to log
    log_tec = np.log10(tec + .001)
    # estimate background
    bg = estimate_background(log_tec, bg_est_shape)
    # subtract background
    log_tec, t, = utils.moving_func_trim(bg_est_shape[0], log_tec, times)
    x = log_tec - bg
    return x, t


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


def remove_auroral(inp, arb, offset=3):
    output = inp.copy()
    output *= (config.mlat_grid[None, :, :] < arb[:, None, :] + offset)
    return output


def postprocess(initial_trough, perimeter_th=50, area_th=1, arb=None, closing_r=0):
    trough = initial_trough.copy()
    if closing_r > 0:
        selem = morphology.disk(closing_r, dtype=bool)[:, :, None]
        trough = np.pad(trough, ((0, 0), (0, 0), (closing_r, closing_r)), 'wrap')
        trough = morphology.binary_closing(trough, selem)[:, :, closing_r:-closing_r]
    if arb is not None:
        trough = remove_auroral(trough, arb)
    for t in range(trough.shape[0]):
        tmap = trough[t]
        labeled = measure.label(tmap, connectivity=2)
        labeled = fix_boundaries(labeled)
        props = pandas.DataFrame(measure.regionprops_table(labeled, properties=('label', 'area', 'perimeter')))
        error_mask = (props['perimeter'] < perimeter_th) | (props['area'] < area_th)
        for i, r in props[error_mask].iterrows():
            tmap[labeled == r['label']] = 0
        trough[t] = tmap
    return trough


def get_rbf_matrix(shape, bandwidth=1):
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    gamma = np.log(2) / bandwidth ** 2
    basis = rbf_kernel(xy, xy, gamma)
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


def get_optimization_args(x, times, model_mlat, model_weight_max, rbf_bw, tv_hw, tv_vw, l2_weight, tv_weight,
                          prior_order, mlat_grid=None):
    if mlat_grid is None:
        mlat_grid = config.mlat_grid
    all_args = []
    # get rbf basis matrix
    basis = get_rbf_matrix(x.shape[1:], rbf_bw)
    # get tv matrix
    tv = get_tv_matrix(x.shape[1:], tv_hw, tv_vw) * tv_weight
    cos_ = np.cos(np.radians(mlat_grid)) + .5
    for i in range(times.shape[0]):
        # l2 norm cost away from model
        if prior_order == 1:
            l2 = abs(mlat_grid - model_mlat[i, :])
        elif prior_order == 2:
            l2 = (mlat_grid - model_mlat[i, :]) ** 2
        else:
            raise Exception("Invalid prior order")
        l2 -= l2.min()
        l2 = (model_weight_max - 1) * l2 / l2.max() + 1
        l2 *= l2_weight
        fin_mask = np.isfinite(x[i].ravel())
        args = (cp.Variable(x.shape[1] * x.shape[2]), basis[fin_mask, :], x[i].ravel()[fin_mask], tv, l2.ravel(),
                times[i], mlat_grid.shape)
        all_args.append(args)
    return all_args


def run_single(u, basis, x, tv, l2, t, output_shape):
    print(t)
    if x.size == 0:
        return np.zeros(output_shape)
    main_cost = u.T @ basis.T @ x
    tv_cost = cp.norm1(tv @ u)
    l2_cost = l2 @ (u ** 2)
    total_cost = main_cost + tv_cost + l2_cost
    prob = cp.Problem(cp.Minimize(total_cost))
    try:
        prob.solve(solver=cp.GUROBI)
    except Exception as e:
        print("FAILED, USING ECOS:", e)
        prob.solve(solver=cp.ECOS)
    return u.value.reshape(output_shape)


def run_multiple(args, parallel=True):
    if parallel:
        with multiprocessing.Pool(processes=4) as p:
            results = p.starmap(run_single, args)
    else:
        results = []
        for arg in args:
            results.append(run_single(*arg))
    return np.stack(results, axis=0)


def label_trough_one_year(year):
    one_h = np.timedelta64(1, 'h')
    params = config.trough_id_params

    days = np.arange(year, year + np.timedelta64(1, 'Y'), np.timedelta64(1, 'D'))
    trough_labels = []
    trough_times = []
    for day in days:
        start_time = day.astype('datetime64[D]').astype('datetime64[s]')
        end_time = start_time + np.timedelta64(1, 'D')
        tec_start = start_time - np.floor(params.bg_est_shape[0] / 2) * one_h
        tec_end = end_time + (np.floor(params.bg_est_shape[0] / 2)) * one_h

        tec, tec_times = trough_tec.get_tec_data(tec_start, tec_end)
        x, times = preprocess_interval(tec, tec_times, bg_est_shape=params.bg_est_shape)
        arb, _ = trough_data.get_arb_data(start_time, end_time)
        print("Setting up inversion optimization")
        ut = times.astype('datetime64[s]').astype(float)
        model_mlat = trough_model.get_model(ut, config.get_mlt_vals())

        args = get_optimization_args(x, times, model_mlat, params.model_weight_max, params.rbf_bw, params.tv_hw,
                                     params.tv_vw, params.l2_weight, params.tv_weight, params.prior_order)
        # run optimization
        print("Running inversion optimization")
        model_output = run_multiple(args)
        # threshold
        initial_trough = model_output >= params.threshold
        # postprocess
        print("Postprocessing inversion results")
        trough_labels.append(postprocess(initial_trough, params.perimeter_th, params.area_th, arb, params.closing_rad))
        trough_times.append(tec_times)
    trough_labels = np.concatenate(trough_labels, axis=0)
    trough_times = np.concatenate(trough_times, axis=0)

    output_fn = Path(config.processed_labels_dir) / f"labels_{year}.h5"
    trough_utils.write_h5(output_fn, labels=trough_labels, times=trough_times.astype(float))


def label_trough(start_date, end_date):
    for year in range(start_date.year, end_date.year + 1):
        label_trough_one_year(year)

