"""
perform PCA on the madrigal TEC data:
    - should be done in MLT because features will be stationary
    - only consider north america region to avoid averaging over a big longitude range
    - computation of covariance should be batched and run with tensorflow on GPU
        - running sum, running dot products, evaluate mean and covariance at the end
"""
import numpy as np
import xarray as xr
import tensorflow as tf
import time
from scipy.sparse import linalg
import matplotlib.pyplot as plt


@tf.function
def covariance_sums(batch):
    """Use tensorflow to get the sum and outer product of the input data as well as the number of non-null data
    for each.
    """
    batch = tf.cast(batch, tf.float32)
    finite_mask = tf.math.is_finite(batch)
    finite_mask_int = tf.cast(finite_mask, tf.float32)
    filled_batch = tf.where(finite_mask, batch, 0)
    sa = tf.reduce_sum(filled_batch, axis=0)
    san = tf.reduce_sum(finite_mask_int, axis=0)
    oa = tf.reduce_sum(filled_batch[:, :, None] * filled_batch[:, None, :], axis=0)
    oan = tf.reduce_sum(finite_mask_int[:, :, None] * finite_mask_int[:, None, :], axis=0)
    return sa, san, oa, oan


print("Loading data")
madrigal_paths = "E:\\tec_data\\data\\madrigal\\mag\\*.nc"
madrigal = xr.open_mfdataset(madrigal_paths, combine='by_coords', parallel=True)['tec']
mlt, mlat = np.meshgrid(madrigal.mlt.values, madrigal.mlat.values)

recalculate = True
if recalculate:
    batch_size = 4
    n_items = madrigal.time.size
    batches = int(np.ceil(n_items / batch_size))
    item_size = madrigal.mlt.size * madrigal.mlat.size
    sum_acc = np.zeros(item_size)
    outer_acc = np.zeros((item_size, item_size))
    sum_acc_n = np.zeros(item_size)
    outer_acc_n = np.zeros((item_size, item_size))

    print(f"Processing {n_items} items in {batches} batches")
    total_time = 0
    for i in range(batches):
        t0 = time.time()
        index_slice = slice(i*batch_size, (i+1)*batch_size)
        data_batch = madrigal.isel(time=index_slice).values.reshape((-1, item_size))
        data_batch[data_batch > 200] = np.nan
        sa, san, oa, oan = covariance_sums(data_batch.astype(float))
        sum_acc += sa
        sum_acc_n += san
        outer_acc += oa
        outer_acc_n += oan

        total_time += time.time() - t0
        t_sec = total_time * n_items / ((i + 1) * batch_size)
        if not i % 100:
            print(f"{total_time / 3600} / {t_sec / 3600} hours finished")
            # mean = np.array(sum_acc / sum_acc_n)
            # covariance = np.array(outer_acc / outer_acc_n - mean[:, None] * mean[None, :])
            # fig, ax = plt.subplots(1, 2, figsize=(20, 12))
            # cm = ax[0].pcolormesh(mlt, mlat, mean.reshape((madrigal.mlat.size, madrigal.mlt.size)))
            # plt.colorbar(cm, ax=ax[0])
            # ax[0].set_title("Mean")
            # cm = ax[1].pcolormesh(mlt, mlat, covariance.diagonal().reshape((madrigal.mlat.size, madrigal.mlt.size)))
            # plt.colorbar(cm, ax=ax[1])
            # ax[1].set_title("Variance")
            # plt.savefig(f"E:\\tec_data\\data\\madrigal\\pca_experiment\\{i}.png", fig=fig)
            # plt.close(fig)

    sum_acc = np.array(sum_acc)
    outer_acc = np.array(outer_acc)
    sum_acc_n = np.array(sum_acc_n)
    outer_acc_n = np.array(outer_acc_n)

    np.save("E:\\tec_data\\data\\madrigal\\pca_experiment\\sum.npy", sum_acc)
    np.save("E:\\tec_data\\data\\madrigal\\pca_experiment\\sum_n.npy", sum_acc_n)
    np.save("E:\\tec_data\\data\\madrigal\\pca_experiment\\outer.npy", outer_acc)
    np.save("E:\\tec_data\\data\\madrigal\\pca_experiment\\outer_n.npy", outer_acc_n)

sum_acc = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\sum.npy")
sum_acc_n = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\sum_n.npy")
outer_acc = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\outer.npy")
outer_acc_n = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\outer_n.npy")

mean = sum_acc / sum_acc_n
covariance = outer_acc / outer_acc_n - mean[:, None] * mean[None, :]
covariance[np.isnan(covariance)] = np.nanmean(covariance)

values, vectors = linalg.eigsh(covariance, k=16)
fig, ax = plt.subplots(4, 4, figsize=(20, 12))
for i, a in enumerate(ax.flatten()):
    a.contourf(mlt, mlat, vectors[:, i].reshape(mlt.shape))
    a.set_title(values[i])
plt.tight_layout()


# fig, ax = plt.subplots(1, 2)
# cm = ax[0].pcolormesh(mean.reshape((madrigal.mlat.size, madrigal.mlt.size)))
# plt.colorbar(cm, ax=ax[0])
# cm = ax[1].pcolormesh(covariance.diagonal().reshape((madrigal.mlat.size, madrigal.mlt.size)))
# plt.colorbar(cm, ax=ax[1])
#
# fig, ax = plt.subplots(1, 2)
# cm = ax[0].pcolormesh(sum_acc_n.reshape((madrigal.mlat.size, madrigal.mlt.size)))
# plt.colorbar(cm, ax=ax[0])
# cm = ax[1].pcolormesh(outer_acc_n.diagonal().reshape((madrigal.mlat.size, madrigal.mlt.size)))
# plt.colorbar(cm, ax=ax[1])
plt.show()
