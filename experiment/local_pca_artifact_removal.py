import numpy as np
import xarray as xr
import tensorflow as tf
import time
from scipy.sparse import linalg
import matplotlib.pyplot as plt


@tf.function
def covariance_sums2(batch, lat_size=7, lon_size=7, time_size=3):
    batch = tf.cast(batch, tf.float32)  # [batch_size, longitude, latitude, times]
    patches = tf.image.extract_patches(batch, (1, lat_size, lon_size, 1), (1, 1, 1, 1), (1, 1, 1, 1), "SAME")
    patches = tf.reshape(patches, (-1, lat_size * lon_size * time_size))
    finite_mask = tf.math.is_finite(patches)
    finite_mask_int = tf.cast(finite_mask, tf.float32)
    filled_patches = tf.where(finite_mask, patches, 0)
    sa = tf.reduce_sum(filled_patches, axis=0)
    san = tf.reduce_sum(finite_mask_int, axis=0)
    oa = tf.reduce_sum(filled_patches[:, :, None] * filled_patches[:, None, :], axis=0)
    oan = tf.reduce_sum(finite_mask_int[:, :, None] * finite_mask_int[:, None, :], axis=0)
    return sa, san, oa, oan


@tf.function
def covariance_sums(batch, lat_size=7, lon_size=7, time_size=3, processing_size=100):
    batch = tf.cast(batch, tf.float32)  # [batch_size, longitude, latitude, times]
    patches = tf.image.extract_patches(batch, (1, lat_size, lon_size, 1), (1, 3, 3, 1), (1, 1, 1, 1), "VALID")
    patches = tf.reshape(patches, (-1, lat_size * lon_size * time_size))
    patches = patches[tf.reduce_any(tf.math.is_finite(patches), axis=1)]
    sa = tf.zeros(patches.shape[1])
    san = tf.zeros(patches.shape[1])
    oa = tf.zeros((patches.shape[1], patches.shape[1]))
    oan = tf.zeros((patches.shape[1], patches.shape[1]))
    for i in range(int(np.ceil(patches.shape[1] / processing_size))):
        sla = slice(processing_size * i, min(processing_size * (i + 1), patches.shape[1]))
        inda = tf.constant(np.arange(processing_size * i, min(processing_size * (i + 1), patches.shape[1]))[:, None])
        finite_mask_a = tf.math.is_finite(patches[:, sla])
        finite_mask_int_a = tf.cast(finite_mask_a, tf.float32)
        filled_patches_a = tf.where(finite_mask_a, patches[:, sla], 0)
        sa = tf.tensor_scatter_nd_update(sa, inda, tf.reduce_sum(filled_patches_a, axis=0))
        san = tf.tensor_scatter_nd_update(san, inda, tf.reduce_sum(finite_mask_int_a, axis=0))
        for j in range(int(np.ceil(patches.shape[1] / processing_size))):
            slb = slice(processing_size * j, min(processing_size * (j + 1), patches.shape[1]))
            indb = tf.constant(
                np.arange(processing_size * j, min(processing_size * (j + 1), patches.shape[1]))[:, None])
            finite_mask_b = tf.math.is_finite(patches[:, slb])
            finite_mask_int_b = tf.cast(finite_mask_b, tf.float32)
            filled_patches_b = tf.where(finite_mask_b, patches[:, slb], 0)
            B, A = tf.meshgrid(indb[:, 0], inda[:, 0])
            ab = tf.stack((tf.reshape(A, (-1,)), tf.reshape(B, (-1,))), axis=1)
            oa = tf.tensor_scatter_nd_update(oa, ab, tf.reshape(
                tf.reduce_sum(filled_patches_a[:, :, None] * filled_patches_b[:, None, :], axis=0), (-1,)))
            oan = tf.tensor_scatter_nd_update(oan, ab, tf.reshape(
                tf.reduce_sum(finite_mask_int_a[:, :, None] * finite_mask_int_b[:, None, :], axis=0), (-1,)))
    return sa, san, oa, oan


@tf.function
def reconstruct(batch, pca_vectors, mean_patch, lat_size=7, lon_size=7, time_size=3, k=10):
    batch = tf.cast(batch, tf.float32)  # [batch_size, longitude, latitude, times]
    patches = tf.image.extract_patches(batch, (1, lat_size, lon_size, 1), (1, 1, 1, 1), (1, 1, 1, 1), "SAME")
    patches = tf.reshape(patches, (-1, lat_size * lon_size * time_size))
    centered_patches = tf.where(tf.math.is_finite(patches), patches - mean_patch, 0)
    z = tf.matmul(centered_patches, tf.transpose(
        tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(pca_vectors), pca_vectors)), tf.transpose(pca_vectors))))
    recon = tf.matmul(z, tf.transpose(pca_vectors)) + mean_patch
    rpatch_centers = tf.reshape(recon, batch.shape[:3] + (lat_size, lon_size, time_size))
    z = tf.reshape(z, batch.shape[:3] + (k,))
    return rpatch_centers, z


tec_paths = "E:\\tec_data\\data\\madrigal\\*.nc"
madrigal = xr.open_mfdataset(tec_paths, combine='by_coords')['tec']

mad_times = madrigal.time.values
start = np.argmax(mad_times >= np.datetime64("2018-01-15T00:00:00"))
end = np.argmax(mad_times >= np.datetime64("2019-01-15T00:00:00"))
average_time = 11
lat_size = 7
lon_size = 7
time_size = 3
k = 10
major_chunk = 1000
minor_chunk = 1
recalculate_covariance = True

if recalculate_covariance:
    print("Calculating covariance")
    n_items = end - start + 1
    item_size = lon_size * lat_size * time_size
    sum_acc = np.zeros(item_size)
    outer_acc = np.zeros((item_size, item_size))
    sum_acc_n = np.zeros(item_size)
    outer_acc_n = np.zeros((item_size, item_size))
    c_time = 0
    t0 = time.time()
    for i in range(int(np.ceil(n_items / major_chunk))):
        major_sl = slice(start + i * major_chunk - (average_time - 1) // 2,
                         min(start + (i + 1) * major_chunk + (average_time - 1) // 2, end + 1))
        major = madrigal[major_sl].load()
        smoothed = major.rolling(time=average_time, center=True).mean()
        major = major[(average_time - 1) // 2:-1 * (average_time - 1) // 2]
        res = major - smoothed
        res.load()
        resdata = res.values
        resdata = np.stack([resdata[i:resdata.shape[0] - time_size + i] for i in range(time_size)], axis=-1)

        for j in range(int(np.ceil(resdata.shape[0] / minor_chunk))):
            index_slice = slice(j * minor_chunk, (j + 1) * minor_chunk)
            data_batch = resdata[index_slice]
            sa, san, oa, oan = covariance_sums(data_batch.astype(float))
            sum_acc += sa
            sum_acc_n += san
            outer_acc += oa
            outer_acc_n += oan
        c_time = time.time() - t0
        c_item = i * major_chunk + j * minor_chunk
        t_sec = c_time * n_items / c_item
        print(f"{c_time / 3600:04.2f} / {t_sec / 3600:04.2f} hours finished")

    sum_acc = np.array(sum_acc)
    outer_acc = np.array(outer_acc)
    sum_acc_n = np.array(sum_acc_n)
    outer_acc_n = np.array(outer_acc_n)

    np.save("E:\\tec_data\\data\\dataset 1\\intermediate\\sum_acc.npy", sum_acc)
    np.save("E:\\tec_data\\data\\dataset 1\\intermediate\\sum_acc_n.npy", sum_acc_n)
    np.save("E:\\tec_data\\data\\dataset 1\\intermediate\\outer_acc.npy", outer_acc)
    np.save("E:\\tec_data\\data\\dataset 1\\intermediate\\outer_acc_n.npy", outer_acc_n)

sum_acc = np.load("E:\\tec_data\\data\\dataset 1\\intermediate\\sum_acc.npy")
sum_acc_n = np.load("E:\\tec_data\\data\\dataset 1\\intermediate\\sum_acc_n.npy")
outer_acc = np.load("E:\\tec_data\\data\\dataset 1\\intermediate\\outer_acc.npy")
outer_acc_n = np.load("E:\\tec_data\\data\\dataset 1\\intermediate\\outer_acc_n.npy")

print("Calculating eigenvectors")
mean = sum_acc / sum_acc_n
covariance = outer_acc / outer_acc_n - mean[:, None] * mean[None, :]
values, vectors = linalg.eigsh(covariance, k=k)
vectors = tf.constant(vectors)

print("Calculating reconstruction")
n_items = end - start + 1
item_size = lon_size * lat_size * time_size
sum_acc = np.zeros(item_size)
outer_acc = np.zeros((item_size, item_size))
sum_acc_n = np.zeros(item_size)
outer_acc_n = np.zeros((item_size, item_size))
c_time = 0
t0 = time.time()
for i in range(int(np.ceil(n_items / major_chunk))):
    major_sl = slice(start + i * major_chunk - (average_time - 1) // 2,
                     min(start + (i + 1) * major_chunk + (average_time - 1) // 2, end + 1))
    major = madrigal[major_sl].load()
    smoothed = major.rolling(time=average_time, center=True).mean()
    major = major[(average_time - 1) // 2:-1 * (average_time - 1) // 2]
    res = major - smoothed
    res.load()
    resdata = res.values
    resdata = np.stack([resdata[i:resdata.shape[0] - time_size + i] for i in range(time_size)], axis=-1)
    recon = np.empty_like(res.values)
    components = np.empty(res.shape[:3] + (k,), dtype=float)

    for j in range(int(np.ceil(resdata.shape[0] / minor_chunk))):
        index_slice = slice(j * minor_chunk, (j + 1) * minor_chunk)
        data_batch = resdata[index_slice]
        r, p = reconstruct(data_batch, vectors, mean)
        r = np.array(r)[..., (lat_size - 1) // 2, (lon_size - 1) // 2, (time_size - 1) // 2]
        recon[index_slice] = r
        components[index_slice] = p
    major.data = major.values - recon
    major.to_netcdf(f"E:\\tec_data\\data\\dataset 1\\major\\chunk_{i}.nc")
    c_time = time.time() - t0
    c_item = i * major_chunk + j * minor_chunk
    t_sec = c_time * n_items / c_item
    print(f"{c_time / 3600:04.2f} / {t_sec / 3600:04.2f} hours finished")
