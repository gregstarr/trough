import numpy as np
import xarray as xr
import time
import matplotlib.pyplot as plt


print("Loading data")
madrigal_paths = "E:\\tec_data\\data\\madrigal\\mag\\*.nc"
madrigal = xr.open_mfdataset(madrigal_paths, combine='by_coords', parallel=True)['tec']
mlt, mlat = np.meshgrid(madrigal.mlt.values, madrigal.mlat.values)
item_size = madrigal.mlt.size * madrigal.mlat.size
n_items = madrigal.time.size
madrigal = madrigal.stack(z=('mlat', 'mlt'))

sum_acc = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\sum.npy")
sum_acc_n = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\sum_n.npy")
outer_acc = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\outer.npy")
outer_acc_n = np.load("E:\\tec_data\\data\\madrigal\\pca_experiment\\outer_n.npy")
mean = sum_acc / sum_acc_n
covariance = outer_acc / outer_acc_n - mean[:, None] * mean[None, :]
std = np.sqrt(covariance.diagonal())

# model
batch_size = 100
n_batches = int(np.ceil(n_items / batch_size))
k = 10
l = .02
step_size = .001
A = np.random.randn(n_items, k)
B = np.random.randn(item_size, k)
epochs = 5
total_time = 0
for e in range(epochs):
    for i in range(n_batches):
        t0 = time.time()
        index_slice = slice(i * batch_size, (i+1) * batch_size)
        data_batch = madrigal.isel(time=index_slice).values
        data_batch -= mean[None, :]
        data_batch /= std[None, :]
        ij = np.argwhere(np.isfinite(data_batch))
        data_batch = data_batch[ij[:, 0], ij[:, 1]]
        recon = np.sum(A[ij[:, 0]] * B[ij[:, 1]], axis=1)
        if (i // 10) % 2:
            A[ij[:, 0]] -= step_size * (-2 * (data_batch - recon)[:, None] * B[ij[:, 1]] + 2 * l * A[ij[:, 0]])
        else:
            B[ij[:, 1]] -= step_size * (-2 * (data_batch - recon)[:, None] * A[ij[:, 0]] + 2 * l * B[ij[:, 1]])
        total_time += time.time() - t0
        t_sec = total_time * n_items * epochs / ((i + 1) * batch_size + e * n_batches * batch_size)
        if not i % 10:
            print(np.sum((data_batch - recon)**2))
            print(f"{total_time / 3600} / {t_sec / 3600} hours finished")

np.save("E:\\tec_data\\data\\madrigal\\pca_experiment\\A.npy", A)
np.save("E:\\tec_data\\data\\madrigal\\pca_experiment\\B.npy", B)

for i in range(k):
    plt.figure()
    plt.pcolormesh(B[:, i].reshape(mlt.shape))
plt.show()
