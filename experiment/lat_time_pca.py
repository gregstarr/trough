import numpy as np
import xarray as xr
import os
from scipy.sparse import linalg
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

from trough import plotting

plot_directory = "E:\\tec_data\\plots\\poleward transition zone"
dataset_path = "E:\\tec_data\\data\\dataset 1\\test.nc"
dataset = xr.open_dataset(dataset_path)
data = dataset['data']
times = dataset['time']

reshaped = data.stack(ex=('example', 'mlon'), x=('i', 'mlat')).values
finvals = np.isfinite(reshaped)
mean = np.nanmean(reshaped, axis=0)
filled = np.where(finvals, reshaped, 0)
cov_n = np.sum(finvals[:, :, None] * finvals[:, None, :], axis=0)
cov = np.sum(filled[:, :, None] * filled[:, None, :], axis=0) / cov_n
cov -= mean[:, None] * mean[None, :]
values, vectors = linalg.eigsh(cov, k=100)
reorder = np.argsort(abs(values))
values = values[reorder]
vectors = vectors[:, reorder]

white = np.zeros_like(vectors[0])
t = 100
for i in range(400):
    white += .2 * ((reshaped[t, finvals[t], None] - (vectors @ white[:, None])[finvals[t]]).T @ vectors[finvals[t]])[0]
recon = np.sum(vectors * white[None], axis=1)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].pcolormesh(reshaped[t].reshape((-1, 30)).T, vmin=0, vmax=12)
ax[1].pcolormesh(recon.reshape((-1, 30)).T, vmin=0, vmax=12)
plt.show()
