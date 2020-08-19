"""
- recon contour on top of madrigal
- component contours on top of madrigal
"""
import numpy as np
import matplotlib.pyplot as plt

recon = np.load("E:\\tec_data\\artifacts\\reconstruction.npy")
comp = np.load("E:\\tec_data\\artifacts\\components.npy")
x = np.load("E:\\tec_data\\artifacts\\x.npy")
y = np.load("E:\\tec_data\\artifacts\\y.npy")
fixed = np.load("E:\\tec_data\\artifacts\\fixed.npy")
madrigal = np.load("E:\\tec_data\\artifacts\\madrigal.npy")

for t in range(recon.shape[0]):
    print(t)
    fig, ax = plt.subplots(1, 3, figsize=(24, 12), sharex=True, sharey=True)
    ax[0].pcolormesh(x, y, madrigal[t].T, vmin=0, vmax=10)
    ax[1].pcolormesh(x, y, recon[t].T, vmin=-2, vmax=2, cmap='coolwarm')
    ax[2].pcolormesh(x, y, fixed[t].T, vmin=0, vmax=10)
    ax[0].set_ylabel("GLAT")
    ax[0].set_xlabel("GLON")
    plt.tight_layout()
    plt.savefig(f"E:\\plots\\artifact reconstruction\\{t}.png", fig=fig)
    plt.close(fig)
