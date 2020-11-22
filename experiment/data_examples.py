import numpy as np
import matplotlib.pyplot as plt
import datetime
from teclab import config, utils
from ttools import data, plotting
plt.style.use('ggplot')

for i in range(4):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, subplot_kw=dict(projection='polar'))
    year, month, idx = utils.get_random_map_id(np.datetime64("2016-01-01T00:00:00"), np.datetime64("2020-06-01T00:00:00"))
    tecb, *_ = config.data(year - (month == 1), ((month - 1) % 12) + 1)
    tec, labels, timestamps = config.data(year, month)
    x = tec[:, :, idx]
    teca, *_ = config.data(year + (month == 12), ((month + 1) % 12) + 1)
    ut = timestamps[idx]
    idx += tecb.shape[-1]
    tec = np.concatenate((tecb, tec, teca), axis=-1)
    xp = data.preprocess(tec, idx)
    ax[0].pcolormesh(config.theta_grid - np.pi/360, config.radius_grid + .5, x, vmin=0, vmax=20, zorder=-1)
    ax[1].pcolormesh(config.theta_grid - np.pi / 360, config.radius_grid + .5, xp, vmin=-.5, vmax=.5, cmap='coolwarm', zorder=-1)
    for a in ax.flatten():
        plotting.format_polar_mag_ax(a)
        a.grid(True)
    fig.suptitle(datetime.datetime.fromtimestamp(ut))
    fig.savefig(f"C:\\Users\\Greg\\Documents\\trough meetings\\examples\\{year}_{month}_{idx}.png")
    plt.close(fig)