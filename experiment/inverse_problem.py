import numpy as np
import matplotlib.pyplot as plt
from ttools import utils, plotting, rbf_inversion
from teclab import config

model = rbf_inversion.RbfIversion(config.mlt_grid[:, ::2], config.mlat_grid[:, ::2])

year, month, index = utils.get_random_map_id()
ut, tec = rbf_inversion.get_tec_map_interval(year, month, index)
ut = ut[ut.shape[0]//2]
x = rbf_inversion.preprocess_tec_interval(tec)[:, ::2]
u = model.run(x, ut).reshape(x.shape)

fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 6), subplot_kw=dict(projection='polar'))
ax[0].pcolormesh(config.theta_grid[:, ::2] - np.pi/360, config.radius_grid[:, ::2] + .5, x, cmap='coolwarm', vmin=-.5, vmax=.5)
ax[1].pcolormesh(config.theta_grid[:, ::2] - np.pi/360, config.radius_grid[:, ::2] + .5, u)
for a in ax:
    plotting.format_polar_mag_ax(a)
plt.show()
