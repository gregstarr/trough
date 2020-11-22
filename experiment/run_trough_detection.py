import numpy as np
import pandas
import time
import os
import h5py
from sklearn.metrics import confusion_matrix
from skimage.measure import label

from ttools import rbf_inversion
from teclab import config


save_dir = "E:"
troughs = []
year = []
month = []
index = []

model = rbf_inversion.RbfIversion(config.mlt_grid, config.mlat_grid, ds=(1, 4))

for y in range(2010, 2021):
    for m in config.map_tree[y]:
        for i in config.map_tree[y][m]['index']:
            try:
                x, ut, tec = model.load_and_preprocess(y, m, i)
                tec_trough_model = model.run(x, ut)
                tec_trough_model = model.postprocess(tec_trough_model)
                tec_trough = model.decision(tec_trough_model)
                tec_trough = model.postprocess_labels(tec_trough)
                year.append(y)
                month.append(m)
                index.append(i)
                print(y, m, i)
            except Exception as e:
                print(e.__class__)
                print(e)

with h5py.File(os.path.join(save_dir, 'labels.h5'), 'w') as f:
    f.create_dataset('labels', data=np.array(troughs))
    f.create_dataset('year', data=np.array(year))
    f.create_dataset('month', data=np.array(month))
    f.create_dataset('index', data=np.array(index))
