import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import linregress

from teclab import config
from dl.supervised_with_indices import create_dataset
from dl import model_def

X_train, X_val, X_test, y_train, y_val, y_test = create_dataset()
BATCH_SIZE = 64
test_data = (
    tf.data.Dataset.from_tensor_slices((tf.constant(X_test, dtype=tf.float32), tf.constant(y_test, dtype=tf.float32)))
    .batch(BATCH_SIZE)
)
emb, model = model_def.get_basic_resnet(11)
model.compile(optimizer='adam', loss='mse')
checkpoint_filepath = f'cp/checkpoint_resnet_0'
model.load_weights(checkpoint_filepath)

output = model.predict(X_test)
for i in range(11):
    slope, intercept, r, p, stderr = linregress(output[:, i], y_test[:, i])
    print(i, r)
    plt.figure()
    plt.plot(output[:, i], y_test[:, i], '.')
plt.show()