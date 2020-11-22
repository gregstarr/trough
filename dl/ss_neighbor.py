import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from teclab import config
from dl.model_def import get_unet


def create_dataset():
    # get images
    images = []
    for yr in range(2010, 2020):
        print(yr)
        for mo in range(1, 13):
            try:
                tec, _, start_time = config.data(yr, mo)
                images.append(tec)
            except:
                pass
    X = np.concatenate(images, axis=-1)
    X = np.moveaxis(X, -1, 0)
    fin = np.isfinite(X)
    X = np.where(fin, X, np.random.randn(*X.shape)*10)
    X = np.stack((X, fin), axis=-1)
    y = X[:-1, :, :, 0]
    X = X[1:]
    return train_test_split(X, y, test_size=.1)


def augment(image, label):
    scale = tf.random.uniform([1], .5, 2)
    bias = tf.random.uniform([1], 0, 5)
    roll = tf.random.uniform([1], 0, 360, dtype=tf.int32)[0]
    image = image * scale
    image = image + bias
    image = tf.roll(image, roll, 1)
    label = tf.roll(label, roll, 1)
    return image, label


X_train, X_test, y_train, y_test = create_dataset()
A = tf.data.experimental.AUTOTUNE
training_data = (
    tf.data.Dataset.from_tensor_slices((tf.constant(X_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.float32)))
    .shuffle(256)
    .map(augment, num_parallel_calls=A)
    .batch(64)
    .prefetch(A)
)
test_data = (
    tf.data.Dataset.from_tensor_slices((tf.constant(X_test, dtype=tf.float32), tf.constant(y_test, dtype=tf.float32)))
    .map(augment, num_parallel_calls=A)
    .batch(64)
    .prefetch(A)
)

emb, model = get_unet(1)
model.compile(optimizer='adam', loss='mse')

RETRAIN = False
checkpoint_filepath = 'cp/checkpoint'
if RETRAIN:
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)
    history = model.fit(training_data, epochs=100, validation_data=test_data,
                        callbacks=[model_checkpoint_callback])
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('mse')
else:
    model.load_weights(checkpoint_filepath)

X_test = X_test[-100:]
y_test = y_test[-100:]
output = model.predict(X_test)

for i in range(16):
    plt.figure(figsize=(20, 12))
    plt.subplot(131, projection='polar')
    plt.pcolormesh(config.theta_grid, config.radius_grid, X_test[i, :, :, 0], vmin=0, vmax=15)
    plt.subplot(132, projection='polar')
    plt.pcolormesh(config.theta_grid, config.radius_grid, y_test[i, :, :], vmin=0, vmax=15)
    plt.subplot(133, projection='polar')
    plt.pcolormesh(config.theta_grid, config.radius_grid, output[i, :, :, 0], vmin=0, vmax=15)


# l = 55
# for i in range(16):
#     plt.figure()
#     plt.subplot(121, projection='polar')
#     plt.pcolormesh(config.theta_grid, config.radius_grid, output[l, :, :, 2*i])
#     plt.subplot(122, projection='polar')
#     plt.pcolormesh(config.theta_grid, config.radius_grid, output[l, :, :, 2*i+1])
#
# plt.figure()
# plt.subplot(111, projection='polar')
# plt.pcolormesh(config.theta_grid, config.radius_grid, X_test[l, :, :, 0])

plt.show()
