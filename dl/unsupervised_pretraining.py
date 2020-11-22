"""
Make a CNN to predict Kp from the TEC images
"""
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ttools.data import get_gm_index_kyoto
from teclab import config
from dl import model_def


def create_dataset():
    # get kp index
    gm_index_df = get_gm_index_kyoto()
    kp = gm_index_df['kp'].values
    kp_ut = gm_index_df['ut'].values
    # get images
    images = []
    ut = []
    for yr in range(2019, 2021):
        for mo in range(1, 13):
            try:
                tec, _, start_time = config.data(yr, mo)
                images.append(tec)
                ut.append(start_time)
            except:
                pass
    X = np.concatenate(images, axis=-1)
    X = np.moveaxis(X, -1, 0)
    fin = np.isfinite(X)
    X = np.where(fin, X, 0)
    X = np.stack((X, fin), axis=-1)
    ut = np.concatenate(ut, axis=0)
    y = interp1d(kp_ut, kp, kind='previous')(ut)

    return train_test_split(X, y, test_size=.1)


X_train, X_test, y_train, y_test = create_dataset()

emb, model = model_def.get_basic_cnn(1)

model.compile(optimizer='adam', loss='mse')

RETRAIN = True
checkpoint_filepath = 'cp/checkpoint'
if RETRAIN:
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)
    history = model.fit(X_train, y_train, batch_size=256, epochs=10, validation_data=(X_test, y_test),
                        callbacks=[model_checkpoint_callback])
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('mse')
else:
    model.load_weights(checkpoint_filepath)

test_loss = model.evaluate(X_test, y_test, verbose=2)
print(test_loss)

plt.show()
