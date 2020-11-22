import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ttools.io import get_borovsky_data
from teclab import config
from dl import model_def


def create_dataset(seed=12):
    # get indices
    bt, bdata = get_borovsky_data()
    # get images
    images = []
    ut = []
    for yr in range(2000, 2008):
        print(yr)
        for mo in range(1, 13):
            try:
                tec, _, start_time = config.data(yr, mo)
                images.append(np.moveaxis(tec, -1, 0))
                ut.append(start_time)
            except:
                pass
    ut = np.concatenate(ut, axis=0)
    tec_mask = np.in1d(ut, bt)
    bdata_mask = np.in1d(bt, ut)
    X = np.concatenate(images, axis=0)
    fin = np.isfinite(X)
    X = np.where(fin, X, np.random.randn(*X.shape))
    X = np.stack((X, fin), axis=-1)
    X = X[tec_mask]
    y = bdata[bdata_mask, 3:14]
    # y = bdata[bdata_mask, 1]
    X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=.5, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset()
    A = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 128
    training_data = (
        tf.data.Dataset.from_tensor_slices(
            (tf.constant(X_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.float32)))
            .shuffle(1024)
            .batch(BATCH_SIZE)
            .prefetch(A)
    )
    val_data = (
        tf.data.Dataset.from_tensor_slices((tf.constant(X_val, dtype=tf.float32), tf.constant(y_val, dtype=tf.float32)))
            .batch(BATCH_SIZE)
            .prefetch(A)
    )
    test_data = (
        tf.data.Dataset.from_tensor_slices(
            (tf.constant(X_test, dtype=tf.float32), tf.constant(y_test, dtype=tf.float32)))
            .batch(BATCH_SIZE)
            .prefetch(A)
    )
    plt.figure()
    for it in range(5):
        checkpoint_filepath = f'cp/checkpoint_resnet_{it}'
        emb, model = model_def.get_basic_resnet(11)
        model.compile(optimizer='adam', loss='mse')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                       save_weights_only=True,
                                                                       monitor='val_loss',
                                                                       mode='min',
                                                                       save_best_only=True)
        history = model.fit(training_data, epochs=30, validation_data=val_data,
                            callbacks=[model_checkpoint_callback], verbose=2)
        model.load_weights(checkpoint_filepath).expect_partial()
        print(model.evaluate(test_data))
        plt.plot(history.history['loss'], label=f'loss{it}')
        plt.plot(history.history['val_loss'], label=f'val_loss{it}')
    plt.show()
