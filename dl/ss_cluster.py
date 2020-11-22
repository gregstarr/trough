import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

from teclab import config
from dl.model_def import get_unet


def create_dataset():
    # get images
    images = []
    for yr in range(2019, 2020):
        for mo in range(1, 13):
            try:
                tec, _, start_time = config.data(yr, mo)
                images.append(tec)
            except:
                pass
    X = np.concatenate(images, axis=-1)
    X = np.moveaxis(X, -1, 0)
    fin = np.isfinite(X)
    X = np.where(fin, X, 0)
    X = np.stack((X, fin), axis=-1)
    return X


X = create_dataset()
X = X[:1000]

k = 5
emb, model = get_unet(k)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS = 30
for epoch in range(EPOCHS):
    z = emb.predict(X)
    z = z.reshape((-1, 32))
    clusters = MiniBatchKMeans(k).fit_predict(z)
    clusters = clusters.reshape(X.shape[:-1])
    model.fit(X, clusters, epochs=2)

output = model.predict(X)
output = np.argmax(output, axis=-1)
for i in range(16):
    plt.figure()
    plt.subplot(121, projection='polar')
    plt.pcolormesh(config.theta_grid, config.radius_grid, X[5 * i, :, :, 0])
    plt.subplot(122, projection='polar')
    plt.pcolormesh(config.theta_grid, config.radius_grid, output[5 * i])
plt.show()
