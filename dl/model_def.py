import tensorflow as tf


def residual_block(x, filters, kernel_size=3, stride=1, first=False, last=False):
    """downsampling and filter increase in same layer
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if first:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if stride != 1 or stride != (1, 1):
        shortcut = tf.keras.layers.Conv2D(filters, 1, stride)(x)  # valid conv
    else:
        shortcut = x

    if not first:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = WrapPad((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)(x)  # pad
    x = tf.keras.layers.Conv2D(filters, kernel_size, stride)(x)  # valid conv

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = WrapPad((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)(x)  # pad
    x = tf.keras.layers.Conv2D(filters, kernel_size)(x)  # valid conv

    x = tf.keras.layers.Add()([shortcut, x])
    if last:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    return x


class WrapPad(tf.keras.layers.Layer):
    def __init__(self, v_pad, h_pad):
        super().__init__()
        self.v_pad = v_pad
        self.h_pad = h_pad

    def call(self, input):
        # batch x height x width x channels
        in_shape = tf.shape(input)
        # bottom and top
        bottom_repeat = input[:, 0, None] * tf.ones((1, self.v_pad, 1, 1))
        top_wrap = tf.roll(input[:, -1:-self.v_pad - 1:-1], in_shape[2]//2, axis=2)
        x = tf.concat([bottom_repeat, input, top_wrap], axis=1)
        # wrap sides
        x = tf.concat([x[:, :, -1:-self.h_pad - 1:-1], x, x[:, :, self.h_pad - 1::-1]], axis=2)
        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), upsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.upsample = upsample
        self.pad = WrapPad((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        if self.upsample:
            in_shape = tf.shape(inputs)
            x = tf.image.resize(inputs, in_shape[1:3] * self.upsample)
            x = self.pad(x)
        else:
            x = self.pad(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x


def get_unet(k=10):
    inp = tf.keras.layers.Input(shape=(60, 360, 2))
    x1 = ConvBlock(32, (5, 5))(inp)
    x2 = ConvBlock(32, (5, 5), 2)(x1)
    x2 = ConvBlock(64, (3, 3))(x2)
    x3 = ConvBlock(64, (3, 3), 2)(x2)
    x3 = ConvBlock(128, (3, 3))(x3)
    x4 = ConvBlock(128, (3, 3), 3)(x3)
    x4 = ConvBlock(256, (3, 3))(x4)
    x = ConvBlock(128, (3, 3))(x4)
    x = ConvBlock(128, (3, 3), upsample=3)(x)
    x = tf.keras.layers.Concatenate()([x, x3])
    x = ConvBlock(64, (3, 3))(x)
    x = ConvBlock(64, (3, 3), upsample=2)(x)
    x = tf.keras.layers.Concatenate()([x, x2])
    x = ConvBlock(32, (3, 3))(x)
    x = ConvBlock(32, (5, 5), upsample=2)(x)
    x = tf.keras.layers.Concatenate()([x, x1])
    x = ConvBlock(32, (5, 5))(x)
    output = tf.keras.layers.Conv2D(k, 1)(x)
    return tf.keras.Model(inputs=inp, outputs=x), tf.keras.Model(inputs=inp, outputs=output)


def get_basic_cnn(k=1):
    inp = tf.keras.layers.Input(shape=(60, 360, 2))
    x = tf.keras.layers.AveragePooling2D((1, 2), strides=(1, 2))(inp)  # 60, 180
    x = tf.keras.layers.Conv2D(32, 3)(x)
    x = ConvBlock(32, 3)(x)
    x = ConvBlock(32, 3)(x)
    x = ConvBlock(64, 3, strides=2)(x)  # 30, 90
    x = ConvBlock(64, 3, strides=2)(x)  # 15, 45
    x = ConvBlock(128, 3, strides=3)(x)  # 5, 15
    embedding = tf.keras.layers.GlobalMaxPooling2D()(x)
    output = tf.keras.layers.Dense(k)(embedding)
    embedder = tf.keras.Model(inputs=inp, outputs=embedding)
    regressor = tf.keras.Model(inputs=inp, outputs=output)
    return embedder, regressor


def get_basic_resnet(k=1):
    inp = tf.keras.layers.Input(shape=(60, 360, 2))
    x = WrapPad(2, 2)(inp)
    x = tf.keras.layers.Conv2D(32, 5)(x)
    x = residual_block(x, 32, first=True)
    x = residual_block(x, 64, stride=2)  # 60, 180
    x = residual_block(x, 64, stride=2)  # 30, 90
    x = residual_block(x, 64, stride=2)  # 15, 45
    x = residual_block(x, 128, stride=3, last=True)  # 5, 15
    embedding = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(k)(embedding)
    embedder = tf.keras.Model(inputs=inp, outputs=embedding)
    regressor = tf.keras.Model(inputs=inp, outputs=output)
    return embedder, regressor


def change_model_output(embedder, k=11):
    x = embedder.input
    x = embedder(x)
    output = tf.keras.layers.Dense(k)(x)
    model = tf.keras.Model(inputs=embedder.input, outputs=output)
    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from teclab import config

    X, _, _ = config.data(2019, 4)
    X = np.moveaxis(X, -1, 0)
    fin = np.isfinite(X)
    X = np.where(fin, X, 0)
    X = np.stack((X, fin), axis=-1)
