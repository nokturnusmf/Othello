import sys
import struct
import math
import numpy
import tensorflow as tf


def load_data(path):
    def read_array(file, shape):
        return numpy.fromfile(file, dtype=numpy.float32, count=math.prod(shape)).reshape(shape)

    with open(path, 'rb') as file:
        n, = struct.unpack('N', file.read(8))
        x = read_array(file, (n, 2, 8, 8))
        p = read_array(file, (n, 60))
        v = read_array(file, (n, 1))
        return x, p, v


def make_conv(filters, size):
    return tf.keras.layers.Conv2D(filters, size, data_format="channels_first", padding="same", use_bias=False)


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, size):
        super(ResBlock, self).__init__()

        self.conv1 = make_conv(filters, size)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.conv2 = make_conv(filters, size)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)


    def call(self, input):
        res = self.conv1(input)
        res = self.bn1(res)
        res = tf.nn.relu(res)

        res = self.conv2(res)
        res = self.bn2(res)

        res = input + res
        res = tf.nn.relu(res)

        return res


class Model(tf.keras.models.Model):
    def __init__(self, blocks, filters, size):
        super(Model, self).__init__()

        self.conv1 = make_conv(filters, 5)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.tower = [ResBlock(filters, size) for _ in range(blocks)]

        self.flatten = tf.keras.layers.Flatten()

        self.policy_conv = make_conv(2, 1)
        self.policy_bn = tf.keras.layers.BatchNormalization(axis=1)
        self.policy_fc = tf.keras.layers.Dense(60)

        self.value_conv = make_conv(1, 1)
        self.value_bn = tf.keras.layers.BatchNormalization(axis=1)
        self.value_fc1 = tf.keras.layers.Dense(256)
        self.value_fc2 = tf.keras.layers.Dense(1)


    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = tf.nn.relu(x)

        for block in self.tower:
            x = block(x)

        p = self.policy_conv(x)
        p = self.policy_bn(p, training)
        p = tf.nn.relu(p)
        p = self.flatten(p)
        p = self.policy_fc(p)

        v = self.value_conv(x)
        v = self.value_bn(v, training)
        v = tf.nn.relu(v)
        v = self.flatten(v)
        v = self.value_fc1(v)
        v = tf.nn.relu(v)
        v = self.value_fc2(v)
        v = tf.math.tanh(v)

        return p, v


def load_network(path):
    pass


def save_conv(file, layer):
    w = layer.kernel.numpy().transpose()
    shape = w.shape
    w = numpy.flip(w.transpose((0, 1, 3, 2)).reshape(shape[0], shape[1], shape[2] * shape[3]), 2)

    for d in shape:
        file.write(struct.pack('i', d))

    file.write(w.tobytes())


def save_norm(file, norm):
    ws = [w.numpy() for w in norm.weights]
    c, = ws[0].shape

    file.write(struct.pack('i', c))
    for w in ws:
        file.write(w.tobytes())


def save_fc(file, layer):
    w = layer.kernel.numpy().transpose()
    b = layer.bias.numpy()
    shape = w.shape

    for d in shape:
        file.write(struct.pack('i', d))

    file.write(w.tobytes())
    file.write(b.tobytes())


def save_network(path, model):
    version = struct.pack('i', 2)

    with open(path, "wb") as file:
        file.write(version)

        filters = model.conv1.kernel.shape[3]
        file.write(struct.pack('ii', filters, len(model.tower)))

        save_conv(file, model.conv1)
        save_norm(file, model.bn1)

        for block in model.tower:
            save_conv(file, block.conv1)
            save_norm(file, block.bn1)
            save_conv(file, block.conv2)
            save_norm(file, block.bn2)

        file.write(struct.pack('ii', 1, 1))
        save_conv(file, model.policy_conv)
        save_norm(file, model.policy_bn)
        save_fc  (file, model.policy_fc)

        file.write(struct.pack('ii', 1, 2))
        save_conv(file, model.value_conv)
        save_norm(file, model.value_bn)
        save_fc  (file, model.value_fc1)
        save_fc  (file, model.value_fc2)


class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, prefix):
        self.prefix = prefix


    def on_epoch_end(self, epoch, logs):
        save_network(self.prefix + '_' + str(epoch + 1), self.model)


def policy_loss(target, output):
    return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)


# def policy_loss(target, output):
#     legal = tf.greater_equal(target, 0)
#
#     target_ = tf.where(legal, target, 0.)
#     output_ = tf.where(legal, output, -1e9)
#
#     return tf.nn.softmax_cross_entropy_with_logits(labels=target_, logits=output_)


def value_loss(target, output):
    return tf.losses.mean_squared_error(target, output)


def main(model_path, save_path, data_path):
    model = Model(8, 64, 3)
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss=[policy_loss, value_loss])

    x, p, v = load_data(data_path)

    model.fit(x, [p, v], epochs=4, batch_size=32, callbacks=[SaveCallback(save_path)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
