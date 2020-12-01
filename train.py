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
        v = read_array(file, (n, 3))
        return x, p, v


def make_conv(filters, size):
    return tf.keras.layers.Conv2D(
        filters, size,
        data_format="channels_first", padding="same",
        use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResBlock, self).__init__()

        self.conv1 = make_conv(filters, 3)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.conv2 = make_conv(filters, 3)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)


    def call(self, input, training=False):
        res = self.conv1(input)
        res = self.bn1(res, training)
        res = tf.nn.relu(res)

        res = self.conv2(res)
        res = self.bn2(res, training)

        res = input + res
        res = tf.nn.relu(res)

        return res


class Model(tf.keras.models.Model):
    def __init__(self, blocks, filters):
        super(Model, self).__init__()

        self.conv1 = make_conv(filters, 3)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)

        self.tower = [ResBlock(filters) for _ in range(blocks)]

        self.flatten = tf.keras.layers.Flatten()

        self.policy_conv = make_conv(8, 1)
        self.policy_bn = tf.keras.layers.BatchNormalization(axis=1)
        self.policy_fc = tf.keras.layers.Dense(60)

        self.value_conv = make_conv(6, 1)
        self.value_bn = tf.keras.layers.BatchNormalization(axis=1)
        self.value_fc1 = tf.keras.layers.Dense(256)
        self.value_fc2 = tf.keras.layers.Dense(3)


    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = tf.nn.relu(x)

        for block in self.tower:
            x = block(x, training)

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

        return p, v


NET_FILE_VERSION = 3


def load_conv(file, layer):
    shape = struct.unpack('iiii', file.read(16))

    w = numpy.fromfile(file, dtype=numpy.float32, count=math.prod(shape))
    w = numpy.flip(w.reshape(shape[0], shape[1], shape[2] * shape[3]), 2).reshape(shape).transpose((2, 3, 1, 0))

    layer.kernel.assign(w)


def load_norm(file, norm):
    c, = struct.unpack('i', file.read(4))

    for w in norm.weights:
        w.assign(numpy.fromfile(file, dtype=numpy.float32, count=c))


def load_fc(file, layer):
    shape = struct.unpack('ii', file.read(8))

    w = numpy.fromfile(file, dtype=numpy.float32, count=math.prod(shape)).reshape(shape).transpose()
    b = numpy.fromfile(file, dtype=numpy.float32, count=shape[0])

    layer.weights[0].assign(w)
    layer.weights[1].assign(b)


def load_network(path):
    with open(path, "rb") as file:
        version, = struct.unpack('i', file.read(4))
        assert version == NET_FILE_VERSION

        filters, tower_size = struct.unpack('ii', file.read(8))
        model = Model(tower_size, filters)

        model.build((None, 2, 8, 8))

        load_conv(file, model.conv1)
        load_norm(file, model.bn1)

        for block in model.tower:
            load_conv(file, block.conv1)
            load_norm(file, block.bn1)
            load_conv(file, block.conv2)
            load_norm(file, block.bn2)

        policy_conv_count, policy_fc_count = struct.unpack('ii', file.read(8))
        assert policy_conv_count == 1
        assert policy_fc_count   == 1

        load_conv(file, model.policy_conv)
        load_norm(file, model.policy_bn)
        load_fc  (file, model.policy_fc)

        value_conv_count, value_fc_count = struct.unpack('ii', file.read(8))
        assert value_conv_count == 1
        assert value_fc_count   == 2

        load_conv(file, model.value_conv)
        load_norm(file, model.value_bn)
        load_fc  (file, model.value_fc1)
        load_fc  (file, model.value_fc2)

        return model


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
    with open(path, "wb") as file:
        file.write(struct.pack('i', NET_FILE_VERSION))

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
        save_network(self.prefix + str(epoch + 1), self.model)


def policy_loss(target, output):
    return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)


def value_loss(target, output):
    return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)


def main(model_path, save_path, data_path):
    model = load_network(model_path)
    opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=[policy_loss, value_loss], metrics=["accuracy"])

    x, p, v = load_data(data_path)

    model.fit(x, [p, v], epochs=10, batch_size=256, callbacks=[SaveCallback(save_path), tf.keras.callbacks.LearningRateScheduler(lambda e, r: r * 0.9)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
