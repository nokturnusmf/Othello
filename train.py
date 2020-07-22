import sys
import struct
import math
import numpy
import tensorflow as tf


class Move:
    def __init__(self, row, col, p):
        self.row = row
        self.col = col
        self.p = p


class Position:
    def __init__(self, black, white, col):
        self.ours, self.theirs = (black, white) if col == 0 else (white, black)
        self.col = col
        self.moves = []

    def make_input(self):
        b1 = []
        b2 = []
        for i in range(8):
            b1_ = []
            b2_ = []
            for j in range(8):
                b = 1 << (i * 8 + j)
                b1_.append(1. if self.ours   & b else 0.)
                b2_.append(1. if self.theirs & b else 0.)
            b1.append(b1_)
            b2.append(b2_)
        return [b1, b2]

    def make_output(self):
        out = 64 * [0.]
        # out = 64 * [-1.]
        for m in self.moves:
            out[m.row * 8 + m.col] = m.p
        return out


class Game:
    def __init__(self, score):
        self.result = float((score > 0) - (score < 0))
        self.positions = []

    def make_input(self, index):
        return self.positions[index].make_input()

    def make_output(self, index):
        pos = self.positions[index]
        return pos.make_output(), self.result if pos.col == 0 else -self.result


def split_tuples(x):
    return [list(y) for y in zip(*x)]


def sample_batch(games, batch_size):
    def random_pos(game):
        i = numpy.random.randint(len(game.positions))
        return game.make_input(i), game.make_output(i)

    return split_tuples(random_pos(game) for game in numpy.random.choice(games, size=batch_size))


def load_data(path):
    result = []

    with open(path, 'rb') as file:
        game_count, = struct.unpack('N', file.read(8))

        for i in range(game_count):
            print(f"\rLoading games from {path}... {i+1}/{game_count}", end='')

            score, pos_count = struct.unpack('ii', file.read(8))

            game = Game(score)
            for _ in range(pos_count):
                black, white, col, move_count = struct.unpack('NNii', file.read(24))

                pos = Position(black, white, col)
                for _ in range(move_count):
                    row, col, p = struct.unpack('iif', file.read(12))
                    pos.moves.append(Move(row, col, p))

                game.positions.append(pos)

            result.append(game)

    print()
    return result


# class ResBlock(tf.keras.layers.Layer):
#     def __init__(self, filters, size):
#         super(ResBlock, self).__init__()
#
#         self.conv1 = tf.keras.layers.Conv2D(filters, size, padding='same')
#         self.conv2 = tf.keras.layers.Conv2D(filters, size, padding='same')
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU(negative_slope=0.01)
#
#     def build(self, input_shape):
#         self.conv1.build(input_shape)
#         self.conv2.build(self.conv1.output_shape)
#
#     def call(self, input):
#         res = self.conv1(input)
#         res = self.bn(res)
#         res = self.relu(res)
#
#         res = self.conv2(res)
#         res = self.bn(res)
#
#         res = input + res
#         res = self.relu(res)
#
#         return res


class Model(tf.keras.models.Model):
    def __init__(self, blocks, filters, size):
        super(Model, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters, size, data_format="channels_first", padding='same')

        self.tower = [tf.keras.layers.Conv2D(filters, size, data_format="channels_first", padding='same') for _ in range(blocks)]

        self.bn = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()

        self.policy_conv = tf.keras.layers.Conv2D(2, 1, data_format="channels_first")
        self.policy_bn = tf.keras.layers.BatchNormalization()
        self.policy_fc = tf.keras.layers.Dense(64)

        self.value_conv = tf.keras.layers.Conv2D(1, 1, data_format="channels_first")
        self.value_bn = tf.keras.layers.BatchNormalization()
        self.value_fc1 = tf.keras.layers.Dense(256)
        self.value_fc2 = tf.keras.layers.Dense(1)


    def call(self, x):
        x = self.conv1(x)
        # x = self.bn(x)
        x = tf.nn.relu(x)

        for conv in self.tower:
            x = conv(x)
            # x = self.bn(x)
            x = tf.nn.relu(x)


        p = self.policy_conv(x)
        # p = self.policy_bn(p)
        p = tf.nn.relu(p)
        p = self.flatten(p)
        p = self.policy_fc(p)

        v = self.value_conv(x)
        # v = self.value_bn(v)
        v = tf.nn.relu(v)
        v = self.flatten(v)
        v = self.value_fc1(v)
        v = tf.nn.relu(v)
        v = self.value_fc2(v)
        v = tf.math.tanh(v)

        return p, v


def load_network(path):
    pass


def save_network(path, model):
    def save_layer(file, layer):
        w = layer.kernel.numpy().transpose()
        b = layer.bias.numpy()
        shape = w.shape

        if len(shape) == 4:
            w = numpy.flip(w.transpose((0, 1, 3, 2)).reshape(shape[0], shape[1], shape[2] * shape[3]), 2)

        for d in shape:
            file.write(struct.pack('i', d))

        file.write(w.tobytes())
        file.write(b.tobytes())

    version = struct.pack('i', 1)

    with open(path, "wb") as file:
        file.write(version)

        filters, = model.conv1.bias.shape
        file.write(struct.pack('ii', filters, len(model.tower)))

        save_layer(file, model.conv1)

        for conv in model.tower:
            save_layer(file, conv)

        file.write(struct.pack('ii', 1, 1))
        save_layer(file, model.policy_conv)
        save_layer(file, model.policy_fc)

        file.write(struct.pack('ii', 1, 2))
        save_layer(file, model.value_conv)
        save_layer(file, model.value_fc1)
        save_layer(file, model.value_fc2)


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


def main(model_path, save_path, files):
    games = sum([load_data(path) for path in files], [])

    model = Model(8, 64, 3)
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=[policy_loss, value_loss])

    for _ in range(len(games) // 800):
        x, y = sample_batch(games, 4096)
        p, v = split_tuples(y)
        model.fit(numpy.asarray(x), [numpy.asarray(p), numpy.asarray(v)], batch_size=32)

    save_network(save_path, model)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
