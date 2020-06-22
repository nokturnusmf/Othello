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
        for i in range(64):
            b = 1 << i
            b1.append(1. if self.ours   & b else 0.)
            b2.append(1. if self.theirs & b else 0.)
        return b1 + b2

    # def make_input(self):
    #     b1 = []
    #     b2 = []
    #     for i in range(8):
    #         b1_ = []
    #         b2_ = []
    #         for j in range(8):
    #             b = 1 << (i * 8 + j)
    #             b1_.append(1. if self.ours   & b else 0.)
    #             b2_.append(1. if self.theirs & b else 0.)
    #         b1.append(b1_)
    #         b2.append(b2_)
    #     return [b1, b2]

    def make_output(self):
        out = 64 * [-1.]
        for m in self.moves:
            out[m.row * 8 + m.col] = m.p
        return out


class Game:
    def __init__(self, score):
        self.result = (math.copysign(1, score) + 1) * 0.5
        self.positions = []

    def make_input(self, index):
        return self.positions[index].make_input()

    def make_output(self, index):
        return self.positions[index].make_output() + [self.result if self.positions[index].col == 0 else 1 - self.result]


def sample_batch(games, batch_size):
    def random_pos(game):
        i = numpy.random.randint(len(game.positions))
        return game.make_input(i), game.make_output(i)

    return zip(*[random_pos(game) for game in numpy.random.choice(games, size=batch_size)])


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


def make_network():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(96, activation='sigmoid'),
        tf.keras.layers.Dense(96, activation='sigmoid'),
        tf.keras.layers.Dense(96, activation='sigmoid'),
        tf.keras.layers.Dense(65, activation='sigmoid')
    ])


def save_network(path, model):
    weights = [w.numpy().transpose() for w in model.weights]

    with open(path, "wb") as file:
        file.write(struct.pack('i', len(weights) // 2))
        for weight in weights:
            shape = weight.shape
            if len(shape) == 1:
                file.write(struct.pack('ll', shape[0], 1))
            else:
                file.write(struct.pack('ll', shape[0], shape[1]))
            file.write(weight.tostring())


def policy_loss(target, output):
    legal = tf.greater_equal(target, 0)

    target_ = tf.where(legal, target, 0.)
    output_ = tf.where(legal, output, -1e9)

    return tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(target_), output_)


def value_loss(target, output):
    return tf.losses.mean_squared_error(target, output)


def policy_value_loss(target, output):
    return policy_loss(target[:,:64], output[:,:64]) + value_loss(target[:,64], output[:,64])


def main(save_path, files):
    games = sum([load_data(path) for path in files], [])

    model = make_network()
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=policy_value_loss)

    for _ in range(1000):
        x, y = sample_batch(games, 4096)
        model.fit(x, y, epochs=4, batch_size=4096)

    save_network(save_path, model)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
