#include "neural_cpu.h"

#include <algorithm>
#include <cmath>

Matrix::Matrix(long height, long width, const float* data)
    : data(std::make_unique<float[]>(height * width)), h(height), w(width) {
    if (data) {
        std::copy(data, data + h * w, this->data.get());
    }
}

float& Matrix::operator()(long r, long c) {
    return data[r * w + c];
}

float Matrix::operator()(long r, long c) const {
    return data[r * w + c];
}

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result(h, w);
    for (long i = 0; i < h * w; ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    Matrix result(this->h, other.w);
    for (int i = 0; i < this->h; ++i) {
        for (int j = 0; j < other.w; ++j) {
            float v = 0;
            for (int k = 0; k < this->w; ++k) {
                v += (*this)(i, k) * other(k, j);
            }
            result(i, j) = v;
        }
    }
    return result;
}

static float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

Vector NeuralNet::operator()(Vector input) const {
    for (auto& layer : layers) {
        input = layer.weights * input + layer.biases;
        std::for_each(&input.data[0], &input.data[input.h], [](float& f) {
            f = sigmoid(f);
        });
    }
    return input;
}

NeuralNet load_net(std::istream& file) {
    NeuralNet result;

    int n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));

    for (int i = 0; i < n; ++i) {
        auto w = load_matrix(file);
        auto b = load_matrix(file);
        result.layers.emplace_back(std::move(w), std::move(b));
    }

    return result;
}

Matrix load_matrix(std::istream& file) {
    long h, w;
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    file.read(reinterpret_cast<char*>(&w), sizeof(w));

    Matrix result(h, w);
    file.read(reinterpret_cast<char*>(result.data.get()), h * w * sizeof(float));

    return result;
}
