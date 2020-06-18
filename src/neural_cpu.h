#pragma once

#include <memory>
#include <vector>
#include <istream>

struct Matrix {
    Matrix(long height, long width, const float* data = nullptr);

    float& operator()(long r, long c);
    float  operator()(long r, long c) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

    std::unique_ptr<float[]> data = nullptr;
    long h;
    long w;
};

using Vector = Matrix;

struct NeuralNet {
    struct Layer {
        Layer(Matrix w, Vector b)
            : weights(std::move(w)), biases(std::move(b)) {}

        Matrix weights;
        Vector biases;
    };

    std::vector<Layer> layers;

    Vector operator()(Vector input) const;
};

NeuralNet load_net(std::istream& file);
Matrix load_matrix(std::istream& file);
