#pragma once

#include <vector>
#include <istream>

#include "cuda_matrix.h"

struct NeuralNet {
    struct Layer {
        Layer(Matrix&& w, Vector&& b, long depth = 1)
            : weights(std::move(w)), biases(std::move(b)), activations(this->biases.h, 1, nullptr, depth) {}

        Matrix weights;
        Vector biases;

        mutable Vector activations;
    };

    std::vector<Layer> layers;

    Vector* operator()(const Vector& input, long depth, Vector* result = nullptr) const;
};

void upload_data(float* dst, const float* src, int n);

NeuralNet load_net(std::istream& file, long depth = 1);
