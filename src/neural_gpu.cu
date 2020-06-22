#include "neural_gpu.h"

#include <cmath>

static __device__ float sigmoid(float f) {
    return 1 / (1 + std::exp(-f));
}

__global__ void eval_layer_kernel(float* output, const float* input, const float* weights, const float* biases, int input_length) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadIdx.x;
    output[o] = 0;

    for (int j = 0; j < input_length; ++j) {
        output[o] += weights[i * input_length + j] * input[blockIdx.x * input_length + j];
    }

    output[o] += biases[i];
    output[o] = sigmoid(output[o]);
}

Vector* NeuralNet::operator()(const Vector& input, long depth, Vector* result) const {
    const Vector* prev = &input;
    for (auto& layer : layers) {
        eval_layer_kernel<<<depth, layer.activations.h>>>(layer.activations.data, prev->data, layer.weights.data, layer.biases.data, prev->h);
        prev = &layer.activations;
    }

    auto& final = layers.back().activations;
    if (result) {
        cudaMemcpy(result->data, final.data, final.h * input.d * sizeof(float), cudaMemcpyDeviceToDevice);
        return result;
    } else {
        return &final;
    }
}

void upload_data(float* dst, const float* src, int n) {
    cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice);
}

NeuralNet load_net(std::istream& file, long depth) {
    NeuralNet result;
    int n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    result.layers.reserve(n);
    for (int i = 0; i < n; ++i) {
        auto w = load_matrix(file);
        auto b = load_matrix(file);
        result.layers.emplace_back(std::move(w), std::move(b), depth);
    }
    return result;
}
