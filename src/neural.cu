#include "neural.h"

#include <vector>

#include <cudnn.h>

struct CudaException : public std::exception {
    CudaException(cudaError_t status) : status(status) {}

    const char* what() const noexcept {
        return cudaGetErrorName(status);
    }

    cudaError_t status;
};

struct CudnnException : public std::exception {
    CudnnException(cudnnStatus_t status) : status(status) {}

    const char* what() const noexcept {
        return cudnnGetErrorString(status);
    }

    cudnnStatus_t status;
};

#define cudaCheckError(f) if (auto x = f) { throw CudaException(x); }
#define cudnnCheckError(f) if (auto x = f) { throw CudnnException(x); }

struct CudnnParams {
    static CudnnParams* get() {
        thread_local CudnnParams p(256 * 1024 * 1024);
        return &p;
    }

    CudnnParams(const CudnnParams&) = delete;
    CudnnParams(CudnnParams&&) = delete;

    cudnnHandle_t handle;

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnActivationDescriptor_t activation;

    void* workspace;
    size_t workspace_size;

private:
    CudnnParams(size_t workspace_size) {
        cudnnCheckError(cudnnCreate(&handle));

        cudnnCheckError(cudnnCreateTensorDescriptor(&input_desc));
        cudnnCheckError(cudnnCreateTensorDescriptor(&output_desc));

        cudnnCheckError(cudnnCreateFilterDescriptor(&filter_desc));
        cudnnCheckError(cudnnCreateConvolutionDescriptor(&conv_desc));
        cudnnCheckError(cudnnCreateTensorDescriptor(&bias_desc));
        cudnnCheckError(cudnnCreateActivationDescriptor(&activation));

        cudaCheckError(cudaMalloc(&workspace, this->workspace_size = workspace_size));
    }

    ~CudnnParams() {
        cudnnDestroy(handle);
    }
};

struct Convolution {
    Convolution() {}

    Convolution(const Convolution&) = delete;

    Convolution(Convolution&& other) {
        this->~Convolution();

        this->kernel = other.kernel;
        this->bias = other.bias;

        this->dims = other.dims;

        other.kernel = nullptr;
        other.bias = nullptr;
    }

    ~Convolution() {
        cudaFree(kernel);
        cudaFree(bias);
    }

    void operator()(float* input, float* output) const {
        static const float alpha = 1.0f;
        static const float beta  = 0.0f;

        auto cudnn = CudnnParams::get();

        int padding = (dims.h - 1) / 2;
        auto algorithm = dims.h == 3 ? CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        cudnnCheckError(cudnnSetFilter4dDescriptor(cudnn->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dims.o, dims.i, dims.h, dims.w));
        cudnnCheckError(cudnnSetConvolution2dDescriptor(cudnn->conv_desc, padding, padding, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
        cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, dims.o, 1, 1));
        cudnnCheckError(cudnnSetActivationDescriptor(cudnn->activation, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));

        cudnnCheckError(cudnnConvolutionBiasActivationForward(
            cudnn->handle, &alpha, cudnn->input_desc, input, cudnn->filter_desc, kernel, cudnn->conv_desc, algorithm, cudnn->workspace,
            cudnn->workspace_size, &beta, cudnn->output_desc, output, cudnn->bias_desc, bias, cudnn->activation, cudnn->output_desc, output));
    }

    struct Dimensions {
        int o;
        int i;
        int h;
        int w;
    };

    float* kernel = nullptr;
    float* bias = nullptr;

    Dimensions dims;
};

__global__ void eval_layer_kernel(float* output, const float* input, const float* weights, const float* biases, int input_length, bool relu) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadIdx.x;
    output[o] = 0;

    for (int j = 0; j < input_length; ++j) {
        output[o] += weights[i * input_length + j] * input[blockIdx.x * input_length + j];
    }

    output[o] += biases[i];

    if (relu && output[o] < 0) output[o] = 0;
}

struct Dense {
    Dense() {}

    Dense(const Dense&) = delete;

    Dense(Dense&& other) {
        this->~Dense();

        this->weight = other.weight;
        this->bias = other.bias;

        this->dims = other.dims;

        other.weight = nullptr;
        other.bias = nullptr;
    }

    ~Dense() {
        cudaFree(weight);
        cudaFree(bias);
    }

    void operator()(float* input, float* output, int batch_size, bool relu) const {
        eval_layer_kernel<<<batch_size, dims.h>>>(output, input, weight, bias, dims.w, relu);
    }

    struct Dimensions {
        int h;
        int w;
    };

    float* weight = nullptr;
    float* bias = nullptr;

    Dimensions dims;
};

struct CudaNeuralNet : public NeuralNet {
    CudaNeuralNet(int max_batch_size, int filters);
    ~CudaNeuralNet();

    CudaNeuralNet(const CudaNeuralNet&) = delete;
    CudaNeuralNet(CudaNeuralNet&&) = delete;

    void compute(const float* input, int count);

    void retrieve_policy(float* dst, int count) const {
        cudaCheckError(cudaMemcpy(dst, policy_output, 60 * count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void retrieve_value(float* dst, int count) const {
        cudaCheckError(cudaMemcpy(dst, value_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float* tensor_mem_a;
    float* tensor_mem_b;
    float* policy_output;
    float* value_output;

    Convolution conv1;
    std::vector<Convolution> tower;

    Convolution policy_conv;
    Dense policy_fc;

    Convolution value_conv;
    Dense value_fc1;
    Dense value_fc2;
};

CudaNeuralNet::CudaNeuralNet(int max_batch_size, int filters) : NeuralNet(max_batch_size) {
    cudaCheckError(cudaMalloc(&tensor_mem_a, max_batch_size * filters * 64 * sizeof(float)));
    cudaCheckError(cudaMalloc(&tensor_mem_b, max_batch_size * filters * 64 * sizeof(float)));

    cudaCheckError(cudaMalloc(&policy_output, max_batch_size * 61 * sizeof(float)));
    value_output = &policy_output[max_batch_size * 60];
}

CudaNeuralNet::~CudaNeuralNet() {
    cudaFree(tensor_mem_a);
    cudaFree(tensor_mem_b);
    cudaFree(policy_output);
}

void CudaNeuralNet::compute(const float* input, int count) {
    auto cudnn = CudnnParams::get();

    cudaCheckError(cudaMemcpy(tensor_mem_a, input, count * 2 * 8 * 8 * sizeof(float), cudaMemcpyHostToDevice));

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, conv1.dims.i, 8, 8));
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, conv1.dims.o, 8, 8));

    conv1(tensor_mem_a, tensor_mem_b);

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, tower.front().dims.i, 8, 8));

    for (auto& conv : tower) {
        std::swap(tensor_mem_a, tensor_mem_b);
        conv(tensor_mem_a, tensor_mem_b);
    }

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, policy_conv.dims.o, 8, 8));
    policy_conv(tensor_mem_b, tensor_mem_a);
    policy_fc(tensor_mem_a, policy_output, count, false);

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, value_conv.dims.o, 8, 8));
    value_conv(tensor_mem_b, tensor_mem_a);
    value_fc1(tensor_mem_a, tensor_mem_b, count, true);
    value_fc2(tensor_mem_b, value_output, count, false);
}

template<typename T>
inline T read(std::istream& file) {
    T t;
    file.read(reinterpret_cast<char*>(&t), sizeof(T));
    return t;
}

void load_conv(Convolution* conv, std::istream& file) {
    conv->dims = read<Convolution::Dimensions>(file);

    auto n = conv->dims.h * conv->dims.w * conv->dims.i * conv->dims.o;

    cudaCheckError(cudaMalloc(&conv->kernel, n * sizeof(float)));
    cudaCheckError(cudaMalloc(&conv->bias, conv->dims.o * sizeof(float)));

    auto buffer = std::make_unique<float[]>(n);

    file.read(reinterpret_cast<char*>(buffer.get()), n * sizeof(float));
    cudaCheckError(cudaMemcpy(conv->kernel, buffer.get(), n * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.get()), conv->dims.o * sizeof(float));
    cudaCheckError(cudaMemcpy(conv->bias, buffer.get(), conv->dims.o * sizeof(float), cudaMemcpyHostToDevice));
}

void load_dense(Dense* dense, std::istream& file) {
    dense->dims = read<Dense::Dimensions>(file);

    auto n = dense->dims.h * dense->dims.w;

    cudaCheckError(cudaMalloc(&dense->weight, n * sizeof(float)));
    cudaCheckError(cudaMalloc(&dense->bias, dense->dims.h * sizeof(float)));

    auto buffer = std::make_unique<float[]>(n);

    file.read(reinterpret_cast<char*>(buffer.get()), n * sizeof(float));
    cudaCheckError(cudaMemcpy(dense->weight, buffer.get(), n * sizeof(float), cudaMemcpyHostToDevice));

    file.read(reinterpret_cast<char*>(buffer.get()), dense->dims.h * sizeof(float));
    cudaCheckError(cudaMemcpy(dense->bias, buffer.get(), dense->dims.h * sizeof(float), cudaMemcpyHostToDevice));
}

std::unique_ptr<NeuralNet> load_net(std::istream& file, int max_batch_size) {
    auto version = read<int>(file);
    if (version != 1) throw;

    auto filters = read<int>(file);
    auto tower_size = read<int>(file);

    auto network = std::make_unique<CudaNeuralNet>(max_batch_size, filters);

    load_conv(&network->conv1, file);

    for (int i = 0; i < tower_size; ++i) {
        network->tower.emplace_back();
        load_conv(&network->tower.back(), file);
    }

    auto policy_conv_count = read<int>(file);
    auto policy_fc_count = read<int>(file);
    if (policy_conv_count != 1 || policy_fc_count != 1) throw;

    load_conv(&network->policy_conv, file);
    load_dense(&network->policy_fc, file);

    auto value_conv_count = read<int>(file);
    auto value_fc_count = read<int>(file);
    if (value_conv_count != 1 || value_fc_count != 2) throw;

    load_conv(&network->value_conv, file);
    load_dense(&network->value_fc1, file);
    load_dense(&network->value_fc2, file);

    return network;
}
