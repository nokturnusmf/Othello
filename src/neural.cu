#include "neural.h"

#include <vector>

#include <cudnn.h>
#include <cublas_v2.h>

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

struct CublasException : public std::exception {
    CublasException(cublasStatus_t status) : status(status) {}

    const char* what() const noexcept {
        switch (status) {
            case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
            default:                             return "Unknown";
        }
    }

    cublasStatus_t status;
};

#define cudaCheckError(f) if (auto _error_code = f) { throw CudaException(_error_code); }
#define cudnnCheckError(f) if (auto _error_code = f) { throw CudnnException(_error_code); }
#define cublasCheckError(f) if (auto _error_code = f) { throw CublasException(_error_code); }

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
    cudnnTensorDescriptor_t norm_desc;
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
        cudnnCheckError(cudnnCreateTensorDescriptor(&norm_desc));
        cudnnCheckError(cudnnCreateActivationDescriptor(&activation));

        cudaCheckError(cudaMalloc(&workspace, this->workspace_size = workspace_size));
    }

    ~CudnnParams() {
        cudaFree(workspace);
        cudnnDestroy(handle);
    }
};

struct CublasParams {
    static CublasParams* get() {
        thread_local CublasParams p;
        return &p;
    }

    CublasParams(const CudnnParams&) = delete;
    CublasParams(CudnnParams&&) = delete;

    cublasHandle_t handle;

private:
    CublasParams() {
        cublasCheckError(cublasCreate(&handle));
    }

    ~CublasParams() {
        cublasDestroy(handle);
    }
};

struct Convolution {
    Convolution() {}

    Convolution(const Convolution&) = delete;

    Convolution(Convolution&& other) {
        this->~Convolution();

        this->dims = other.dims;

        this->kernel = other.kernel;
        other.kernel = nullptr;
    }

    ~Convolution() {
        cudaFree(kernel);
    }

    void operator()(const float* input, float* output) const {
        static const float one  = 1.0f;
        static const float zero = 0.0f;

        auto cudnn = CudnnParams::get();

        int padding = (dims.h - 1) / 2;
        auto algorithm = (dims.h == 3 || dims.h == 5) ? CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        cudnnCheckError(cudnnSetFilter4dDescriptor(cudnn->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dims.o, dims.i, dims.h, dims.w));
        cudnnCheckError(cudnnSetConvolution2dDescriptor(cudnn->conv_desc, padding, padding, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

        cudnnCheckError(cudnnConvolutionForward(
            cudnn->handle, &one, cudnn->input_desc, input, cudnn->filter_desc, kernel, cudnn->conv_desc,
            algorithm, cudnn->workspace, cudnn->workspace_size, &zero, cudnn->output_desc, output));
    }

    struct Dimensions {
        int o;
        int i;
        int h;
        int w;
    };

    float* kernel = nullptr;

    Dimensions dims;
};

struct BatchNorm {
    BatchNorm() {}

    BatchNorm(const BatchNorm&) = delete;

    BatchNorm(BatchNorm&& other) {
        this->~BatchNorm();

        this->channels = other.channels;

        this->gamma = other.gamma;
        this->beta = other.beta;
        this->mean = other.mean;
        this->var = other.var;

        other.gamma = nullptr;
    }

    ~BatchNorm() {
        cudaFree(gamma);
    }

    void operator()(const float* input, float* output, const float* residual = nullptr) const {
        static const float one  = 1.f;
        static const float zero = 0.f;

        auto cudnn = CudnnParams::get();

        auto op = CUDNN_NORM_OPS_NORM; // residual ? CUDNN_NORM_OPS_NORM_ADD_ACTIVATION : CUDNN_NORM_OPS_NORM_ACTIVATION;

        cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->norm_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channels, 1, 1));
        cudnnCheckError(cudnnSetActivationDescriptor(cudnn->activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

        cudnnCheckError(cudnnNormalizationForwardInference(
            cudnn->handle, CUDNN_NORM_PER_CHANNEL, op, CUDNN_NORM_ALGO_STANDARD, &one, &zero,
            cudnn->input_desc, input, cudnn->norm_desc, gamma, beta, cudnn->norm_desc, mean, var,
            cudnn->input_desc, residual, cudnn->activation, cudnn->output_desc, output, 0.001, 1));

        // temporary workaround until CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are supported
        if (residual) cudnnCheckError(cudnnAddTensor(cudnn->handle, &one, cudnn->output_desc, residual, &one, cudnn->output_desc, output));
        cudnnCheckError(cudnnActivationForward(cudnn->handle, cudnn->activation, &one, cudnn->output_desc, output, &zero, cudnn->output_desc, output));
    }

    int channels;

    float* gamma = nullptr;
    float* beta = nullptr;

    float* mean = nullptr;
    float* var = nullptr;
};

struct ResBlock {
    ResBlock() {}
    ResBlock(const ResBlock&) = delete;
    ResBlock(ResBlock&& other) = default;

    void operator()(const float* input, float* output, float* intermediate) {
        conv1(input, intermediate);
        bn1(intermediate, output);

        conv2(output, intermediate);
        bn2(intermediate, output, input);
    }

    Convolution conv1;
    BatchNorm bn1;
    Convolution conv2;
    BatchNorm bn2;
};

__global__ void add_bias(float* out, const float* bias, int n, int m, bool relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += bias[i % m];
        if (relu && out[i] < 0) out[i] = 0;
    }
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

    void operator()(const float* input, float* output, int batch_size, bool relu) const {
        static const float one  = 1.f;
        static const float zero = 0.f;

        auto cublas = CublasParams::get();

        cublasCheckError(cublasSgemm(cublas->handle, CUBLAS_OP_T, CUBLAS_OP_N, dims.h, batch_size, dims.w, &one, weight, dims.w, input, dims.w, &zero, output, dims.h));

        int n = batch_size * dims.h;
        int threads = std::min(n, 128);
        int blocks = (n + threads - 1) / threads;
        add_bias<<<blocks, threads>>>(output, bias, n, dims.h, relu);
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
        cudaCheckError(cudaMemcpy(dst,  value_output,  3 * count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float* tensor_mem_a;
    float* tensor_mem_b;
    float* tensor_mem_c;

    float* policy_output;
    float* value_output;

    Convolution conv1;
    BatchNorm bn1;

    std::vector<ResBlock> tower;

    Convolution policy_conv;
    BatchNorm policy_bn;
    Dense policy_fc;

    Convolution value_conv;
    BatchNorm value_bn;
    Dense value_fc1;
    Dense value_fc2;
};

CudaNeuralNet::CudaNeuralNet(int max_batch_size, int filters) : NeuralNet(max_batch_size) {
    cudaCheckError(cudaMalloc(&tensor_mem_a, max_batch_size * filters * 64 * sizeof(float)));
    cudaCheckError(cudaMalloc(&tensor_mem_b, max_batch_size * filters * 64 * sizeof(float)));
    cudaCheckError(cudaMalloc(&tensor_mem_c, max_batch_size * filters * 64 * sizeof(float)));

    cudaCheckError(cudaMalloc(&policy_output, max_batch_size * 60 * sizeof(float)));
    cudaCheckError(cudaMalloc(&value_output,  max_batch_size *  3 * sizeof(float)));
}

CudaNeuralNet::~CudaNeuralNet() {
    cudaFree(tensor_mem_a);
    cudaFree(tensor_mem_b);
    cudaFree(tensor_mem_c);

    cudaFree(policy_output);
    cudaFree(value_output);
}

void CudaNeuralNet::compute(const float* input, int count) {
    auto cudnn = CudnnParams::get();

    cudaCheckError(cudaMemcpy(tensor_mem_a, input, count * 2 * 8 * 8 * sizeof(float), cudaMemcpyHostToDevice));

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, conv1.dims.i, 8, 8));
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, conv1.dims.o, 8, 8));
    conv1(tensor_mem_a, tensor_mem_c);
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, conv1.dims.o, 8, 8));
    bn1(tensor_mem_c, tensor_mem_b);

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, tower.front().conv1.dims.i, 8, 8));
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, tower.front().conv1.dims.o, 8, 8));
    for (auto& conv : tower) {
        std::swap(tensor_mem_a, tensor_mem_b);
        conv(tensor_mem_a, tensor_mem_b, tensor_mem_c);
    }

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, policy_conv.dims.i, 8, 8));
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, policy_conv.dims.o, 8, 8));
    policy_conv(tensor_mem_b, tensor_mem_c);
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, policy_conv.dims.o, 8, 8));
    policy_bn(tensor_mem_c, tensor_mem_a);
    policy_fc(tensor_mem_a, policy_output, count, false);

    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, value_conv.dims.i, 8, 8));
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, value_conv.dims.o, 8, 8));
    value_conv(tensor_mem_b, tensor_mem_c);
    cudnnCheckError(cudnnSetTensor4dDescriptor(cudnn->input_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, count, value_conv.dims.o, 8, 8));
    value_bn(tensor_mem_c, tensor_mem_a);
    value_fc1(tensor_mem_a, tensor_mem_c, count, true);
    value_fc2(tensor_mem_c, value_output, count, false);
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

    auto buffer = std::make_unique<float[]>(n);
    file.read(reinterpret_cast<char*>(buffer.get()), n * sizeof(float));

    cudaCheckError(cudaMemcpy(conv->kernel, buffer.get(), n * sizeof(float), cudaMemcpyHostToDevice));
}

void load_norm(BatchNorm* norm, std::istream& file) {
    norm->channels = read<int>(file);

    auto sz = norm->channels * sizeof(float);

    cudaCheckError(cudaMalloc(&norm->gamma, sz * 4));
    norm->beta = &norm->gamma[norm->channels * 1];
    norm->mean = &norm->gamma[norm->channels * 2];
    norm->var  = &norm->gamma[norm->channels * 3];

    auto buffer = std::make_unique<float[]>(sz * 4);
    file.read(reinterpret_cast<char*>(buffer.get()), sz * 4);
    cudaCheckError(cudaMemcpy(norm->gamma, buffer.get(), sz * 4, cudaMemcpyHostToDevice));
}

void load_resblock(ResBlock* block, std::istream& file) {
    load_conv(&block->conv1, file);
    load_norm(&block->bn1, file);
    load_conv(&block->conv2, file);
    load_norm(&block->bn2, file);
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
    if (version != 3) throw;

    auto filters = read<int>(file);
    auto tower_size = read<int>(file);

    auto network = std::make_unique<CudaNeuralNet>(max_batch_size, filters);

    load_conv(&network->conv1, file);
    load_norm(&network->bn1, file);

    for (int i = 0; i < tower_size; ++i) {
        network->tower.emplace_back();
        load_resblock(&network->tower.back(), file);
    }

    auto policy_conv_count = read<int>(file);
    auto policy_fc_count = read<int>(file);
    if (policy_conv_count != 1 || policy_fc_count != 1) throw;

    load_conv(&network->policy_conv, file);
    load_norm(&network->policy_bn, file);
    load_dense(&network->policy_fc, file);

    auto value_conv_count = read<int>(file);
    auto value_fc_count = read<int>(file);
    if (value_conv_count != 1 || value_fc_count != 2) throw;

    load_conv(&network->value_conv, file);
    load_norm(&network->value_bn, file);
    load_dense(&network->value_fc1, file);
    load_dense(&network->value_fc2, file);

    return network;
}
