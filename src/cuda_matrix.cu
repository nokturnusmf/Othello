#include "cuda_matrix.h"

Matrix::Matrix(long h, long w, const float* data, long d) : h(h), w(w), d(d) {
    cudaMalloc(&this->data, h * w * d * sizeof(float));
    if (data) {
        cudaMemcpy(this->data, data, h * w * d * sizeof(float), cudaMemcpyHostToDevice);
    }
}

Matrix::Matrix(Matrix&& other) {
    *this = std::move(other);
}

Matrix& Matrix::operator=(Matrix&& other) {
    cudaFree(this->data);

    this->h = other.h;
    this->w = other.w;
    this->d = other.d;
    this->data = other.data;

    other.data = nullptr;

    return *this;
}

Matrix::~Matrix() {
    if (this->data) cudaFree(this->data);
}

std::unique_ptr<float[]> Matrix::retrieve_data(long n) const {
    auto result = std::make_unique<float[]>(h * w);
    cudaMemcpy(result.get(), data + n * h * w, h * w * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

std::unique_ptr<float[]> Matrix::retrieve_all() const {
    auto result = std::make_unique<float[]>(h * w * d);
    cudaMemcpy(result.get(), data, h * w * d * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

Matrix load_matrix(std::istream& file) {
    long h, w;
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    file.read(reinterpret_cast<char*>(&w), sizeof(w));

    auto data = std::make_unique<float[]>(h * w);
    file.read(reinterpret_cast<char*>(data.get()), h * w * sizeof(float));

    Matrix result(h, w, data.get());

    return result;
}

void save_matrix(std::ostream& file, const Matrix& mat) {
    file.write(reinterpret_cast<const char*>(&mat.h), sizeof(mat.h));
    file.write(reinterpret_cast<const char*>(&mat.w), sizeof(mat.w));
    file.write(reinterpret_cast<const char*>(mat.retrieve_data().get()), mat.h * mat.w * sizeof(float));
}
