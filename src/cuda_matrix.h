#include <memory>
#include <istream>

struct Matrix {
    Matrix(long height, long width, const float* data = nullptr, long depth = 1);
    ~Matrix();

    Matrix(const Matrix&) = delete;
    Matrix(Matrix&&);

    Matrix& operator=(const Matrix&) = delete;
    Matrix& operator=(Matrix&&);

    std::unique_ptr<float[]> retrieve_data(long n = 0) const;
    std::unique_ptr<float[]> retrieve_all() const;

    float* data = nullptr;
    long h;
    long w;
    long d;
};

using Vector = Matrix;

Matrix load_matrix(std::istream& file);
