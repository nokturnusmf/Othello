#pragma once

#include <memory>
#include <istream>

class NeuralNet {
public:
    NeuralNet(int max_batch_size) : max_batch_size(max_batch_size) {}

    virtual ~NeuralNet() {}

    virtual void compute(const float* input, int count) = 0;

    virtual void retrieve_policy(float* dst, int count) const = 0;
    virtual void retrieve_value(float* dst, int count) const = 0;

    int get_max_batch_size() const {
        return max_batch_size;
    }

private:
    int max_batch_size;
};

std::unique_ptr<NeuralNet> load_net(std::istream& file, int max_batch_size);
