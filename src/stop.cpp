#include "stop.h"

#include <vector>
#include <chrono>
#include <cmath>

#include "tree.h"

class IterationStopper : public SearchStopper {
public:
    IterationStopper(int n) : n(n) {}

    bool operator()(const Tree* tree) {
        return tree->n + tree->n_inflight >= n;
    }

    void reset() {}

private:
    int n;
};

std::unique_ptr<SearchStopper> search_stop_iterations(int n) {
    return std::make_unique<IterationStopper>(n);
}

class KldStopper : public SearchStopper {
public:
    KldStopper(int min, int max, int step, double threshold)
        : min(min), max(max), step(step), threshold(threshold) {}

    bool operator()(const Tree* tree) {
        if (tree->n < min) return false;
        if (tree->n >= max) return true;
        if (tree->n - previous_n < step) return false;

        previous_n = tree->n;

        if (previous.empty()) {
            previous = get_counts(tree);
            return false;
        }

        auto current = get_counts(tree);
        auto x = kld(current, previous);
        previous = current;

        return x < threshold;
    }

    void reset() {
        previous_n = 0;
        previous.clear();
    }

    static std::vector<int> get_counts(const Tree* tree) {
        std::vector<int> p;
        for (int i = 0; i < tree->next_count; ++i) {
            p.push_back(tree->next[i].tree->n);
        }
        return p;
    }

    static double kld(const std::vector<int>& p, const std::vector<int>& q) {
        double n1 = 0, n2 = 0;
        for (int i = 0; i < p.size(); ++i) {
            n1 += p[i];
            n2 += q[i];
        }

        double x = 0;
        for (int i = 0; i < p.size(); ++i) {
            if (q[i]) {
                auto p_ = p[i] / n1;
                auto q_ = q[i] / n2;
                x += p_ * std::log(p_ / q_);
            }
        }

        return x / (n1 - n2);
    }

private:
    int min;
    int max;
    int step;
    double threshold;

    int previous_n;
    std::vector<int> previous;
};

std::unique_ptr<SearchStopper> search_stop_kld(int min, int max, int step, double threshold) {
    return std::make_unique<KldStopper>(min, max, step, threshold);
}

class TimeStopper : public SearchStopper {
public:
    TimeStopper(long duration) : duration(duration) {}

    bool operator()(const Tree* tree) {
        return tree->n > 1 && std::chrono::steady_clock::now() > end;
    }

    void reset() {
        end = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration);
    }

private:
    long duration;

    std::chrono::steady_clock::time_point end;
};

std::unique_ptr<SearchStopper> search_stop_time(long duration_ms) {
    return std::make_unique<TimeStopper>(duration_ms);
}
