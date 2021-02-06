#pragma once

#include <memory>

struct Tree;

class SearchStopper {
public:
    virtual ~SearchStopper() {}

    virtual bool operator()(const Tree* tree) = 0;

    virtual void reset() = 0;
};

std::unique_ptr<SearchStopper> search_stop_iterations(int n);
std::unique_ptr<SearchStopper> search_stop_kld(int min, int max, int step, double threshold);
std::unique_ptr<SearchStopper> search_stop_time(long duration_ms);
