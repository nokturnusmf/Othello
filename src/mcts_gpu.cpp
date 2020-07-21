#include "mcts.h"

#include <memory>

#include "neural_gpu.h"

const float VIRTUAL_LOSS = 0.1f;

struct Batch {
    Batch(const NeuralNet& net, int max_size)
        : net(net), input(128, 1, nullptr, max_size), data(std::make_unique<float[]>(128 * max_size)) {
        this->max_size = max_size;
    }

    void add_input(std::vector<Tree*>&& path) {
        if (entries.size() == max_size) {
            this->go();
        }
        entries.emplace_back(path);
    }

    void go() {
        if (entries.empty()) return;

        for (int i = 0; i < entries.size(); ++i) {
            auto tree = entries[i].back();
            expand_board(&data[128 * i], tree->board, tree->colour);
        }

        upload_data(input.data, data.get(), 128 * entries.size());
        auto result = net(input, entries.size())->retrieve_all();

        for (int i = 0; i < entries.size(); ++i) {
            ensure_next(entries[i].back());
            if (entries[i].back()->next_cap != 1) init_next(entries[i].back(), &result[65 * i]);
            backprop(entries[i], result[65 * i + 64] * 2 - 1);
        }

        entries.clear();
    }

    const NeuralNet& net;
    Vector input;
    std::unique_ptr<float[]> data;

    std::vector<std::vector<Tree*>> entries;
    int max_size;
};

void backprop(std::vector<Tree*>& path, float value) {
    auto colour = path.back()->colour;
    auto n = path.back()->n_inflight;

    for (auto tree : path) {
        tree->n += n;
        tree->n_inflight -= n;

        auto value_ = tree->colour == colour ? value : -value;
        tree->w += (value_ - VIRTUAL_LOSS) * n;
    }
}

void mcts(Tree* tree, const NeuralNet& net, int iterations, bool noise) {
    Batch batch(net, 256);

    if (!tree->n) {
        tree->n_inflight = 1;
        tree->w += VIRTUAL_LOSS;
        batch.add_input(std::vector<Tree*>(1, tree));
        batch.go();
    }

    if (noise) add_exploration_noise(tree);

    int collisions = 0, max_collisions = 4;
    int cur_iterations = iterations, max_iterations = iterations * 3;
    for (int i = 0; i < cur_iterations; ++i) {
        tree->n_inflight++;
        tree->w += VIRTUAL_LOSS;

        auto sim = tree;

        std::vector<Tree*> path(1, sim);
        while (sim->n && !game_over(sim)) {
            sim = select_child(sim);

            sim->n_inflight++;
            sim->w += VIRTUAL_LOSS;

            path.push_back(sim);
        }

        if (game_over(sim)) {
            int s = net_score(sim->board, sim->colour);
            auto value = (s > 0) - (s < 0);
            backprop(path, value);
        } else {
            if (sim->n_inflight > 1) {
                if (++collisions > max_collisions) {
                    batch.go();
                    collisions = 0;
                }
                if (cur_iterations < max_iterations) {
                    ++cur_iterations;
                }
            } else {
                batch.add_input(std::move(path));
            }
        }
    }

    batch.go();
}
