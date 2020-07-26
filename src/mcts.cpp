#include "mcts.h"

#include <memory>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

#include "neural.h"

const float VIRTUAL_LOSS = 0.1f;

static thread_local std::default_random_engine gen(std::chrono::steady_clock::now().time_since_epoch().count());

struct Batch {
    Batch(NeuralNet& net)
        : net(net),
          input(std::make_unique<float[]>(128 * net.get_max_batch_size())),
          policy(std::make_unique<float[]>(60 * net.get_max_batch_size())),
          value(std::make_unique<float[]>(net.get_max_batch_size())) {}

    void add_input(std::vector<Tree*>&& path) {
        if (entries.size() == net.get_max_batch_size()) {
            this->go();
        }
        entries.emplace_back(path);
    }

    void go() {
        if (entries.empty()) return;

        int count = entries.size();

        for (int i = 0; i < count; ++i) {
            auto tree = entries[i].back();
            expand_board(&input[128 * i], tree->board, tree->colour);
        }

        net.compute(input.get(), count);

        for (auto& entry : entries) {
            ensure_next(entry.back());
        }

        net.retrieve_policy(policy.get(), count);
        net.retrieve_value(value.get(), count);

        for (int i = 0; i < count; ++i) {
            if (entries[i].back()->next_cap != 1) init_next(entries[i].back(), &policy[60 * i]);
            backprop(entries[i], value[i]);
        }

        entries.clear();
    }

    NeuralNet& net;

    std::unique_ptr<float[]> input;
    std::unique_ptr<float[]> policy;
    std::unique_ptr<float[]> value;

    std::vector<std::vector<Tree*>> entries;
};

void mcts(Tree* tree, NeuralNet& net, int iterations, bool noise) {
    Batch batch(net);

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

void init_next(Tree* tree, const float* inf) {
    std::array<float, 60> buffer;

    float max_p = std::numeric_limits<float>::lowest();
    for (int i = 0; i < tree->next_count; ++i) {
        float p = inf[nn_index(tree->next[i].move)];

        buffer[i] = p;
        max_p = std::max(max_p, p);
    }

    float total = 0.f;
    for (int i = 0; i < tree->next_count; ++i) {
        buffer[i] = std::exp(buffer[i] - max_p);
        total += buffer[i];
    }

    float scale = 1.f / total;
    for (int i = 0; i < tree->next_count; ++i) {
        tree->next[i].tree->p = buffer[i] * scale;
    }
}

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

void add_exploration_noise(Tree* tree) {
    const float noise = 0.3f;
    const float frac  = 0.25f;

    std::gamma_distribution<float> dist(noise);

    for (int i = 0; i < tree->next_count; ++i) {
        (tree->next[i].tree->p *= 1 - frac) += dist(gen) * frac;
    }
}

Tree* select_child(Tree* tree) {
    if (tree->next_count == 1) return tree->next[0].tree.get();

    Tree* best = tree->next[0].tree.get();
    auto best_score = action_value(best, tree->n + tree->n_inflight);

    for (int i = 1; i < tree->next_count; ++i) {
        auto score = action_value(tree->next[i].tree.get(), tree->n + tree->n_inflight);
        if (score > best_score) {
            best = tree->next[i].tree.get();
            best_score = score;
        }
    }

    return best;
}

float action_value(const Tree* next, int parent_visit) {
    static const float c_base = 19652.f;
    static const float c_init = 4.25f;

    auto c = std::log((1 + parent_visit + c_base) / c_base) + c_init;
    auto u = c * next->p * std::sqrt(static_cast<float>(parent_visit)) / (1.f + next->n + next->n_inflight);

    return u - next->q();
}

void expand_board(float* out, const Board& board, Colour colour) {
    auto ours   = colour == Colour::Black ? board.black : board.white;
    auto theirs = colour == Colour::Black ? board.white : board.black;

    for (int i = 0; i < 64; ++i) {
        size_t b = 1UL << i;
        out[i + 0 ] = ours   & b ? 1 : 0;
        out[i + 64] = theirs & b ? 1 : 0;
    }
}

Tree* select_move_proportional(Tree* tree) {
    int move = std::uniform_int_distribution<int>(0, tree->n - 1)(gen);

    int cur = 0;
    for (int j = 0; j < tree->next_count; ++j) {
        if (move <= (cur += tree->next[j].tree->n)) {
            return tree->next[j].tree.get();
        }
    }

    return tree;
}

Tree* select_move_visit_count(Tree* tree) {
    return std::max_element(&tree->next[0], &tree->next[tree->next_count], [](const Tree::Next& a, const Tree::Next& b) {
        return std::make_tuple(a.tree->n, -a.tree->w) < std::make_tuple(b.tree->n, -b.tree->w);
    })->tree.get();
}
