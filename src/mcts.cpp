#include "mcts.h"

#include <memory>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

#include "neural.h"

const int MAX_COLLISIONS = 4;

static thread_local std::default_random_engine gen(std::chrono::steady_clock::now().time_since_epoch().count());

struct Batch {
    Batch(NeuralNet& net)
        : net(net),
          input(std::make_unique<float[]>(128 * net.get_max_batch_size())),
          policy(std::make_unique<float[]>(60 * net.get_max_batch_size())),
          value(std::make_unique<float[]>(3 * net.get_max_batch_size())) {}

    void add_input(std::vector<Tree*>&& path) {
        if (path.back()->n_inflight >= MAX_COLLISIONS) {
            this->go();
            return;
        }

        entries.emplace_back(std::move(path));

        if (entries.size() == net.get_max_batch_size()) {
            this->go();
        }
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
            init_next(entries[i].back(), &policy[60 * i]);
            backprop(entries[i], &value[3 * i], true);
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
        batch.add_input(std::vector<Tree*>(1, tree));
        batch.go();
    }

    if (noise) add_exploration_noise(tree);

    for (int i = 0; i < iterations; ++i) {
        auto sim = tree;
        tree->n_inflight++;

        std::vector<Tree*> path(1, sim);
        while (sim->n && !game_over(sim)) {
            sim = select_child(sim);
            sim->n_inflight++;
            path.push_back(sim);
        }

        if (game_over(sim)) {
            int s = net_score(sim->board, sim->colour);
            float value[3] = {
                s >  0 ? 1.f : 0.f,
                s == 0 ? 1.f : 0.f,
                s <  0 ? 1.f : 0.f
            };
            backprop(path, value, false);
        } else {
            batch.add_input(std::move(path));
        }
    }

    batch.go();
}

void init_next(Tree* tree, const float* inf) {
    if (tree->next[0].move.pass()) return;

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

void backprop(std::vector<Tree*>& path, float* value, bool softmax) {
    auto colour = path.back()->colour;
    auto n = path.back()->n_inflight;

    if (softmax) {
        for (int i = 0; i < 3; ++i) {
            value[i] = std::exp(value[i]);
        }
        float scale = 1 / std::accumulate(&value[0], &value[3], 0.f);
        for (int i = 0; i < 3; ++i) {
            value[i] *= scale;
        }
    }

    float wl = (value[0] - value[2]) * n;
    float d = value[1] * n;

    for (auto node : path) {
        node->n += n;
        node->n_inflight -= n;

        node->w += node->colour == colour ? wl : -wl;
        node->d += d;
    }
}

void add_exploration_noise(Tree* tree) {
    const float noise = 0.3f;
    const float frac  = 0.25f;

    std::gamma_distribution<float> dist(noise);

    std::array<float, 60> buffer;
    float total = 0;

    for (int i = 0; i < tree->next_count; ++i) {
        buffer[i] = dist(gen);
        total += buffer[i];
    }

    float scale = frac / total;

    for (int i = 0; i < tree->next_count; ++i) {
        (tree->next[i].tree->p *= 1 - frac) += buffer[i] * scale;
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
    static const float c_base = 0.0001f;
    static const float c_init = 1.25f;

    auto c = std::log(1 + parent_visit * c_base) + c_init;
    auto n = std::sqrt(static_cast<float>(parent_visit)) / (1.f + next->n + next->n_inflight);
    auto u = c * next->p * n;

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

bool order_next(const Tree::Next& a, const Tree::Next& b) {
    return std::make_tuple(a.tree->n, -a.tree->w, a.tree->p) < std::make_tuple(b.tree->n, -b.tree->w, b.tree->p);
}

Tree::Next* select_move_visit_count(Tree* tree) {
    return std::max_element(&tree->next[0], &tree->next[tree->next_count], order_next);
}

const Tree::Next* select_move_visit_count(const Tree* tree) {
    return select_move_visit_count(const_cast<Tree*>(tree));
}

Tree::Next* select_move_proportional(Tree* tree) {
    int n = 0;
    for (int i = 0; i < tree->next_count; ++i) {
        n += tree->next[i].tree->n;
    }

    int move = std::uniform_int_distribution<int>(0, n - 1)(gen);

    int cur = 0;
    for (int i = 0; i < tree->next_count; ++i) {
        cur += tree->next[i].tree->n;
        if (move < cur) return &tree->next[i];
    }

    __builtin_unreachable();
}

const Tree::Next* select_move_proportional(const Tree* tree) {
    return select_move_proportional(const_cast<Tree*>(tree));
}
