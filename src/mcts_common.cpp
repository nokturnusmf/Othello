#include "mcts.h"

#include <array>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

static thread_local std::default_random_engine gen(std::chrono::steady_clock::now().time_since_epoch().count());

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

void init_next(Tree* tree, const float* inf) {
    std::array<float, 60> buffer;

    float max_p = std::numeric_limits<float>::lowest();
    for (int i = 0; i < tree->next_count; ++i) {
        auto move = tree->next[i].move;
        float p = inf[move.row * 8 + move.col];

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
    static const float c_init = 1.25f;

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
