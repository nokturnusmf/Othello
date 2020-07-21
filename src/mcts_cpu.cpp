#include "mcts.h"

#include "neural_cpu.h"

void expand_node(std::vector<Tree*> path, const NeuralNet& net) {
    auto tree = path.back();
    ensure_next(tree);
    if (tree->next_cap == 1) return;

    Vector input(128, 1);
    expand_board(input.data.get(), tree->board, tree->colour);

    auto result = net(std::move(input));

    if (tree->next_cap != 1) init_next(tree, result.data.get());
    backprop(path, result.data[64] * 2 - 1);
}

void backprop(std::vector<Tree*>& path, float value) {
    auto colour = path.back()->colour;

    for (auto tree : path) {
        tree->n++;
        tree->w += tree->colour == colour ? value : -value;
    }
}

void mcts(Tree* tree, const NeuralNet& net, int iterations, bool noise) {
    if (!tree->n) expand_node(std::vector(1, tree), net);
    if (noise) add_exploration_noise(tree);

    for (int i = 0; i < iterations; ++i) {
        auto sim = tree;

        std::vector<Tree*> path(1, sim);
        while (sim->next && !game_over(sim)) {
            sim = select_child(sim);
            path.push_back(sim);
        }

        if (game_over(sim)) {
            int s = net_score(sim->board, sim->colour);
            int value = (s > 0) - (s < 0);
            backprop(path, value);
        } else {
            expand_node(path, net);
        }
    }
}
