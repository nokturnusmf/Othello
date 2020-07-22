#pragma once

#include <vector>

#include "tree.h"

struct NeuralNet;

void mcts(Tree* tree, NeuralNet& net, int iterations, bool noise = true);

Tree* select_move_proportional(Tree* tree);
Tree* select_move_visit_count(Tree* tree);

void init_next(Tree* tree, const float* inf);
void backprop(std::vector<Tree*>& path, float value);

void add_exploration_noise(Tree* tree);
Tree* select_child(Tree* tree);
float action_value(const Tree* next, int parent_visit);

void expand_board(float* out, const Board& board, Colour colour);
