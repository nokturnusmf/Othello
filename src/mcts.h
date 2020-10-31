#pragma once

#include <vector>

#include "tree.h"

struct NeuralNet;

void mcts(Tree* tree, NeuralNet& net, int iterations, bool noise = true);

bool order_next(const Tree::Next& a, const Tree::Next& b);

Tree::Next* select_move_visit_count(Tree* tree);
const Tree::Next* select_move_visit_count(const Tree* tree);

Tree::Next* select_move_proportional(Tree* tree);
const Tree::Next* select_move_proportional(const Tree* tree);

void init_next(Tree* tree, const float* inf);
void backprop(std::vector<Tree*>& path, float* value, bool softmax);

void add_exploration_noise(Tree* tree);
Tree* select_child(Tree* tree);
float action_value(const Tree* next, int parent_visit);

void expand_board(float* out, const Board& board, Colour colour);
