#pragma once

#include <memory>
#include <vector>
#include <istream>
#include <ostream>

#include "tree.h"

struct MoveProb {
    MoveProb() = default;
    MoveProb(Move move, int n, int parent_n);

    int row;
    int col;
    float p;
};

struct SearchProb {
    SearchProb() = default;
    SearchProb(const Tree* tree);

    Board board;
    Colour colour;

    std::vector<MoveProb> moves;
};

struct Game : public std::vector<SearchProb> {
    int result;
};

Game load_game(std::istream& file);
void save_game(std::ostream& file, const Game& game);
