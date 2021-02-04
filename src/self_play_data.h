#pragma once

#include <memory>
#include <vector>
#include <istream>
#include <ostream>

#include "tree.h"

struct MoveProb {
    Move move;
    int n;
};

struct SearchProb {
    SearchProb() = default;
    SearchProb(const Tree* tree, Move played);

    Board board;
    Colour colour;

    Move played;

    float q;
    float d;

    std::vector<MoveProb> moves;
};

struct Game : public std::vector<SearchProb> {
    int result;
};

Game load_game(std::istream& file);
void save_game(std::ostream& file, const Game& game);
