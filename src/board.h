#pragma once

#include <cstddef>

enum class Colour {
    Black,
    White
};

struct Board {
    size_t black = 0x0000000810000000;
    size_t white = 0x0000001008000000;
};

struct Move {
    int row;
    int col;
};

static inline size_t move_bit(size_t r, size_t c) {
    return 1UL << (r * 8 + c);
}

Colour other(Colour c);

int played(const Board& board);
bool available(const Board& board, Colour colour);
bool game_over(const Board& board);

bool play_move(Board* board, const Move& move, Colour colour);

int individual_score(const Board& board, Colour colour);
int net_score(const Board& board);
