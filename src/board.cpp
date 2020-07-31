#include "board.h"

#include <utility>
#include <cstdint>

static inline int popcount(size_t x) {
    int r = 0;
    while (x) {
        x &= x - 1;
        ++r;
    }
    return r;
}

static inline size_t changes_shl(const Board* board, size_t rc, int shift, size_t mask) {
    size_t changes = 0;
    while (board->white & ((rc <<= shift) &= mask)) {
        changes |= rc;
    }
    return board->black & rc ? changes : 0;
}

static inline size_t changes_shr(const Board* board, size_t rc, int shift, size_t mask) {
    size_t changes = 0;
    while (board->white & ((rc >>= shift) &= mask)) {
        changes |= rc;
    }
    return board->black & rc ? changes : 0;
}

static inline size_t row_mask(int r) {
    return 0xFFUL << (r * 8);
}

static inline size_t diagonal_mask_a(int r, int c) {
    const size_t data[] = {
        0x8040201008040201, 0x0080402010080402, 0x0000804020100804, 0x0000008040201008, 0x0000000080402010, 0x0000000000804020, 0x0000000000008040, 0x0000000000000080,
        0x4020100804020100, 0x2010080402010000, 0x1008040201000000, 0x0804020100000000, 0x0402010000000000, 0x0201000000000000, 0x0100000000000000
    };

    return r > c ? data[7 + r - c] : data[c - r];
}

static inline size_t diagonal_mask_b(int r, int c) {
    const size_t data[] = {
        0x0000000000000001, 0x0000000000000102, 0x0000000000010204, 0x0000000001020408, 0x0000000102040810, 0x0000010204081020, 0x0001020408102040, 0x0102040810204080,
        0x0204081020408000, 0x0408102040800000, 0x0810204080000000, 0x1020408000000000, 0x2040800000000000, 0x4080000000000000, 0x8000000000000000
    };

    return data[r + c];
}

bool play_move(Board* board, const Move& move, Colour colour) {
    size_t rc = move_bit(move.row, move.col);
    if ((board->black | board->white) & rc) {
        return false;
    }

    if (colour == Colour::White) {
        std::swap(board->black, board->white);
    }

    size_t r_mask = row_mask(move.row);
    size_t d_mask_a = diagonal_mask_a(move.row, move.col);
    size_t d_mask_b = diagonal_mask_b(move.row, move.col);

    size_t changes = changes_shl(board, rc, 8, -1)
                   | changes_shr(board, rc, 8, -1)
                   | changes_shl(board, rc, 1, r_mask)
                   | changes_shr(board, rc, 1, r_mask)
                   | changes_shl(board, rc, 9, d_mask_a)
                   | changes_shr(board, rc, 9, d_mask_a)
                   | changes_shl(board, rc, 7, d_mask_b)
                   | changes_shr(board, rc, 7, d_mask_b);

    if (changes) {
        changes |= rc;

        board->black |= changes;
        board->white &= ~changes;
    }

    if (colour == Colour::White) {
        std::swap(board->black, board->white);
    }

    return changes;
}

Colour other(Colour c) {
    return c == Colour::Black ? Colour::White : Colour::Black;
}

Board flip(const Board& board) {
    return Board { board.white, board.black };
}

static inline size_t mirror(size_t x) {
    size_t k1 = 0x00FF00FF00FF00FFUL;
    size_t k2 = 0x0000FFFF0000FFFFUL;
    x = ((x >>  8) & k1) | ((x & k1) <<  8);
    x = ((x >> 16) & k2) | ((x & k2) << 16);
    x = ( x >> 32)       | ( x       << 32);
    return x;
}

static inline size_t transpose(size_t x) {
    size_t t;
    size_t k1 = 0x5500550055005500UL;
    size_t k2 = 0x3333000033330000UL;
    size_t k4 = 0x0f0f0f0f00000000UL;
    t  = k4 & (x ^ (x << 28));
    x ^=       t ^ (t >> 28) ;
    t  = k2 & (x ^ (x << 14));
    x ^=       t ^ (t >> 14) ;
    t  = k1 & (x ^ (x <<  7));
    x ^=       t ^ (t >>  7) ;
    return x;
}

Board mirror(const Board& board) {
    return { mirror(board.black), mirror(board.white) };
}

Board transpose(const Board& board) {
    return { transpose(board.black), transpose(board.white) };
}

bool Board::operator==(const Board& other) const {
    return this->black == other.black && this->white == other.white;
}

bool Move::pass() const {
    return this->row < 0;
}

int nn_index(const Move& move) {
    int i = move.row * 8 + move.col;
    if (i > 36) {
        return i - 4;
    } else if (i > 28) {
        return i - 2;
    } else {
        return i;
    }
}

int played(const Board& board) {
    return popcount(board.black) + popcount(board.white);
}

bool available(const Board& board, Colour colour) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            Board b = board;
            if (play_move(&b, { i, j }, colour)) {
                return true;
            }
        }
    }

    return false;
}

bool game_over(const Board& board) {
    return played(board) >= 64 || (!available(board, Colour::Black) && !available(board, Colour::White));
}

int individual_score(const Board& board, Colour colour) {
    return popcount(colour == Colour::Black ? board.black : board.white);
}

int net_score(const Board& board, Colour colour) {
    int score = popcount(board.black) - popcount(board.white);
    return colour == Colour::Black ? score : -score;
}
