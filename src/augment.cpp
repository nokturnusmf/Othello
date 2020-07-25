#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <cstring>

#include "board.h"
#include "self_play_data.h"

namespace std {
    template<>
    struct hash<Board> {
        size_t operator()(const Board& board) const {
            return board.black - board.white;
        }
    };
}

bool operator==(const Board& a, const Board& b) {
    return a.black == b.black && a.white == b.white;
}

struct Record {
    void add(const std::vector<MoveProb>& moves, float result) {
        w += result;
        for (auto& move : moves) {
            p[nn_index(Move { move.row, move.col })] += move.p;
        }
        ++n;
    }

    void calculate() {
        for (float& f : p) f /= n;
        w /= n;
    }

    float p[60] = {};
    float w = 0.f;
    int n = 0;
};

void mirror(Board* board, std::vector<MoveProb>* moves) {
    board->black = __builtin_bswap64(board->black);
    board->white = __builtin_bswap64(board->white);

    for (auto& move : *moves) {
        move.row = 7 - move.row;
    }
}

inline size_t diagonal(size_t x) {
    size_t t;
    size_t k1 = 0xaa00aa00aa00aa00UL;
    size_t k2 = 0xcccc0000cccc0000UL;
    size_t k4 = 0xf0f0f0f00f0f0f0fUL;
    t  =       x ^ (x << 36) ;
    x ^= k4 & (t ^ (x >> 36));
    t  = k2 & (x ^ (x << 18));
    x ^=       t ^ (t >> 18) ;
    t  = k1 & (x ^ (x <<  9));
    x ^=       t ^ (t >>  9) ;
    return x;
}

void rotate(Board* board, std::vector<MoveProb>* moves) {
    board->black = __builtin_bswap64(diagonal(board->black));
    board->white = __builtin_bswap64(diagonal(board->white));

    for (auto& move : *moves) {
        int t = move.col;
        move.col = 7 - move.row;
        move.row = t;
    }
}

void expand_board(float* out, const Board& board) {
    for (int i = 0; i < 64; ++i) {
        size_t b = 1UL << i;
        out[i + 0 ] = board.black & b ? 1 : 0;
        out[i + 64] = board.white & b ? 1 : 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <output> <inputs>\n";
        return 1;
    }

    std::unordered_map<Board, Record> map;

    for (int i = 2; i < argc; ++i) {
        std::ifstream file(argv[i]);
        if (!file) {
            std::cerr << "Skipping '" << argv[i] << "': " << std::strerror(errno) << '\n';
            continue;
        }

        size_t n;
        file.read(reinterpret_cast<char*>(&n), sizeof(n));

        for (size_t j = 0; j < n; ++j) {
            std::cout << '\r' << argv[i] << "... " << j + 1 << '/' << n << std::flush;

            auto game = load_game(file);
            int z = (game.result > 0) - (game.result < 0);

            for (auto& pos : game) {
                auto board = pos.colour == Colour::Black ? pos.board : flip(pos.board);
                float val  = pos.colour == Colour::Black ? z : -z;

                for (int k = 0; k < 4; ++k) {
                    map[board].add(pos.moves, val);
                    mirror(&board, &pos.moves);
                    map[board].add(pos.moves, val);
                    mirror(&board, &pos.moves);
                    rotate(&board, &pos.moves);
                }
            }
        }

        std::cout << '\n';
    }

    std::ofstream file(argv[1]);
    if (!file) {
        std::perror("Couldn't open output file");
        return 1;
    }

    size_t total = map.size();
    file.write(reinterpret_cast<char*>(&total), sizeof(total));

    std::cout << "Saving boards...\n";
    for (auto& entry : map) {
        entry.second.calculate();

        float buffer[128];
        expand_board(buffer, entry.first);
        file.write(reinterpret_cast<char*>(buffer), sizeof(buffer));
    }

    std::cout << "Saving moves...\n";
    for (auto& entry : map) {
        file.write(reinterpret_cast<char*>(entry.second.p), sizeof(entry.second.p));
    }

    std::cout << "Saving values...\n";
    for (auto& entry : map) {
        file.write(reinterpret_cast<char*>(&entry.second.w), sizeof(float));
    }
}
