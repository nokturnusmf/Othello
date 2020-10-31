#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <cstring>

#include "board.h"
#include "self_play_data.h"

struct Record {
    void add(const std::vector<MoveProb>& moves, float result) {
        w[result ? result > 0 ? 0 : 2 : 1] += 1;

        for (auto& move : moves) {
            p[nn_index(Move { move.row, move.col })] += move.p;
        }

        ++n;
    }

    void calculate() {
        for (float& f : p) f /= n;
        for (float& f : w) f /= n;
    }

    float p[60] = {};
    float w[3] = {};
    int n = 0;
};

void mirror(std::vector<MoveProb>* moves) {
    for (auto& move : *moves) {
        move.row = 7 - move.row;
    }
}

void transpose(std::vector<MoveProb>* moves) {
    for (auto& move : *moves) {
        std::swap(move.row, move.col);
    }
}

void expand_board(float* out, const Board& board) {
    for (int i = 0; i < 64; ++i) {
        size_t b = 1UL << i;
        out[i + 0 ] = board.black & b ? 1 : 0;
        out[i + 64] = board.white & b ? 1 : 0;
    }
}

void process_position(std::unordered_map<Board, Record>& map, SearchProb pos, int z) {
    auto board = pos.colour == Colour::Black ? pos.board : flip(pos.board);
    float val  = pos.colour == Colour::Black ? z : -z;

    for (int k = 0; k < 4; ++k) {
        map[board].add(pos.moves, val);
        board = mirror(board);
        mirror(&pos.moves);

        map[board].add(pos.moves, val);
        board = transpose(board);
        transpose(&pos.moves);
    }
}

void process_file(std::unordered_map<Board, Record>& map, const char* path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Skipping '" << path << "': " << std::strerror(errno) << '\n';
        return;
    }

    size_t n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));

    for (size_t j = 0; j < n; ++j) {
        std::cout << '\r' << path << "... " << j + 1 << '/' << n << std::flush;

        auto game = load_game(file);
        int z = (game.result > 0) - (game.result < 0);

        for (auto& pos : game) {
            process_position(map, std::move(pos), z);
        }
    }

    std::cout << '\n';
}

int output(std::unordered_map<Board, Record>& map, const char* path) {
    std::ofstream file(path);
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
        file.write(reinterpret_cast<char*>(entry.second.w), sizeof(entry.second.w));
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <output> <inputs>\n";
        return 1;
    }

    std::unordered_map<Board, Record> map;

    for (int i = 2; i < argc; ++i) {
        process_file(map, argv[i]);
    }

    return output(map, argv[1]);
}
