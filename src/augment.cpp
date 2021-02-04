#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <cstring>

#include "board.h"
#include "self_play_data.h"

struct Record {
    void add(const std::vector<MoveProb>& moves, int z, float q, float d) {
        this->w[z ? z > 0 ? 0 : 2 : 1] += 1;

        this->q[0] += (1 + q - d) / 2;
        this->q[1] += d;
        this->q[2] += (1 - q - d) / 2;

        float N = 0;
        for (auto& move : moves) {
            N += move.n;
        }
        for (auto& move : moves) {
            this->p[nn_index(move.move)] += move.n / N;
        }

        ++this->n;
    }

    void calculate(float r = 0) {
        for (float& f : p) f /= n;

        for (int i = 0; i < 3; ++i) {
            w[i] /= n;
            q[i] /= n;

            w[i] = q[i] * r + w[i] * (1 - r);
        }
    }

    float p[60] = {};

    float w[3] = {};
    float q[3] = {};

    int n = 0;
};

void transpose(SearchProb* pos) {
    pos->board = transpose(pos->board);
    for (auto& m : pos->moves) {
        std::swap(m.move.row, m.move.col);
    }
}

void anti_transpose(SearchProb* pos) {
    pos->board = anti_transpose(pos->board);
    for (auto& m : pos->moves) {
        std::swap(m.move.row, m.move.col);
        m.move.row = 7 - m.move.row;
        m.move.col = 7 - m.move.col;
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
    z = pos.colour == Colour::Black ? z : -z;

    for (int i = 0; i < 4; ++i) {
        auto board = pos.colour == Colour::Black ? pos.board : flip(pos.board);
        map[board].add(pos.moves, z, pos.q, pos.d);
        i % 2 == 0 ? transpose(&pos) : anti_transpose(&pos);
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

    for (size_t i = 0; i < n; ++i) {
        std::cout << '\r' << path << "... " << i + 1 << '/' << n << std::flush;

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

    for (auto& entry : map) {
        entry.second.calculate();
    }

    std::cout << "Saving boards...\n";
    for (auto& entry : map) {
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
