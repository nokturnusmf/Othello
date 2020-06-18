#include "self_play_data.h"

MoveProb::MoveProb(Move move, int n, int parent_n) {
    row = move.row;
    col = move.col;
    p = static_cast<float>(n) / parent_n;
}

SearchProb::SearchProb(const Tree* tree) {
    this->board  = tree->board;
    this->colour = tree->colour;

    for (int i = 0; i < tree->next_count; ++i) {
        moves.emplace_back(tree->next[i].move, tree->next[i].tree->n, tree->n);
    }
}

template<typename T>
inline void write(std::ostream& file, const T& t) {
    file.write(reinterpret_cast<const char*>(&t), sizeof(T));
}

template<typename T>
inline T read(std::istream& file) {
    T t;
    file.read(reinterpret_cast<char*>(&t), sizeof(T));
    return t;
}

Game load_game(std::istream& file) {
    Game game;

    game.result = read<int>(file);
    game.resize(read<int>(file));

    for (auto& pos : game) {
        pos.board  = read<Board>(file);
        pos.colour = read<Colour>(file);
        pos.moves.resize(read<int>(file));

        for (auto& move : pos.moves) {
            move.row = read<int>(file);
            move.col = read<int>(file);
            move.p = read<float>(file);
        }
    }

    return game;
}

void save_game(std::ostream& file, const Game& game) {
    write<int>(file, game.result);
    write<int>(file, game.size());

    for (auto& pos : game) {
        write<Board>(file, pos.board);
        write<Colour>(file, pos.colour);
        write<int>(file, pos.moves.size());

        for (auto& move : pos.moves) {
            write<int>(file, move.row);
            write<int>(file, move.col);
            write<float>(file, move.p);
        }
    }
}
