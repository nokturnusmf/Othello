#include "self_play_data.h"

#include "board.h"

static const int SCALE = (1 << 23) - 1;

SearchProb::SearchProb(const Tree* tree, Move played) {
    this->board  = tree->board;
    this->colour = tree->colour;

    this->played = played;

    this->q = tree->w / tree->n;
    this->d = tree->d / tree->n;

    for (int i = 0; i < tree->next_count; ++i) {
        moves.emplace_back(MoveProb { tree->next[i].move, tree->next[i].tree->n });
    }
}

struct MoveProbPacked {
    MoveProbPacked() = default;

    MoveProbPacked(const MoveProb& p) {
        this->row = p.move.row;
        this->col = p.move.col;
        this->n   = p.n;
    }

    operator MoveProb() const {
        MoveProb p;
        p.move.row = this->row;
        p.move.col = this->col;
        p.n        = this->n;
        return p;
    }

    int row : 4;
    int col : 4;
    int n   : 24;
};

struct SearchProbPacked {
    SearchProbPacked() = default;

    SearchProbPacked(const SearchProb& p) {
        this->played_row = p.played.row;
        this->played_col = p.played.col;

        this->moves = p.moves.size();

        this->q = p.q * SCALE;
        this->d = p.d * SCALE;
    }

    operator SearchProb() const {
        SearchProb p;

        p.played.row = played_row;
        p.played.col = played_col;

        p.moves.resize(moves);

        p.q = static_cast<float>(q) / SCALE;
        p.d = static_cast<float>(d) / SCALE;

        return p;
    }

    long played_row : 4;
    long played_col : 4;
    long moves      : 8;
    long q          : 24;
    long d          : 24;
};

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

    game.resize(read<int>(file));

    Board board;
    Colour colour = Colour::Black;

    for (auto& pos : game) {
        pos = read<SearchProbPacked>(file);
        pos.board  = board;
        pos.colour = colour;

        for (auto& move : pos.moves) {
            move = read<MoveProbPacked>(file);
        }

        play_move(&board, pos.played, colour);
        if (available(board, other(colour))) {
            colour = other(colour);
        }
    }

    game.result = net_score(board);

    return game;
}

void save_game(std::ostream& file, const Game& game) {
    write<int>(file, game.size());
    for (auto& pos : game) {
        write<SearchProbPacked>(file, pos);
        for (auto& move : pos.moves) {
            write<MoveProbPacked>(file, move);
        }
    }
}
