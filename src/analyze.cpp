#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <optional>
#include <iomanip>
#include <cstdlib>

#include <getopt.h>

#include "board.h"
#include "tree.h"
#include "mcts.h"
#include "neural.h"

struct Position {
    Board board;
    Colour colour = Colour::Black;
};

struct Arguments {
    std::unique_ptr<NeuralNet> net;
    Position pos;
    int iterations = 10000;
};

std::optional<Move> parse_move(const char* move) {
    int row = move[1] - '1';
    int col = toupper(move[0]) - 'A';

    if (row >= 0 && row < 8 && col >= 0 && col < 8) {
        return Move { row, col };
    } else {
        return std::nullopt;
    }
}

std::optional<Position> parse_moves(const std::string& moves) {
    if (moves.size() % 2 != 0) return std::nullopt;

    Board board;
    Colour colour = Colour::Black;

    for (int i = 0; moves[i]; i += 2) {
        auto move = parse_move(&moves[i]);
        if (!move || !play_move(&board, *move, colour)) {
            return std::nullopt;
        }

        if (available(board, other(colour))) colour = other(colour);
    }

    return Position { board, colour };
}

std::optional<Position> parse_board(const std::string& str) {
    if (str.size() < 65) return std::nullopt;

    Board board { 0, 0 };
    for (int i = 63; i >= 0; --i) {
        char c = toupper(str[i]);
        if (c == 'B') {
            board.black |= 1UL << i;
        } else if (c == 'W') {
            board.white |= 1UL << i;
        }
    }

    switch (toupper(str[64])) {
    case 'B':
        return Position { board, Colour::Black };
    case 'W':
        return Position { board, Colour::White };
    default:
        return std::nullopt;
    }
}

std::optional<int> parse_int(const char* str, int base = 10) {
    char* end;
    int res = strtol(str, &end, base);
    return end > str ? std::make_optional(res) : std::nullopt;
}

std::optional<Arguments> parse_args(int argc, char** argv) {
    Arguments args;
    int batch_size = 256;

    static struct option long_options[] = {
        { "moves",      1, 0, 'm' },
        { "iterations", 1, 0, 'i' },
        { "batch-size", 1, 0, 'b' },
        { 0,            0, 0,  0  }
    };

    for (int c; (c = getopt_long(argc, argv, "m:b:i:r", long_options, 0)) != -1;) {
        switch (c) {
        case 'm':
            if (auto pos = parse_moves(optarg)) {
                args.pos = *pos;
                break;
            } else {
                std::cerr << "Error parsing moves\n";
                return std::nullopt;
            }

        case 'i':
            if (auto iterations = parse_int(optarg)) {
                args.iterations = *iterations;
            } else {
                std::cerr << "Invalid number of iterations: " << optarg << '\n';
                return std::nullopt;
            }
            break;

        case 'b':
            if (auto size = parse_int(optarg)) {
                batch_size = *size;
            } else {
                std::cerr << "Invalid batch size: " << optarg << '\n';
                return std::nullopt;
            }
            break;
        }
    }

    if (optind < argc) {
        std::ifstream file(argv[optind]);
        if (!file) {
            std::cerr << "Couldn't open net file\n";
            return std::nullopt;
        }

        args.net = load_net(file, batch_size);
        return args;
    } else {
        std::cerr << "Usage: " << argv[0] << " [options] <net>\n";
        return std::nullopt;
    }
}

std::ostream& operator<<(std::ostream& out, const Board& board) {
    out << "  ";
    for (int i = 0; i < 8; ++i) {
        out << static_cast<char>('A' + i) << ' ';
    }
    for (int i = 0; i < 8; ++i) {
        out << '\n' << (i + 1) << ' ';
        for (int j = 0; j < 8; ++j) {
            size_t b = move_bit(i, j);
            if (board.black & b) {
                out << "B ";
            } else if (board.white & b) {
                out << "W ";
            } else {
                out << ". ";
            }
        }
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, const Move& move) {
    if (move.pass()) {
        out << "--";
    } else {
        out << static_cast<char>(move.col + 'A') << move.row + 1;
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, const WDLProb& wdl) {
    out << wdl.w * 100 << " W / " << wdl.d * 100 << " D / " << wdl.l * 100 << " L";
    return out;
}

void build_pv(std::vector<Move>* moves, const Tree* tree, int minimum) {
    if (!tree->next || tree->n < minimum) return;

    auto best = select_move_visit_count(tree);

    moves->push_back(best->move);
    build_pv(moves, best->tree.get(), minimum);
}

void print_pv(const Tree* tree) {
    std::vector<Move> pv;
    build_pv(&pv, tree, tree->n / 1000);

    std::cout << std::setprecision(1);

    std::cout << "\nPV: ";
    for (auto& move : pv) {
        std::cout << move << ' ';
    }
    std::cout << "~ " << tree->wdl() << '\n';
}

void print_moves(const Tree* tree) {
    std::cout << std::setprecision(5);

    std::cout << "\n   |    P    | Nc / Np |    Q\n---+---------+---------+----------\n";

    auto buffer = std::make_unique<Tree::Next[]>(tree->next_count);
    std::partial_sort_copy(&tree->next[0], &tree->next[tree->next_count], &buffer[0], &buffer[tree->next_count], order_next);

    float n = tree->n;
    for (int i = tree->next_count - 1; i >= 0; --i) {
        auto move = buffer[i].move;
        auto sub = buffer[i].tree;

        float p = sub->p;
        float v = sub->n / n;
        float q = -sub->q();

        std::cout << move << " | " << std::setw(7) << p << " | " << std::setw(7) << v << " | " << std::setw(8) << q << '\n';
    }
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (!args) {
        return 1;
    }

    std::cout << std::fixed << std::right;
    std::cout << args->pos.board << '\n' << (args->pos.colour == Colour::Black ? "Black" : "White") << " to play\n";

    auto tree = Tree::make_tree(args->pos.board, args->pos.colour);
    mcts(tree.get(), *args->net, args->iterations, false);

    print_pv(tree.get());
    print_moves(tree.get());

    Tree::tree_allocator.deallocate(tree);
}
