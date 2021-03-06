#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <optional>

#include <getopt.h>

#include "board.h"
#include "tree.h"
#include "mcts.h"
#include "neural.h"

class Engine {
public:
    Engine(std::unique_ptr<NeuralNet>&& net, int iterations, float temperature, float threshold, std::string&& name);
    Engine(Engine&& other);
    ~Engine();

    void new_game();

    Move make_move();

    void update(Move move);

    const std::string& get_name() const { return name; }

private:
    std::unique_ptr<NeuralNet> net;

    int iterations;
    float temperature;
    float threshold;

    Ticket<Tree> root = nullptr;
    Tree* current = nullptr;

    std::string name;
};

struct Arguments {
    std::vector<Engine> nets;
    int repeats = 1;
    std::vector<std::vector<Move>> book;
};

struct Game {
    Game(std::vector<Move>&& moves, int result)
        : moves(std::move(moves)), result(result) {}

    std::vector<Move> moves;
    int result;
};

struct WDL {
    float score() const;

    WDL& operator+=(const WDL& other);
    WDL operator~() const;

    int w = 0;
    int d = 0;
    int l = 0;
};

struct Series {
    void add_game(Game&& game);

    std::vector<Game> games;
    WDL wdl;
};

struct Tournament {
    Tournament(std::vector<Engine>&& engines);

    void run(const std::vector<std::vector<Move>>& book, int repeats);

    void print_scores() const;
    void print_table() const;
    void print_games() const;
    void print_all() const;

    std::vector<Engine> engines;

    std::vector<std::vector<Series>> results;
};

std::ostream& operator<<(std::ostream& out, const Move& move) {
    out << static_cast<char>(move.col + 'a') << move.row + 1;
    return out;
}

std::ostream& operator<<(std::ostream& out, const Game& game) {
    for (auto move : game.moves) {
        out << move;
    }
    out << ": " << game.result;
    return out;
}

std::ostream& operator<<(std::ostream& out, const WDL& wdl) {
    out << wdl.w << '/' << wdl.d << '/' << wdl.l;
    return out;
}

std::optional<long> parse_int(const char* str, int base = 10) {
    char* end;
    long res = strtol(str, &end, base);
    return end > str ? std::make_optional(res) : std::nullopt;
}

std::optional<float> parse_float(const char* str) {
    char* end;
    float res = strtof(str, &end);
    return end > str ? std::make_optional(res) : std::nullopt;
}

std::unique_ptr<NeuralNet> load_net(const char* path, int batch_size) {
    std::ifstream file(path);
    if (!file) {
        return nullptr;
    }

    return load_net(file, batch_size);
}

std::vector<std::vector<Move>> load_book(const char* path) {
    std::ifstream file(path);

    std::vector<std::vector<Move>> book;

    std::string line;
    while (std::getline(file, line)) {
        std::vector<Move> moves;

        for (size_t i = 0; i < line.size(); i += 2) {
            Move move { line[i + 1] - '1', line[i] - 'a' };
            moves.push_back(move);
        }

        book.emplace_back(std::move(moves));
    }

    return book;
}

std::optional<Arguments> parse_args(int argc, char** argv) {
    Arguments args;

    std::optional<int> iterations;
    float temperature = 0;
    float threshold = 2;
    int batch_size = 32;

    static struct option long_options[] = {
        { "iterations",  1, 0, 'i' },
        { "temperature", 1, 0, 'T' },
        { "threshold",   1, 0, 't' },
        { "repeats",     1, 0, 'r' },
        { "batch-size",  1, 0, 'b' },
        { "book",        1, 0, 'B' },
        { 0,             0, 0,  0  }
    };

    for (int c; (c = getopt_long(argc, argv, "i:T:t:r:b:B:", long_options, 0)) != -1;) {
        switch (c) {
        case 'i':
            if (auto i = parse_int(optarg)) {
                iterations = i;
            } else {
                std::cerr << "Invalid number of iterations: " << optarg << '\n';
                return std::nullopt;
            }
            break;

        case 'T':
            if (auto t = parse_float(optarg)) {
                temperature = *t;
            } else {
                std::cerr << "Invalid temperature: " << optarg << '\n';
                return std::nullopt;
            }
            break;

        case 't':
            if (auto t = parse_float(optarg)) {
                threshold = *t;
            } else {
                std::cerr << "Invalid threshold: " << optarg << '\n';
                return std::nullopt;
            }
            break;

        case 'r':
            if (auto r = parse_int(optarg)) {
                args.repeats = *r;
            } else {
                std::cerr << "Invalid number of repeats: " << optarg << '\n';
                return std::nullopt;
            }
            break;

        case 'b':
            if (auto b = parse_int(optarg)) {
                batch_size = *b;
            } else {
                std::cerr << "Invalid batch size: " << optarg << '\n';
                return std::nullopt;
            }
            break;

        case 'B':
            args.book = load_book(optarg);
            if (args.book.empty()) {
                std::cerr << "Error loading book\n";
                return std::nullopt;
            }
        }
    }

    if (argc - optind < 2 || !iterations) {
        std::cerr << "Usage: " << argv[0] << " [options] -i <iterations> <nets...>\n";
        return std::nullopt;
    }

    if (args.book.empty()) {
        args.book.emplace_back();
    }

    for (int i = optind; i < argc; ++i) {
        if (auto net = load_net(argv[i], batch_size)) {
            args.nets.emplace_back(std::move(net), *iterations, temperature, threshold, argv[i]);
        } else {
            std::cerr << "Couldn't open file '" << argv[i] << "'\n";
            return std::nullopt;
        }
    }

    return args;
}

Engine::Engine(std::unique_ptr<NeuralNet>&& net, int iterations, float temperature, float threshold, std::string&& name)
    : net(std::move(net)), name(std::move(name)) {
    this->iterations  = iterations;
    this->temperature = temperature;
    this->threshold   = threshold;
}

Engine::Engine(Engine&& other) {
    std::swap(this->net, other.net);

    this->iterations  = other.iterations;
    this->temperature = other.temperature;
    this->threshold   = other.threshold;

    std::swap(this->root, other.root);
    std::swap(this->current, other.current);

    std::swap(this->name, other.name);
}

Engine::~Engine() {
    Tree::tree_allocator.deallocate(root);
}

void Engine::new_game() {
    Tree::tree_allocator.deallocate(root);

    root = Tree::make_tree();
    current = root.get();
}

Move Engine::make_move() {
    mcts(current, *net, *search_stop_iterations(iterations), false);
    auto next = temperature ? select_move_temperature(current, temperature, threshold)
                            : select_move_visit_count(current);
    return next->move;
}

void Engine::update(Move move) {
    ensure_next(current);

    for (int i = 0; i < current->next_count; ++i) {
        if (current->next[i].move == move) {
            current = current->next[i].tree.get();
            break;
        }
    }

    ensure_next(current);

    if (current->next[0].move.pass()) {
        current = current->next[0].tree.get();
    }
}

Game game(Engine& black, Engine& white, const std::vector<Move>& book) {
    Board board;
    Colour turn = Colour::Black;

    std::vector<Move> moves;

    black.new_game();
    white.new_game();

    for (auto move : book) {
        play_move(&board, move, turn);
        if (available(board, other(turn))) turn = other(turn);

        moves.push_back(move);

        black.update(move);
        white.update(move);
    }

    while (true) {
        auto move = (turn == Colour::Black ? black : white).make_move();

        if (!play_move(&board, move, turn)) {
            throw move;
        }

        moves.push_back(move);

        if (game_over(board)) {
            break;
        } else if (available(board, other(turn))) {
            turn = other(turn);
        }

        black.update(move);
        white.update(move);
    }

    return Game(std::move(moves), net_score(board));
}

float WDL::score() const {
    return w + 0.5 * d;
}

WDL& WDL::operator+=(const WDL& other) {
    this->w += other.w;
    this->d += other.d;
    this->l += other.l;

    return *this;
}

WDL WDL::operator~() const {
    return { l, d, w };
}

void Series::add_game(Game&& game) {
    (game.result ? game.result > 0 ? wdl.w : wdl.l : wdl.d)++;
    games.emplace_back(std::move(game));
}

Tournament::Tournament(std::vector<Engine>&& engines) : engines(std::move(engines)) {
    results.resize(this->engines.size());
    for (auto& row : results) {
        row.resize(this->engines.size());
    }
}

void Tournament::run(const std::vector<std::vector<Move>>& book, int repeats) {
    for (int i = 0; i < engines.size(); ++i) {
        for (int j = 0; j < engines.size(); ++j) {
            if (i == j) continue;

            for (auto& line : book) {
                for (int k = 0; k < repeats; ++k) {
                    results[i][j].add_game(game(engines[i], engines[j], line));
                }
            }
        }
    }
}

void Tournament::print_scores() const {
    std::cout << std::left;

    for (int i = 0; i < engines.size(); ++i) {
        WDL wdl;

        for (int j = 0; j < engines.size(); ++j) {
            if (i == j) continue;

            wdl +=  results[i][j].wdl;
            wdl += ~results[j][i].wdl;
        }

        std::cout << engines[i].get_name() << ": " << std::setw(5) << wdl.score() << " (" << wdl << ")\n";
    }
}

void Tournament::print_table() const {
    std::cout << std::left;

    for (int i = 0; i < engines.size(); ++i) {
        std::cout << engines[i].get_name();

        for (int j = 0; j < engines.size(); ++j) {
            std::cout << " | ";
            if (i == j) {
                std::cout << "-   ";
            } else {
                std::cout << std::setw(4) << results[i][j].wdl.score();
            }
        }

        std::cout << '\n';
    }
}

void Tournament::print_games() const {
    for (int i = 0; i < engines.size(); ++i) {
        for (int j = 0; j < engines.size(); ++j) {
            if (i == j) continue;

            std::cout << engines[i].get_name() << " vs " << engines[j].get_name() << ": "
                      << results[i][j].wdl << " (reverse " << ~results[j][i].wdl << ")\n";

            for (auto& game : results[i][j].games) {
                std::cout << game << '\n';
            }

            std::cout << '\n';
        }

        std::cout << '\n';
    }
}

void Tournament::print_all() const {
    print_scores();
    std::cout << "\n\n";
    print_table();
    std::cout << "\n\n";
    print_games();
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (!args) {
        return 1;
    }

    Tournament tournament(std::move(args->nets));
    tournament.run(args->book, args->repeats);
    tournament.print_all();
}
