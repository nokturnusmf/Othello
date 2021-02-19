#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>

#include "mcts.h"
#include "neural.h"
#include "self_play_data.h"

struct GameStats {
    int black = 0;
    int white = 0;
    int draw  = 0;
};

struct GameBuffer {
    GameBuffer(size_t capacity)
        : games(std::make_unique<Game[]>(capacity)) {
        this->pos = 0;
    }

    void add_game(Game&& game) {
        games[pos++] = std::move(game);
    }

    GameStats stats() const {
        GameStats result;
        for (int i = 0; i < pos; ++i) {
            int r = games[i].result;
            if (r > 0) ++result.black;
            else if (r < 0) ++result.white;
            else ++result.draw;
        }
        return result;
    }

    std::unique_ptr<Game[]> games;
    size_t pos;
};

class SelfPlay {
public:
    SelfPlay(NeuralNet& net, long games, int thread_count) : net(net), buffer(games) {
        this->remaining = games;
        this->running = true;
        for (int i = 0; i < thread_count; ++i) {
            thread_pool.emplace_back(&SelfPlay::worker, this);
        }
    }

    ~SelfPlay() {
        running = false;
        this->wait();
    }

    void wait() {
        for (auto& thread : thread_pool) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void save(std::ostream& file) {
        file.write(reinterpret_cast<const char*>(&buffer.pos), sizeof(buffer.pos));
        for (size_t i = 0; i < buffer.pos; ++i) {
            save_game(file, buffer.games[i]);
        }
    }

    GameStats stats() const {
        return buffer.stats();
    }

private:
    void worker();

    NeuralNet& net;
    GameBuffer buffer;

    long remaining;

    std::vector<std::thread> thread_pool;
    bool running;

    std::mutex mut;
};

void SelfPlay::worker() {
    auto stop = search_stop_kld(200, 5000, 100, 6e-6);

    while (running) {
        std::unique_lock lock(mut);
        if (--remaining < 0) break;
        lock.unlock();

        Game game;

        Board board;
        Colour colour = Colour::Black;

        while (!game_over(board)) {
            auto tree = Tree::make_tree(board, colour);
            mcts(tree.get(), net, *stop, true);

            float temperature = std::exp(-std::pow(played(board) / 40, 3));
            auto next = select_move_temperature(tree.get(), temperature);
            game.emplace_back(tree.get(), next->move);

            board  = next->tree->board;
            colour = next->tree->colour;
            if (!available(board, colour)) colour = other(colour);

            Tree::tree_allocator.deallocate(tree);
        }

        game.result = net_score(board);

        lock.lock();
        buffer.add_game(std::move(game));
        std::cout << "\rGenerating games: " << buffer.pos << std::flush;
        lock.unlock();
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <net> <count> <output>\n";
        return 1;
    }

    int games = atol(argv[2]);
    if (games <= 0) {
        std::cerr << "Invalid count: " << argv[2] << '\n';
        return 1;
    }

    std::ifstream net_file(argv[1]);
    if (!net_file) {
        std::cerr << "Error loading net\n";
        return 1;
    }
    auto net = load_net(net_file, 32);

    SelfPlay sp(*net, games, 1);
    sp.wait();

    auto stats = sp.stats();
    std::cout << "\nBlack " << stats.black << ", white " << stats.white << ", draw " << stats.draw << '\n';

    std::ofstream out_file(argv[3]);
    if (!out_file) {
        std::cerr << "Couldn't open output file\n";
        return 1;
    }

    sp.save(out_file);
}
