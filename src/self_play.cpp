#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>

#include "mcts.h"
#include "neural.h"
#include "self_play_data.h"

extern const int TREE_GC_THRESHOLD = 0;
extern const int TREE_GC_THREADS   = 0;

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
    while (running) {
        std::unique_lock lock(mut);
        if (--remaining < 0) break;
        lock.unlock();

        Game game;
        auto root = Tree::make_tree();
        auto tree = root.get();

        while (!game_over(tree)) {
            ensure_next(tree);

            if (tree->next_cap == 1) {
                tree = tree->next[0].tree.get();
                continue;
            }

            mcts(tree, net, 1600);
            game.emplace_back(tree);
            tree = played(tree->board) < 32 ? select_move_proportional(tree) : select_move_visit_count(tree);
        }

        game.result = net_score(tree->board);

        lock.lock();
        buffer.add_game(std::move(game));
        std::cout << "\rGenerating games: " << buffer.pos << std::flush;
        lock.unlock();

        Tree::tree_allocator.deallocate(root);
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
    auto net = load_net(net_file, 256);

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
