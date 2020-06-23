#include "tree.h"

extern const int TREE_GC_THRESHOLD;
extern const int TREE_GC_THREADS;

GarbageCollector<Tree, 256, true> Tree::tree_gc(&Tree::tree_allocator, TREE_GC_THRESHOLD, TREE_GC_THREADS);

Tree::Tree(const Board& board, Colour colour, int pass)
    : board(board), colour(colour), pass(pass) {}

Tree::~Tree() {
    if (!next.get()) return;

    for (int i = 0; i < next_count; ++i) {
        tree_allocator.deallocate(next[i].tree);
    }

    next_allocator.deallocate(next, next_cap);
}

Ticket<Tree> Tree::make_tree() {
    return make_tree(Board(), Colour::Black);
}

Ticket<Tree> Tree::make_tree(const Board& board, Colour colour) {
    return make_tree(board, colour, 0);
}

Ticket<Tree> Tree::make_tree(const Board& board, Colour colour, int pass) {
    return tree_allocator.allocate(1, board, colour, pass);
}

Tree::Tree(Tree&& other) {
    *this = std::move(other);
}

Tree& Tree::operator=(Tree&& other) {
    *this = other;

    other.next = nullptr;
    other.next_count = 0;
    other.next_cap = 0;

    return *this;
}

void Tree::add_next(const Next& n) {
    if (!next.get()) {
        next = next_allocator.allocate(next_cap = 12);
    } else if (next_count == next_cap) {
        auto new_cap = next_cap * 3 / 2;
        auto replace = next_allocator.allocate(new_cap);

        for (int i = 0; i < next_count; ++i) {
            replace[i] = next[i];
        }

        next_allocator.deallocate(next, next_cap);

        next = replace;
        next_cap = new_cap;
    }

    next[next_count++] = n;
}

bool game_over(const Tree* tree) {
    return tree->pass >= 2 || played(tree->board) == 64;
}

void gen_next(Tree* tree, int depth) {
    if (depth == 0 || game_over(tree)) return;

    Colour next = other(tree->colour);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            Board board = tree->board;
            if (play_move(&board, { i, j }, tree->colour)) {
                auto sub = Tree::make_tree(board, next);
                gen_next(sub.get(), depth - 1);
                tree->add_next(Tree::Next { sub, { i, j } });
            }
        }
    }

    if (!tree->next_count) {
        auto sub = Tree::make_tree(tree->board, next, tree->pass + 1);
        gen_next(sub.get(), depth);
        tree->next = Tree::next_allocator.allocate(tree->next_cap = 1);
        tree->add_next(Tree::Next { sub, { -1, -1 } });
    }
}

void ensure_next(Tree* tree, int depth) {
    if (depth == 0) return;

    if (!tree->next_count) {
        gen_next(tree, depth);
    }

    for (int i = 0; i < tree->next_count; ++i) {
        ensure_next(tree->next[i].tree.get(), depth - 1);
    }
}
