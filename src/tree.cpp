#include "tree.h"

Tree::Tree(const Board& board, Colour colour, int pass)
    : board(board), colour(colour), pass(pass) {}

Tree::~Tree() {
    if (!next.get()) return;

    for (int i = 0; i < next_count; ++i) {
        tree_allocator.deallocate(next[i].tree);
    }

    next_allocator.deallocate(next, next_count);
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

    return *this;
}

WDLProb Tree::wdl() const {
    auto q = this->w / n;
    auto d = this->d / n;

    return {
        (1 + q - d) / 2,
        d,
        (1 - q - d) / 2
    };
}

bool game_over(const Tree* tree) {
    return tree->pass >= 2 || played(tree->board) == 64;
}

struct MoveAndBoard {
    Board board;
    Move move;
};

int get_moves(MoveAndBoard* results, const Board& board, Colour colour) {
    int count = 0;

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            auto copy = board;
            Move move = { i, j };
            if (play_move(&copy, move, colour)) {
                results[count++] = { copy, move };
            }
        }
    }

    return count;
}

void add_next(Tree* tree, const Board& board, Colour colour, char pass, Move move, int depth, int i) {
    auto subtree = Tree::make_tree(board, colour, pass);
    gen_next(subtree.get(), depth);

    tree->next[i] = Tree::Next { subtree, move };
}

void gen_next(Tree* tree, int depth) {
    if (depth == 0 || game_over(tree)) return;

    Colour next = other(tree->colour);

    MoveAndBoard moves[60];
    int count = get_moves(moves, tree->board, tree->colour);

    tree->next_count = count ? count : 1;
    tree->next = Tree::next_allocator.allocate(tree->next_count);

    if (!count) {
        add_next(tree, tree->board, next, tree->pass + 1, { -1, -1 }, depth, 0);
    } else for (int i = 0; i < count; ++i) {
        add_next(tree, moves[i].board, next, 0, moves[i].move, depth - 1, i);
    }
}

void ensure_next(Tree* tree, int depth) {
    if (depth == 0) return;

    if (!tree->next_count) {
        gen_next(tree, depth);
    } else for (int i = 0; i < tree->next_count; ++i) {
        ensure_next(tree->next[i].tree.get(), depth - 1);
    }
}
