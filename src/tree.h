#pragma once

#include "board.h"
#include "block_allocator.h"

struct Tree {
    Tree(Tree&& other);
    Tree& operator=(Tree&& other);

    ~Tree();

    static Ticket<Tree> make_tree();
    static Ticket<Tree> make_tree(const Board& board, Colour colour);
    static Ticket<Tree> make_tree(const Board& board, Colour colour, int pass);

    struct Next {
        Ticket<Tree> tree = nullptr;
        Move move;
    };

    void add_next(const Next& n);

    Board board;
    Colour colour;
    int pass;

    Ticket<Next> next = nullptr;
    int next_count = 0;
    int next_cap = 0;

    int n = 0;
    int n_inflight = 0;
    float w = 0;
    float p = 0;

    inline float q() const {
        return n ? w / n : 0;
    }

    static inline BlockAllocator<Tree, 256, true> tree_allocator;
    static inline BlockAllocator<Next, 4096, true> next_allocator;

    static GarbageCollector<Tree, 256, true> tree_gc;

    template<typename T, unsigned N, bool B>
    friend class BlockAllocator;

private:
    Tree(const Board& board, Colour colour, int pass);

    Tree(const Tree&) = default;
    Tree& operator=(const Tree&) = default;
};

bool game_over(const Tree* tree);

void gen_next(Tree* tree, int depth = 1);
void ensure_next(Tree* tree, int depth = 1);
