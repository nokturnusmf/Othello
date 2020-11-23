#pragma once

#include <vector>
#include <mutex>

template<typename T>
struct Ticket {
    Ticket(T* p = nullptr) : ptr(p) {}

    T* get() { return ptr; }
    const T* get() const { return ptr; }

    T* operator->() { return ptr; }
    const T* operator->() const { return ptr; }

    operator bool() const  { return  ptr; }
    bool operator!() const { return !ptr; }

    bool operator==(const Ticket<T>& other) { return ptr == other.ptr; }
    bool operator!=(const Ticket<T>& other) { return ptr != other.ptr; }

    T& operator[](size_t n) { return ptr[n]; }
    const T& operator[](size_t n) const { return ptr[n]; }

    T* ptr;
    size_t index;
};

template<typename T, unsigned N, bool Destroy>
class BlockAllocator {
public:
    BlockAllocator() = default;

    BlockAllocator(const BlockAllocator<T, N, Destroy>&) = delete;
    BlockAllocator& operator=(const BlockAllocator<T, N, Destroy>&) = delete;

    ~BlockAllocator();

    template<class... Args>
    Ticket<T> allocate(size_t n = 1, Args&&... args);

    void deallocate(const Ticket<T>& ticket, size_t n = 1);

    void add_block();
    void clear();

    size_t in_use() const;
    size_t allocated() const;

private:
    struct Block {
        char* ptr;
        int position;
        int freed;
    };

    std::vector<Block> blocks;

    std::mutex mut;
};

#include "block_allocator.impl"
