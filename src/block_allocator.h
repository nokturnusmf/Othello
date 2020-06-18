#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

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

template<typename T, unsigned N, bool Destroy>
class GarbageCollector {
public:
    GarbageCollector(BlockAllocator<T, N, Destroy>* allocator, long threshold = 1, int thread_count = 1);

    GarbageCollector(const GarbageCollector<T, N, Destroy>&) = delete;
    GarbageCollector& operator=(const GarbageCollector<T, N, Destroy>&) = delete;

    ~GarbageCollector();

    void enqueue(Ticket<T> ticket, size_t n = 1);

private:
    void worker();

    struct Deallocation {
        Deallocation(Ticket<T> t, size_t n) : ticket(t), n(n) {}
        Ticket<T> ticket;
        size_t n;
    };

    BlockAllocator<T, N, Destroy>* allocator;

    std::vector<Deallocation> queue;
    long threshold;

    std::vector<std::thread> thread_pool;
    bool running;

    std::mutex mut;
    std::condition_variable cv;
};

#include "block_allocator.impl"
