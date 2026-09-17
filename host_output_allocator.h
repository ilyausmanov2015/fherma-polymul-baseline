#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <cassert>
#include <cstdio>
#include <atomic>
#include <chrono>
#ifndef FHERMA_OUTPUT_ALLOCATOR_SPIN
#define FHERMA_OUTPUT_ALLOCATOR_SPIN 0
#endif
#ifndef FHERMA_OUTPUT_ALLOCATOR_CPU_INDEX
#define FHERMA_OUTPUT_ALLOCATOR_CPU_INDEX -1
#endif
#ifdef __linux__
#include <sched.h>
#endif

// One ordinary output vector is allocated/zeroed per run, in parallel with input
// work. No result storage is preallocated or retained between calls.
class HostOutputAllocator {
#if !FHERMA_OUTPUT_ALLOCATOR_SPIN
    std::mutex mutex_;
    std::condition_variable ready_,done_;
    bool complete_=false,stop_=false;
#endif
    std::vector<uint32_t>* destination_=nullptr;
    size_t words_=0;
    bool busy_=false;
    std::exception_ptr error_;
    std::thread worker_;
#if FHERMA_HOST_PROFILE
    double resize_us_=0;
#endif
#if FHERMA_OUTPUT_ALLOCATOR_SPIN
    alignas(64) std::atomic<unsigned> request_{0};
    alignas(64) std::atomic<unsigned> completed_{0};
    alignas(64) std::atomic<bool> spin_stop_{false};
    unsigned generation_=0;
    static void pause() {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#else
        std::this_thread::yield();
#endif
    }
#endif
    void run(int cpu) {
#ifdef __linux__
        if(cpu>=0) {
            cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);
            sched_setaffinity(0,sizeof(mask),&mask);
        }
#else
        (void)cpu;
#endif
#if FHERMA_OUTPUT_ALLOCATOR_SPIN
        unsigned seen=0;
        while(!spin_stop_.load(std::memory_order_acquire)) {
            unsigned generation=request_.load(std::memory_order_acquire);
            if(generation==seen) {pause();continue;}
            seen=generation;
#if FHERMA_HOST_PROFILE
            auto start=std::chrono::steady_clock::now();
#endif
            try {destination_->resize(words_);} catch(...) {error_=std::current_exception();}
#if FHERMA_HOST_PROFILE
            resize_us_=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
#endif
            // Publishing completion also publishes all vector metadata, zeroed
            // elements and a possible exception. The caller owns the next job.
            completed_.store(generation,std::memory_order_release);
        }
#else
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;) {
            ready_.wait(lock,[&] {return stop_ || destination_!=nullptr;});
            if(stop_) return;
            auto* destination=destination_;auto words=words_;
            lock.unlock();std::exception_ptr error;
#if FHERMA_HOST_PROFILE
            auto start=std::chrono::steady_clock::now();
#endif
            try {destination->resize(words);} catch(...) {error=std::current_exception();}
#if FHERMA_HOST_PROFILE
            resize_us_=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
#endif
            lock.lock();destination_=nullptr;error_=error;complete_=true;done_.notify_one();
        }
#endif
    }
    void begin(std::vector<uint32_t>& destination,size_t words) {
#if FHERMA_OUTPUT_ALLOCATOR_SPIN
        assert(!busy_);busy_=true;error_=nullptr;destination_=&destination;words_=words;
        request_.store(++generation_,std::memory_order_release);
#else
        std::lock_guard<std::mutex> lock(mutex_);assert(!busy_);
        busy_=true;complete_=false;error_=nullptr;destination_=&destination;words_=words;
        ready_.notify_one();
#endif
    }
    void finish() {
#if FHERMA_OUTPUT_ALLOCATOR_SPIN
        if(!busy_) return;
        while(completed_.load(std::memory_order_acquire)!=generation_) pause();
        busy_=false;destination_=nullptr;auto error=error_;error_=nullptr;
        if(error) std::rethrow_exception(error);
#else
        std::unique_lock<std::mutex> lock(mutex_);
        if(!busy_) return;
        done_.wait(lock,[&] {return complete_;});
        busy_=false;auto error=error_;error_=nullptr;lock.unlock();
        if(error) std::rethrow_exception(error);
#endif
    }
    void finish_noexcept() noexcept {try {finish();} catch(...) {}}
public:
#if FHERMA_HOST_PROFILE
    double resize_us() const {assert(!busy_);return resize_us_;}
#endif
    HostOutputAllocator() {
        int cpu=-1;
#ifdef __linux__
        cpu_set_t allowed;
        if(sched_getaffinity(0,sizeof(allowed),&allowed)==0) {
            int index=0;
            for(int candidate=0;candidate<CPU_SETSIZE;++candidate)
                if(CPU_ISSET(candidate,&allowed)) {
                    cpu=candidate;
                    if(index++==FHERMA_OUTPUT_ALLOCATOR_CPU_INDEX) break;
                }
        }
        std::fprintf(stderr,"OUTPUT_ALLOCATOR cpu=%d spin=%d\n",cpu,FHERMA_OUTPUT_ALLOCATOR_SPIN);
#endif
        worker_=std::thread([this,cpu] {run(cpu);});
    }
    ~HostOutputAllocator() {
        finish_noexcept();
#if FHERMA_OUTPUT_ALLOCATOR_SPIN
        spin_stop_.store(true,std::memory_order_release);
#else
        {std::lock_guard<std::mutex> lock(mutex_);stop_=true;ready_.notify_one();}
#endif
        worker_.join();
    }
    class Work {
        HostOutputAllocator& owner_;
    public:
        Work(HostOutputAllocator& owner,std::vector<uint32_t>& destination,size_t words):owner_(owner) {owner_.begin(destination,words);}
        Work(const Work&)=delete;
        Work& operator=(const Work&)=delete;
        ~Work() {owner_.finish_noexcept();}
        void wait() {owner_.finish();}
    };
};
