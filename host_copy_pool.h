#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <array>
#include <atomic>
#include <cstdio>
#ifdef __linux__
#include <sched.h>
#include <fstream>
#endif
#include "host_stream_copy.h"
#ifndef FHERMA_COPY_THREADS
#define FHERMA_COPY_THREADS 4
#endif
#ifndef FHERMA_OUTPUT_THREADS
#define FHERMA_OUTPUT_THREADS FHERMA_COPY_THREADS
#endif

#ifndef FHERMA_STREAM_OUTPUT
#define FHERMA_STREAM_OUTPUT 1
#endif

#ifndef FHERMA_SPIN_COPY
#define FHERMA_SPIN_COPY 0
#endif
#ifndef FHERMA_COPY_ACKS
#define FHERMA_COPY_ACKS 0
#endif
// Persistent workers plus the caller. Input-dependent copying remains
// entirely within run(); setup creates only the persistent worker threads.
class HostCopyPool {
    static constexpr unsigned Threads=FHERMA_COPY_THREADS;
    static constexpr unsigned OutputThreads=FHERMA_OUTPUT_THREADS;
    static_assert(Threads>=2 && Threads%2==0,"even copy thread count required");
    static_assert(OutputThreads>=1 && OutputThreads<=Threads,"output workers must fit the pool");
    struct Job { const char *a=nullptr,*b=nullptr; char* out=nullptr; size_t bytes=0; bool prefault=false; } job_;
    std::mutex mutex_;
    std::condition_variable start_,done_;
    std::array<std::thread,Threads-1> workers_;
    unsigned generation_=0,pending_=0;
    bool stop_=false;
#if FHERMA_SPIN_COPY
    alignas(64) std::atomic<unsigned> spin_generation_{0};
    alignas(64) std::atomic<unsigned> spin_pending_{0};
    struct alignas(64) Completion { std::atomic<unsigned> generation{0}; };
    std::array<Completion,Threads-1> completed_;
    alignas(64) std::atomic<bool> spin_stop_{false};
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
    static void part(const Job& job,unsigned rank) {
        if(!job.b && rank>=OutputThreads) return;
        if(job.prefault) {
            size_t begin=job.bytes*rank/OutputThreads,end=job.bytes*(rank+1)/OutputThreads;
            for(size_t i=begin;i<end;i+=4096) job.out[i]=0;
        } else if(job.b) {
            unsigned half=rank%(Threads/2);
            size_t begin=job.bytes*half/(Threads/2),end=job.bytes*(half+1)/(Threads/2);
            const char* source=rank<Threads/2 ? job.a : job.b;
            char* dest=job.out+(rank<Threads/2 ? 0 : job.bytes);
            host_copy_bytes(dest+begin,source+begin,end-begin);
#if FHERMA_INPUT_WC && defined(__x86_64__) && defined(__GNUC__)
            // Also order short memcpy tails written to WC staging pages.
            _mm_sfence();
#endif
        } else {
            size_t begin=job.bytes*rank/OutputThreads,end=job.bytes*(rank+1)/OutputThreads;
#if FHERMA_STREAM_OUTPUT
            host_copy_bytes(job.out+begin,job.a+begin,end-begin);
#else
            std::memcpy(job.out+begin,job.a+begin,end-begin);
#endif
        }
    }
    void worker(unsigned rank) {
        unsigned seen=0;
#if FHERMA_SPIN_COPY
        while(!spin_stop_.load(std::memory_order_relaxed)) {
            unsigned generation=spin_generation_.load(std::memory_order_acquire);
            if(generation==seen) { pause(); continue; }
            auto job=job_; seen=generation;
            part(job,rank);
#if FHERMA_COPY_ACKS
            completed_[rank].generation.store(generation,std::memory_order_release);
#else
            spin_pending_.fetch_sub(1,std::memory_order_release);
#endif
        }
#else
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;) {
            start_.wait(lock,[&] { return stop_ || generation_!=seen; });
            if(stop_) return;
            auto job=job_; seen=generation_;
            lock.unlock(); part(job,rank); lock.lock();
            if(--pending_==0) done_.notify_one();
        }
#endif
    }
    void stop() {
#if FHERMA_SPIN_COPY
        spin_stop_.store(true,std::memory_order_relaxed);
#else
        { std::lock_guard<std::mutex> lock(mutex_); stop_=true; }
        start_.notify_all();
#endif
        for(auto& thread:workers_) if(thread.joinable()) thread.join();
    }
    void run(Job job) {
#if FHERMA_SPIN_COPY
        job_=job;
#if FHERMA_COPY_ACKS
        unsigned generation=spin_generation_.fetch_add(1,std::memory_order_release)+1;
        part(job,Threads-1);
        for(auto& worker:completed_)
            while(worker.generation.load(std::memory_order_acquire)!=generation) pause();
#else
        spin_pending_.store(Threads-1,std::memory_order_relaxed);
        spin_generation_.fetch_add(1,std::memory_order_release);
        part(job,Threads-1);
        while(spin_pending_.load(std::memory_order_acquire)) pause();
#endif
#else
        { std::lock_guard<std::mutex> lock(mutex_); job_=job; pending_=Threads-1; ++generation_; }
        start_.notify_all(); part(job,Threads-1);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock,[&] { return pending_==0; });
#endif
        job_={};
    }
public:
    HostCopyPool() {
        std::array<int,Threads-1> worker_cpus; worker_cpus.fill(-1);
#ifdef __linux__
        cpu_set_t allowed; int caller=sched_getcpu();
        if(caller>=0 && sched_getaffinity(0,sizeof(allowed),&allowed)==0) {
            unsigned found=0;
            for(int cpu=0;cpu<CPU_SETSIZE && found<Threads-1;++cpu)
                if(cpu!=caller && CPU_ISSET(cpu,&allowed)) worker_cpus[found++]=cpu;
            cpu_set_t current; CPU_ZERO(&current); CPU_SET(caller,&current);
            sched_setaffinity(0,sizeof(current),&current);
        }
        std::string quota,period;
        std::ifstream("/sys/fs/cgroup/cpu.max")>>quota>>period;
        std::fprintf(stderr,"COPY_POOL threads=%u main_cpu=%d worker0=%d cpu_max=%s/%s\n",
                     Threads,caller,worker_cpus[0],quota.c_str(),period.c_str());
#endif
        try { for(unsigned i=0;i<Threads-1;++i) workers_[i]=std::thread([this,i,cpu=worker_cpus[i]] {
#ifdef __linux__
            if(cpu>=0) { cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu,&mask); sched_setaffinity(0,sizeof(mask),&mask); }
#endif
            worker(i);
        }); }
        catch(...) { stop(); throw; }
    }
    ~HostCopyPool() { stop(); }
    void inputs(void* out,const void* a,const void* b,size_t bytes) {
        run({static_cast<const char*>(a),static_cast<const char*>(b),static_cast<char*>(out),bytes});
    }
    void prefault(void* storage,size_t bytes) {
        run({nullptr,nullptr,static_cast<char*>(storage),bytes,true});
    }
    void output(void* out,const void* source,size_t bytes) {
        run({static_cast<const char*>(source),nullptr,static_cast<char*>(out),bytes});
    }
};
