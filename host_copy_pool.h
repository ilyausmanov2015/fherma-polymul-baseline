#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <array>
#include <cstdio>
#ifdef __linux__
#include <sched.h>
#include <fstream>
#endif
#include "host_stream_copy.h"
#ifndef FHERMA_COPY_THREADS
#define FHERMA_COPY_THREADS 4
#endif

// Sleeping workers plus the caller. Input-dependent copying remains
// entirely within run(); setup creates only the persistent worker threads.
class HostCopyPool {
    static constexpr unsigned Threads=FHERMA_COPY_THREADS;
    static_assert(Threads>=2 && Threads%2==0,"even copy thread count required");
    struct Job { const char *a=nullptr,*b=nullptr; char* out=nullptr; size_t bytes=0; } job_;
    std::mutex mutex_;
    std::condition_variable start_,done_;
    std::array<std::thread,Threads-1> workers_;
    unsigned generation_=0,pending_=0;
    bool stop_=false;
    static void part(const Job& job,unsigned rank) {
        if(job.b) {
            unsigned half=rank%(Threads/2);
            size_t begin=job.bytes*half/(Threads/2),end=job.bytes*(half+1)/(Threads/2);
            const char* source=rank<Threads/2 ? job.a : job.b;
            char* dest=job.out+(rank<Threads/2 ? 0 : job.bytes);
            host_copy_bytes(dest+begin,source+begin,end-begin);
        } else {
            size_t begin=job.bytes*rank/Threads,end=job.bytes*(rank+1)/Threads;
            host_copy_bytes(job.out+begin,job.a+begin,end-begin);
        }
    }
    void worker(unsigned rank) {
        unsigned seen=0;
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;) {
            start_.wait(lock,[&] { return stop_ || generation_!=seen; });
            if(stop_) return;
            auto job=job_; seen=generation_;
            lock.unlock(); part(job,rank); lock.lock();
            if(--pending_==0) done_.notify_one();
        }
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_=true; }
        start_.notify_all();
        for(auto& thread:workers_) if(thread.joinable()) thread.join();
    }
    void run(Job job) {
        { std::lock_guard<std::mutex> lock(mutex_); job_=job; pending_=Threads-1; ++generation_; }
        start_.notify_all(); part(job,Threads-1);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock,[&] { return pending_==0; });
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
    void output(void* out,const void* source,size_t bytes) {
        run({static_cast<const char*>(source),nullptr,static_cast<char*>(out),bytes});
    }
};
