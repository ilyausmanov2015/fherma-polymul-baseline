#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include "host_stream_copy.h"

// Three sleeping workers plus the caller. Input-dependent copying remains
// entirely within run(); setup creates only the persistent worker threads.
class HostCopyPool {
    struct Job { const char *a=nullptr,*b=nullptr; char* out=nullptr; size_t bytes=0; } job_;
    std::mutex mutex_;
    std::condition_variable start_,done_;
    std::thread workers_[3];
    unsigned generation_=0,pending_=0;
    bool stop_=false;
    static void part(const Job& job,unsigned rank) {
        if(job.b) {
            unsigned half=rank%2; size_t begin=job.bytes*half/2,end=job.bytes*(half+1)/2;
            const char* source=rank<2 ? job.a : job.b;
            char* dest=job.out+(rank<2 ? 0 : job.bytes);
            host_copy_bytes(dest+begin,source+begin,end-begin);
        } else {
            size_t begin=job.bytes*rank/4,end=job.bytes*(rank+1)/4;
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
        { std::lock_guard<std::mutex> lock(mutex_); job_=job; pending_=3; ++generation_; }
        start_.notify_all(); part(job,3);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock,[&] { return pending_==0; });
        job_={};
    }
public:
    HostCopyPool() {
        try { for(unsigned i=0;i<3;++i) workers_[i]=std::thread([this,i] { worker(i); }); }
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
