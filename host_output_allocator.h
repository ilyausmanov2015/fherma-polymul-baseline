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
#ifdef __linux__
#include <sched.h>
#endif

// One ordinary output vector is allocated/zeroed per run, in parallel with input
// work. No result storage is preallocated or retained between calls.
class HostOutputAllocator {
    std::mutex mutex_;
    std::condition_variable ready_,done_;
    std::vector<uint32_t>* destination_=nullptr;
    size_t words_=0;
    bool busy_=false,complete_=false,stop_=false;
    std::exception_ptr error_;
    std::thread worker_;
    void run(int cpu) {
#ifdef __linux__
        if(cpu>=0) {
            cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);
            sched_setaffinity(0,sizeof(mask),&mask);
        }
#else
        (void)cpu;
#endif
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;) {
            ready_.wait(lock,[&] {return stop_ || destination_!=nullptr;});
            if(stop_) return;
            auto* destination=destination_;auto words=words_;
            lock.unlock();std::exception_ptr error;
            try {destination->resize(words);} catch(...) {error=std::current_exception();}
            lock.lock();destination_=nullptr;error_=error;complete_=true;done_.notify_one();
        }
    }
    void begin(std::vector<uint32_t>& destination,size_t words) {
        std::lock_guard<std::mutex> lock(mutex_);assert(!busy_);
        busy_=true;complete_=false;error_=nullptr;destination_=&destination;words_=words;
        ready_.notify_one();
    }
    void finish() {
        std::unique_lock<std::mutex> lock(mutex_);
        if(!busy_) return;
        done_.wait(lock,[&] {return complete_;});
        busy_=false;auto error=error_;error_=nullptr;lock.unlock();
        if(error) std::rethrow_exception(error);
    }
    void finish_noexcept() noexcept {try {finish();} catch(...) {}}
public:
    HostOutputAllocator() {
        int cpu=-1;
#ifdef __linux__
        cpu_set_t allowed;
        if(sched_getaffinity(0,sizeof(allowed),&allowed)==0)
            for(int candidate=0;candidate<CPU_SETSIZE;++candidate)
                if(CPU_ISSET(candidate,&allowed)) cpu=candidate;
        std::fprintf(stderr,"OUTPUT_ALLOCATOR cpu=%d\n",cpu);
#endif
        worker_=std::thread([this,cpu] {run(cpu);});
    }
    ~HostOutputAllocator() {
        finish_noexcept();
        {std::lock_guard<std::mutex> lock(mutex_);stop_=true;ready_.notify_one();}
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
