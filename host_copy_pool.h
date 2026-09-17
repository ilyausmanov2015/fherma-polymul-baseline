#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <array>
#include <atomic>
#include <cstdio>
#include <cassert>
#ifdef __linux__
#include <sched.h>
#include <fstream>
#endif
#include "host_stream_copy.h"
#include "host_cpu_topology.h"
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
#ifndef FHERMA_MAIN_OUTPUT
#define FHERMA_MAIN_OUTPUT 0
#endif
#ifndef FHERMA_FIRST_CPU
#define FHERMA_FIRST_CPU 0
#endif
#ifndef FHERMA_DEFER_INPUT_CALLER
#define FHERMA_DEFER_INPUT_CALLER 0
#endif
#ifndef FHERMA_CACHED_VECTOR
#define FHERMA_CACHED_VECTOR 0
#endif
#ifndef FHERMA_DEFER_MAIN_PIN
#define FHERMA_DEFER_MAIN_PIN 0
#endif
#ifndef FHERMA_INPUT_WORKERS_ONLY
#define FHERMA_INPUT_WORKERS_ONLY 0
#endif
#ifndef FHERMA_SELECTIVE_OUTPUT_WAKE
#define FHERMA_SELECTIVE_OUTPUT_WAKE 0
#endif
#ifndef FHERMA_FLUSH_OUTPUT_SOURCE
#define FHERMA_FLUSH_OUTPUT_SOURCE 0
#endif
#ifndef FHERMA_UMWAIT
#define FHERMA_UMWAIT 0
#endif
#ifndef FHERMA_PHYSICAL_CORES
#define FHERMA_PHYSICAL_CORES 0
#endif
#ifndef FHERMA_INTERLEAVED_INPUT
#define FHERMA_INTERLEAVED_INPUT 0
#endif
// Persistent workers plus the caller. Input-dependent copying remains
// entirely within run(); setup creates only the persistent worker threads.
class HostCopyPool {
    static constexpr unsigned Threads=FHERMA_COPY_THREADS;
    static constexpr unsigned OutputThreads=FHERMA_OUTPUT_THREADS;
    static constexpr unsigned Workers=FHERMA_INPUT_WORKERS_ONLY ? Threads : Threads-1;
    static_assert(!FHERMA_INPUT_WORKERS_ONLY || Threads<24,"leave an allowed CPU for the caller");
    static_assert(Threads>=2 && Threads%2==0,"even copy thread count required");
    static_assert(OutputThreads>=1 && OutputThreads<=Threads,"output workers must fit the pool");
    struct Pipeline {
        struct alignas(64) Progress {std::atomic<unsigned> chunks{0};};
        std::array<Progress,Workers> copied;
        std::atomic<unsigned> ready{0};
        std::atomic<bool> cancelled{false};
        unsigned parts=0;
    };
    struct Job { const char *a=nullptr,*b=nullptr; char* out=nullptr; size_t bytes=0; bool prefault=false; Pipeline* pipeline=nullptr; } job_;
    std::mutex mutex_;
    std::condition_variable start_,done_;
    std::array<std::thread,Workers> workers_;
    unsigned generation_=0,pending_=0;
    bool stop_=false;
    bool in_flight_=false;
    bool deferred_caller_=false;
    unsigned active_generation_=0;
    int caller_cpu_=-1;
#if FHERMA_UMWAIT && defined(__x86_64__) && defined(__GNUC__)
    bool waitpkg_=false;
    __attribute__((target("waitpkg"))) static void wait_for_generation(std::atomic<unsigned>& signal,unsigned seen) {
        _umonitor(static_cast<void*>(&signal));
        // Recheck after arming the monitor, so publication cannot be lost.
        // A short deadline also bounds shutdown and CPUs whose firmware
        // chooses to ignore monitoring. C0.1 prioritizes wakeup latency.
        if(signal.load(std::memory_order_acquire)==seen) _umwait(1,__rdtsc()+5000);
    }
#endif
#if FHERMA_SPIN_COPY
    alignas(64) std::atomic<unsigned> spin_generation_{0};
#if FHERMA_SELECTIVE_OUTPUT_WAKE
    alignas(64) std::atomic<unsigned> output_generation_{0};
    unsigned published_generation_=0;
    bool active_output_=false;
#endif
    alignas(64) std::atomic<unsigned> spin_pending_{0};
    struct alignas(64) Completion { std::atomic<unsigned> generation{0}; };
    std::array<Completion,Workers> completed_;
    alignas(64) std::atomic<bool> spin_stop_{false};
#endif
    static void pause() {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#else
        std::this_thread::yield();
#endif
    }
    static void part_plain(const Job& job,unsigned rank) {
#if FHERMA_MAIN_OUTPUT
        // Let the caller copy one output partition instead of only polling.
        if(!job.b && OutputThreads<Threads) {
            if(rank==Threads-1) rank=OutputThreads-1;
            else if(rank==OutputThreads-1) return;
        }
#endif
        if(!job.b && rank>=OutputThreads) return;
        if(job.prefault) {
            size_t begin=job.bytes*rank/OutputThreads,end=job.bytes*(rank+1)/OutputThreads;
            for(size_t i=begin;i<end;i+=4096) job.out[i]=0;
        } else if(job.b) {
            unsigned half=FHERMA_INTERLEAVED_INPUT ? rank/2 : rank%(Threads/2);
            size_t begin=job.bytes*half/(Threads/2),end=job.bytes*(half+1)/(Threads/2);
            bool first=FHERMA_INTERLEAVED_INPUT ? !(rank&1) : rank<Threads/2;
            const char* source=first ? job.a : job.b;
            char* dest=job.out+(first ? 0 : job.bytes);
            host_copy_bytes(dest+begin,source+begin,end-begin);
#if FHERMA_INPUT_WC && defined(__x86_64__) && defined(__GNUC__)
            // Also order short memcpy tails written to WC staging pages.
            _mm_sfence();
#endif
        } else {
            size_t begin=job.bytes*rank/OutputThreads,end=job.bytes*(rank+1)/OutputThreads;
#if FHERMA_OUTPUT_WC
            host_copy_wc(job.out+begin,job.a+begin,end-begin);
#elif FHERMA_STREAM_OUTPUT
            host_copy_bytes(job.out+begin,job.a+begin,end-begin);
#elif FHERMA_CACHED_VECTOR
            host_copy_cached(job.out+begin,job.a+begin,end-begin);
#else
            std::memcpy(job.out+begin,job.a+begin,end-begin);
#endif
#if FHERMA_FLUSH_OUTPUT_SOURCE
            host_discard_cached_reads(job.a+begin,end-begin);
#endif
        }
    }
    static constexpr bool output_worker(unsigned rank) {
        if(FHERMA_INPUT_WORKERS_ONLY && rank==Threads-1) return false;
        if(FHERMA_MAIN_OUTPUT && OutputThreads<Threads) return rank<OutputThreads-1;
        return rank<OutputThreads;
    }
    static Job pipeline_chunk(const Job& job,unsigned chunk) {
        size_t words=job.bytes/4;
        size_t begin=4*(words*chunk/job.pipeline->parts);
        size_t end=4*(words*(chunk+1)/job.pipeline->parts);
        if(job.b) return {job.a+begin,job.b+begin,job.out+2*begin,end-begin};
        return {job.a+begin,nullptr,job.out+begin,end-begin};
    }
    static void part(const Job& job,unsigned rank) {
        if(!job.pipeline) {part_plain(job,rank);return;}
        if(job.b) {
            for(unsigned chunk=0;chunk<job.pipeline->parts;++chunk) {
                if(job.pipeline->cancelled.load(std::memory_order_acquire)) return;
                part_plain(pipeline_chunk(job,chunk),rank);
                job.pipeline->copied[rank].chunks.store(chunk+1,std::memory_order_release);
            }
            return;
        }
        if(!output_worker(rank) && rank!=Threads-1) return;
        for(unsigned chunk=0;chunk<job.pipeline->parts;++chunk) {
            while(job.pipeline->ready.load(std::memory_order_acquire)<=chunk) {
                if(job.pipeline->cancelled.load(std::memory_order_acquire)) return;
                pause();
            }
            part_plain(pipeline_chunk(job,chunk),rank);
        }
    }
    void worker(unsigned rank) {
        unsigned seen=0;
#if FHERMA_UMWAIT && defined(__x86_64__) && defined(__GNUC__) && !FHERMA_SELECTIVE_OUTPUT_WAKE
        unsigned idle=0;
#endif
#if FHERMA_SPIN_COPY && FHERMA_SELECTIVE_OUTPUT_WAKE
        unsigned seen_output=0;
#endif
#if FHERMA_SPIN_COPY
        while(!spin_stop_.load(std::memory_order_relaxed)) {
            unsigned generation=spin_generation_.load(std::memory_order_acquire);
#if FHERMA_SELECTIVE_OUTPUT_WAKE
            if(generation!=seen) seen=generation;
            else if(output_worker(rank)) {
                generation=output_generation_.load(std::memory_order_acquire);
                if(generation==seen_output) {pause();continue;}
                seen_output=generation;
            } else {pause();continue;}
            auto job=job_;
#else
            if(generation==seen) {
#if FHERMA_UMWAIT && defined(__x86_64__) && defined(__GNUC__)
                if(waitpkg_ && idle>=64) wait_for_generation(spin_generation_,seen);
                else {++idle;pause();}
#else
                pause();
#endif
                continue;
            }
#if FHERMA_UMWAIT && defined(__x86_64__) && defined(__GNUC__)
            idle=0;
#endif
            auto job=job_; seen=generation;
#endif
            if(!FHERMA_INPUT_WORKERS_ONLY || job.b || rank<Threads-1) part(job,rank);
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
            lock.unlock();
            if(!FHERMA_INPUT_WORKERS_ONLY || job.b || rank<Threads-1) part(job,rank);
            lock.lock();
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
    void begin(Job job,bool defer_caller=false) {
        assert(!in_flight_);in_flight_=true;
        deferred_caller_=defer_caller && !(FHERMA_INPUT_WORKERS_ONLY && job.b);
#if FHERMA_SPIN_COPY
        job_=job;
#if FHERMA_SELECTIVE_OUTPUT_WAKE
        active_generation_=++published_generation_;active_output_=!job.b;
#if !FHERMA_COPY_ACKS
        unsigned pending=0;
        for(unsigned rank=0;rank<Workers;++rank) pending+=!active_output_ || output_worker(rank);
        spin_pending_.store(pending,std::memory_order_relaxed);
#endif
        auto& published=active_output_ ? output_generation_ : spin_generation_;
        published.store(active_generation_,std::memory_order_release);
#else
#if FHERMA_COPY_ACKS
        active_generation_=spin_generation_.fetch_add(1,std::memory_order_release)+1;
#else
        spin_pending_.store(Workers,std::memory_order_relaxed);
        spin_generation_.fetch_add(1,std::memory_order_release);
#endif
#endif
        if(!deferred_caller_ && !(FHERMA_INPUT_WORKERS_ONLY && job.b)) part(job,Threads-1);
#else
        { std::lock_guard<std::mutex> lock(mutex_); job_=job; pending_=Workers; ++generation_; }
        start_.notify_all();if(!deferred_caller_ && !(FHERMA_INPUT_WORKERS_ONLY && job.b)) part(job,Threads-1);
#endif
    }
    void finish() {
        assert(in_flight_);
        if(deferred_caller_) part(job_,Threads-1);
#if FHERMA_SPIN_COPY
#if FHERMA_COPY_ACKS
        for(unsigned rank=0;rank<Workers;++rank) {
#if FHERMA_SELECTIVE_OUTPUT_WAKE
            if(active_output_ && !output_worker(rank)) continue;
#endif
            while(completed_[rank].generation.load(std::memory_order_acquire)!=active_generation_) pause();
        }
#else
        while(spin_pending_.load(std::memory_order_acquire)) pause();
#endif
#else
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock,[&] { return pending_==0; });
#endif
        job_={};in_flight_=false;
    }
    void run(Job job) { begin(job);finish(); }
public:
    HostCopyPool() {
#if FHERMA_UMWAIT && defined(__x86_64__) && defined(__GNUC__)
        waitpkg_=bool(__builtin_cpu_supports("waitpkg"));
        std::fprintf(stderr,"COPY_IDLE waitpkg=%d\n",waitpkg_);
#endif
        std::array<int,Workers> worker_cpus; worker_cpus.fill(-1);
#ifdef __linux__
        cpu_set_t allowed; int caller=sched_getcpu();
        if(caller>=0 && sched_getaffinity(0,sizeof(allowed),&allowed)==0) {
#if FHERMA_HOST_PROFILE
            for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&allowed)) {
                std::string path="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
                std::string core,package,siblings;
                std::ifstream(path+"core_id")>>core;
                std::ifstream(path+"physical_package_id")>>package;
                std::ifstream(path+"thread_siblings_list")>>siblings;
                std::fprintf(stderr,"CPU_TOPOLOGY cpu=%d package=%s core=%s siblings=%s\n",
                    cpu,package.c_str(),core.c_str(),siblings.c_str());
            }
#endif
#if FHERMA_FIRST_CPU
            for(int cpu=0;cpu<CPU_SETSIZE;++cpu)
                if(CPU_ISSET(cpu,&allowed)) { caller=cpu;break; }
#endif
            unsigned found=0;
#if FHERMA_PHYSICAL_CORES
            std::vector<HostCpuCore> topology;
            for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&allowed)) {
                HostCpuCore record{cpu,-1,-1};
                std::string path="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
                std::ifstream(path+"physical_package_id")>>record.package;
                std::ifstream(path+"core_id")>>record.core;
                topology.push_back(record);
            }
            constexpr unsigned dense_prefix=FHERMA_PHYSICAL_CORES==2 ?
                (FHERMA_MAIN_OUTPUT && OutputThreads<Threads ? OutputThreads-1 : OutputThreads) : 0;
            for(int cpu:host_worker_cpu_order(topology,caller,dense_prefix)) {
                if(found==Workers) break;
                worker_cpus[found++]=cpu;
            }
#else
            for(int cpu=0;cpu<CPU_SETSIZE && found<Workers;++cpu)
                if(cpu!=caller && CPU_ISSET(cpu,&allowed)) worker_cpus[found++]=cpu;
#endif
#if FHERMA_HOST_PROFILE
            for(unsigned rank=0;rank<Workers;++rank)
                std::fprintf(stderr,"COPY_PLACEMENT rank=%u cpu=%d\n",rank,worker_cpus[rank]);
#endif
            caller_cpu_=caller;
#if !FHERMA_DEFER_MAIN_PIN
            pin_caller();
#endif
        }
        std::string quota,period;
        std::ifstream("/sys/fs/cgroup/cpu.max")>>quota>>period;
        std::fprintf(stderr,"COPY_POOL threads=%u input_threads=%u main_cpu=%d worker0=%d cpu_max=%s/%s\n",
                     Workers+1,Threads,caller,worker_cpus[0],quota.c_str(),period.c_str());
#endif
        try { for(unsigned i=0;i<Workers;++i) workers_[i]=std::thread([this,i,cpu=worker_cpus[i]] {
#ifdef __linux__
            if(cpu>=0) { cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu,&mask); sched_setaffinity(0,sizeof(mask),&mask); }
#endif
            worker(i);
        }); }
        catch(...) { stop(); throw; }
    }
    ~HostCopyPool() { stop(); }
    // Defer this until CUDA initialization has created its helper threads.
    // Workers are already individually pinned; the measured run is unchanged.
    void pin_caller() {
#ifdef __linux__
        if(caller_cpu_>=0) {
            cpu_set_t current; CPU_ZERO(&current); CPU_SET(caller_cpu_,&current);
            sched_setaffinity(0,sizeof(current),&current);
        }
#endif
    }
    void inputs(void* out,const void* a,const void* b,size_t bytes) {
        run({static_cast<const char*>(a),static_cast<const char*>(b),static_cast<char*>(out),bytes});
    }
    void begin_inputs(void* out,const void* a,const void* b,size_t bytes) {
        begin({static_cast<const char*>(a),static_cast<const char*>(b),static_cast<char*>(out),bytes},bool(FHERMA_DEFER_INPUT_CALLER));
    }
    void finish_inputs() { finish(); }
    template<unsigned Parts,class Enqueue> void input_pipeline(void* out,const void* a,const void* b,size_t words,Enqueue enqueue) {
        static_assert(Parts>0,"nonempty input pipeline");
        if(!words) return;
        Pipeline pipeline;pipeline.parts=Parts;
        Job job{static_cast<const char*>(a),static_cast<const char*>(b),static_cast<char*>(out),words*4,false,&pipeline};
        begin(job,true);deferred_caller_=false;
        try {
            for(unsigned chunk=0;chunk<Parts;++chunk) {
                auto segment=pipeline_chunk(job,chunk);
                if(!FHERMA_INPUT_WORKERS_ONLY) part_plain(segment,Threads-1);
                for(unsigned rank=0;rank<Workers;++rank)
                    while(pipeline.copied[rank].chunks.load(std::memory_order_acquire)<=chunk) pause();
                size_t begin=words*chunk/Parts,count=segment.bytes/4;
                if(count) {
                    const auto* staged=reinterpret_cast<const uint32_t*>(segment.out);
                    enqueue(begin,staged,staged+count,count);
                }
            }
        } catch(...) {
            pipeline.cancelled.store(true,std::memory_order_release);
            finish();throw;
        }
        finish();
    }
    void prefault(void* storage,size_t bytes) {
        run({nullptr,nullptr,static_cast<char*>(storage),bytes,true});
    }
    void output(void* out,const void* source,size_t bytes) {
        run({static_cast<const char*>(source),nullptr,static_cast<char*>(out),bytes});
    }
    template<unsigned Parts,class Wait> void output_pipeline(void* out,const void* source,size_t words,Wait wait_ready) {
        static_assert(Parts>0,"nonempty output pipeline");
        Pipeline pipeline;pipeline.parts=Parts;
        Job job{static_cast<const char*>(source),nullptr,static_cast<char*>(out),words*4,false,&pipeline};
        begin(job,true);
        // The caller publishes each ready DMA segment and copies its own slice.
        // Workers consume all segments under one job and acknowledge only once.
        deferred_caller_=false;
        try {
            for(unsigned chunk=0;chunk<Parts;++chunk) {
                wait_ready(chunk);
                pipeline.ready.store(chunk+1,std::memory_order_release);
                part_plain(pipeline_chunk(job,chunk),Threads-1);
            }
        } catch(...) {
            pipeline.cancelled.store(true,std::memory_order_release);
            finish();throw;
        }
        finish();
    }
};
