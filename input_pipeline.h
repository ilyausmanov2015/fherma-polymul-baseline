#pragma once
#include "host_copy_pool.h"
#include <cstdint>
#ifndef FHERMA_ASYNC_INPUT
#define FHERMA_ASYNC_INPUT 0
#endif
#ifndef FHERMA_INPUT_WORKER_PIPELINE
#define FHERMA_INPUT_WORKER_PIPELINE 0
#endif
// Each input segment has its own pinned storage until the caller synchronizes
// all enqueued transfers. No segment is overwritten by a subsequent segment.
template<unsigned Parts,class Enqueue>
void copy_input_pipeline(HostCopyPool& pool,uint32_t* pinned,const uint32_t* a,const uint32_t* b,
                         size_t words,Enqueue&& enqueue) {
    static_assert(Parts>0,"at least one input segment");
#if FHERMA_INPUT_WORKER_PIPELINE
    pool.input_pipeline<Parts>(pinned,a,b,words,enqueue);
#elif FHERMA_ASYNC_INPUT
    struct Segment {size_t begin,count;};
    std::array<Segment,Parts> segments{};unsigned used=0;
    for(unsigned part=0;part<Parts;++part) {
        size_t begin=words*part/Parts,end=words*(part+1)/Parts;
        if(end>begin) segments[used++]={begin,end-begin};
    }
    if(!used) return;
    auto first=segments[0];
    pool.inputs(pinned+2*first.begin,a+first.begin,b+first.begin,first.count*4);
    for(unsigned part=0;part<used;++part) {
        bool next=part+1<used;
        if(next) {
            auto s=segments[part+1];
            pool.begin_inputs(pinned+2*s.begin,a+s.begin,b+s.begin,s.count*4);
        }
        try {
            auto s=segments[part];uint32_t* staged=pinned+2*s.begin;
            enqueue(s.begin,staged,staged+s.count,s.count);
        } catch(...) {
            // The caller may release the original inputs after the exception.
            if(next) pool.finish_inputs();
            throw;
        }
        if(next) pool.finish_inputs();
    }
#else
    for(unsigned part=0;part<Parts;++part) {
        size_t begin=words*part/Parts,end=words*(part+1)/Parts,count=end-begin;
        if(count==0) continue;
        uint32_t* staged_a=pinned+2*begin;
        uint32_t* staged_b=staged_a+count;
        pool.inputs(staged_a,a+begin,b+begin,count*sizeof(uint32_t));
        enqueue(begin,staged_a,staged_b,count);
    }
#endif
}
