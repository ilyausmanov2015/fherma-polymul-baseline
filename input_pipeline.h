#pragma once
#include "host_copy_pool.h"
#include <cstdint>
// Each input segment has its own pinned storage until the caller synchronizes
// all enqueued transfers. No segment is overwritten by a subsequent segment.
template<unsigned Parts,class Enqueue>
void copy_input_pipeline(HostCopyPool& pool,uint32_t* pinned,const uint32_t* a,const uint32_t* b,
                         size_t words,Enqueue&& enqueue) {
    static_assert(Parts>0,"at least one input segment");
    for(unsigned part=0;part<Parts;++part) {
        size_t begin=words*part/Parts,end=words*(part+1)/Parts,count=end-begin;
        if(count==0) continue;
        uint32_t* staged_a=pinned+2*begin;
        uint32_t* staged_b=staged_a+count;
        pool.inputs(staged_a,a+begin,b+begin,count*sizeof(uint32_t));
        enqueue(begin,staged_a,staged_b,count);
    }
}
