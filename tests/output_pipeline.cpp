#include "host_copy_pool.h"
#include <vector>
#include <stdexcept>
#include <algorithm>

template<unsigned Parts> void exercise(HostCopyPool& pool,size_t words) {
    constexpr uint32_t guard=0xa5a5a5a5u;
    std::vector<uint32_t> source(words+2,0),output(words+2,guard);
    pool.output_pipeline<Parts>(output.data()+1,source.data()+1,words,[&](unsigned part) {
        size_t begin=words*part/Parts,end=words*(part+1)/Parts;
        for(size_t i=begin;i<end;++i) source[i+1]=uint32_t(i*7919+part*17);
        if(part%2) std::this_thread::yield();
    });
    assert(output.front()==guard && output.back()==guard);
    assert(std::equal(source.begin()+1,source.end()-1,output.begin()+1));
    for(unsigned fail=0;fail<Parts;++fail) {
        std::fill(output.begin(),output.end(),guard);
        bool caught=false;
        try {
            pool.output_pipeline<Parts>(output.data()+1,source.data()+1,words,[&](unsigned part) {
                if(part==fail) throw std::runtime_error("DMA failure");
            });
        } catch(const std::runtime_error&) {caught=true;}
        assert(caught);
        // No worker may consume a segment whose producer failed to publish it.
        for(size_t i=words*fail/Parts;i<words;++i) assert(output[i+1]==guard);
        assert(output.front()==guard && output.back()==guard);
        pool.output(output.data()+1,source.data()+1,words*4);
        assert(std::equal(source.begin()+1,source.end()-1,output.begin()+1));
    }
}
int main() {
    HostCopyPool pool;
    for(auto words:{size_t(0),size_t(1),size_t(17),size_t(4099),size_t(917504)}) {
        exercise<1>(pool,words);exercise<4>(pool,words);
        exercise<8>(pool,words);exercise<16>(pool,words);
    }
    std::puts("20 output pipelines, 145 producer failures, untouched future segments and pool reuse passed");
}
