#include "input_pipeline.h"
#include <vector>
#include <stdexcept>
#include <cstdio>
struct Transfer {size_t begin,count;const uint32_t *a,*b;};
template<unsigned Parts> void exercise(HostCopyPool& pool,size_t words) {
    std::vector<uint32_t> a(words),b(words),pinned(2*words+32,0xaced1234u),out_a(words),out_b(words);
    for(size_t i=0;i<words;++i) {a[i]=uint32_t(i*7793+Parts);b[i]=uint32_t(i*107-Parts);}
    std::vector<Transfer> pending;
    copy_input_pipeline<Parts>(pool,pinned.data()+16,a.data(),b.data(),words,
        [&](size_t begin,const uint32_t* pa,const uint32_t* pb,size_t count) {
            for(size_t j=0;j<count;++j) if(pa[j]!=a[begin+j] || pb[j]!=b[begin+j])
                throw std::runtime_error("transfer enqueued before data ready");
            pending.push_back({begin,count,pa,pb});
        });
    // Complete transfers only after all chunks were packed: catches premature
    // staging reuse, including odd counts and more chunks than input words.
    size_t covered=0;
    for(auto t:pending) {
        if(t.begin!=covered) throw std::runtime_error("gap or overlap");
        std::memcpy(out_a.data()+t.begin,t.a,t.count*4);std::memcpy(out_b.data()+t.begin,t.b,t.count*4);
        covered+=t.count;
    }
    if(covered!=words || out_a!=a || out_b!=b) throw std::runtime_error("delayed transfer mismatch");
    for(size_t i=0;i<16;++i) if(pinned[i]!=0xaced1234u || pinned[2*words+16+i]!=0xaced1234u)
        throw std::runtime_error("staging guard overwritten");
}
int main() {
    HostCopyPool pool;
    for(size_t words:{0,1,2,7,63,127,4097,65537,917504}) {
        exercise<1>(pool,words);exercise<2>(pool,words);exercise<4>(pool,words);exercise<8>(pool,words);
    }
    struct Injected {};
    for(unsigned failed=0;failed<4;++failed) {
        size_t words=65537;std::vector<uint32_t> a(words,17),b(words,23),pinned(2*words);
        unsigned seen=0;bool caught=false;
        try {
            copy_input_pipeline<4>(pool,pinned.data(),a.data(),b.data(),words,
                [&](size_t,const uint32_t*,const uint32_t*,size_t) {if(seen++==failed) throw Injected{};});
        } catch(const Injected&) {caught=true;}
        if(!caught) throw std::runtime_error("expected injected transfer failure");
        // Reusing the pool immediately checks that any in-flight packing job
        // was joined before the transfer exception escaped.
        exercise<4>(pool,words);
    }
    std::puts("Input pipeline: 36 immediate/deferred transfer and ownership cases passed");
    std::puts("Input pipeline: 4 transfer exceptions drained outstanding packing jobs");
}
