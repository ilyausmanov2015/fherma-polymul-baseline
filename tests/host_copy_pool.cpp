// Exercise generation handoff, input/output jobs, odd lengths and guard bytes.
#include "host_copy_pool.h"
#include <vector>
#include <cstdio>
int main() {
    HostCopyPool pool;
    const size_t sizes[]={0,1,63,127,4097,65537,1048593};
    for(unsigned round=0;round<50;++round) for(auto bytes:sizes) {
        std::vector<unsigned char> a(bytes+128),b(bytes+128),got(2*bytes+128,0xa5);
        for(size_t i=0;i<a.size();++i) { a[i]=(i*7919+round); b[i]=(i*3571-round); }
        auto expected=got;
        std::memcpy(expected.data()+31,a.data()+15,bytes);
        std::memcpy(expected.data()+31+bytes,b.data()+1,bytes);
        pool.inputs(got.data()+31,a.data()+15,b.data()+1,bytes);
        if(got!=expected) { std::fprintf(stderr,"inputs mismatch round=%u bytes=%zu\n",round,bytes); return 1; }
        std::vector<unsigned char> out(2*bytes+128,0xb3),reference=out;
        pool.prefault(out.data()+7,2*bytes);
        for(size_t i=0;i<out.size();++i) if((i<7 || i>=7+2*bytes) && out[i]!=0xb3) {
            std::fprintf(stderr,"prefault guard mismatch round=%u bytes=%zu\n",round,bytes); return 1;
        }
        std::memcpy(reference.data()+7,expected.data()+31,2*bytes);
        pool.output(out.data()+7,got.data()+31,2*bytes);
        if(out!=reference) { std::fprintf(stderr,"output mismatch round=%u bytes=%zu\n",round,bytes); return 1; }
    }
    std::puts("Host pool: 1050 sequential copy/prefault jobs passed");
}
