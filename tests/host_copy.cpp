// Byte-exact checks for unaligned heads/tails and untouched guard regions.
#define FHERMA_STREAM_COPY 1
#include "host_stream_copy.h"
#include <cstdio>
#include <vector>

int main() {
    const size_t sizes[]={0,1,31,63,64,65,127,128,4095,4096,4097,65537,1048593};
    const size_t offsets[]={0,1,15,31,63};
    for(unsigned cached=0;cached<2;++cached) for(auto n:sizes) for(auto src:offsets) for(auto dst:offsets) {
        std::vector<unsigned char> in(n+128),got(n+128,0xa5),expected=got;
        for(size_t i=0;i<in.size();++i) in[i]=static_cast<unsigned char>((i*7919)^(i>>9));
        std::memcpy(expected.data()+dst,in.data()+src,n);
        if(cached) host_copy_cached(got.data()+dst,in.data()+src,n);
        else host_copy_bytes(got.data()+dst,in.data()+src,n);
        host_discard_cached_reads(in.data()+src,n);
        if(got!=expected) {
            std::fprintf(stderr,"copy mismatch: bytes=%zu src_offset=%zu dst_offset=%zu\n",n,src,dst);
            return 1;
        }
    }
#if defined(__x86_64__) && defined(__GNUC__)
    std::printf("Host copy: 650 streaming/cached cases passed; AVX-512 available=%d\n",bool(__builtin_cpu_supports("avx512f")));
    std::printf("Cache discard: CLFLUSHOPT available=%d\n",bool(__builtin_cpu_supports("clflushopt")));
#else
    std::puts("Host copy: 650 streaming/cached cases passed; portable fallback");
#endif
}
