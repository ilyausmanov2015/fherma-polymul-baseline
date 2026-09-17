#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define FHERMA_ACC_DEVICE __device__ __forceinline__
#else
#define FHERMA_ACC_DEVICE inline
#endif

namespace quartic {
// Independent limb products; propagate carries only once after the dot product.
// For at most 16 uint32 factors, each high sum fits in 36 bits.
template<unsigned Words> struct LimbAccumulator {
    uint32_t low[Words]{};
    uint64_t high[Words]{};
    FHERMA_ACC_DEVICE void add(const uint32_t* basis,uint32_t factor) {
        #pragma unroll
        for(unsigned word=0;word<Words;++word) {
            uint64_t product=uint64_t(basis[word])*factor;
            uint32_t previous=low[word],next=previous+uint32_t(product);
            low[word]=next;
            high[word]+=(product>>32)+uint64_t(next<previous);
        }
    }
    template<class Output> FHERMA_ACC_DEVICE uint64_t store(Output& output) const {
        uint64_t carry=0;
        #pragma unroll
        for(unsigned word=0;word<Words;++word) {
            uint64_t value=uint64_t(low[word])+carry;
            output[word]=uint32_t(value);
            carry=high[word]+(value>>32);
        }
        return carry;
    }
};
}
#undef FHERMA_ACC_DEVICE
