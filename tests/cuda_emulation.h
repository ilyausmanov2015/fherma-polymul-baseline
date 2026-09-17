#pragma once
// TEST ONLY. Runs the unchanged CUDA kernel bodies sequentially with GMP.
// This checks indexing and algebra; it does NOT validate GPU compilation,
// concurrency, or the cuPQC implementation.
#include <gmpxx.h>
#include <cstdlib>
#include <cstring>
#define __global__
#define __device__
struct dim3 { unsigned x,y,z; dim3(unsigned x_=1,unsigned y_=1,unsigned z_=1):x(x_),y(y_),z(z_){} };
inline dim3 blockIdx,blockDim,threadIdx;
inline unsigned __brev(unsigned x) {
    unsigned y=0; for(unsigned i=0;i<32;++i) { y=(y<<1)|(x&1); x>>=1; } return y;
}
template<class F> void emulate_launch(dim3 grid,dim3 threads,F f) {
    blockDim=threads;
    for(blockIdx.y=0;blockIdx.y<grid.y;++blockIdx.y)
        for(blockIdx.x=0;blockIdx.x<grid.x;++blockIdx.x)
            for(threadIdx.x=0;threadIdx.x<threads.x;++threadIdx.x) f();
}
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
inline const char* cudaGetErrorString(int) { return "host emulation error"; }
inline int cudaMalloc(uint32_t** p,size_t n) { *p=static_cast<uint32_t*>(std::calloc(1,n)); return *p?0:1; }
inline int cudaFree(void* p) { std::free(p); return 0; }
inline int cudaMemcpy(void* dst,const void* src,size_t n,int) { std::memcpy(dst,src,n); return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline int cudaGetLastError() { return 0; }
namespace cupqc {
template<unsigned N> struct BitWidth {};
template<unsigned N> struct SM {};
struct Thread {};
struct EmulatedWide;
struct EmulatedBig {
    mpz_class v;
    explicit EmulatedBig(uint32_t x):v(x){}
    EmulatedBig(const uint32_t* p,unsigned i) { mpz_import(v.get_mpz_t(),28,-1,4,0,0,p+i*28); }
    explicit EmulatedBig(mpz_class x):v(std::move(x)){}
    static mpz_class mask() { return (mpz_class(1)<<896)-1; }
    uint32_t operator[](unsigned k) const { mpz_class x=v>>(k*32); return uint32_t(x.get_ui()); }
    EmulatedBig operator<<(unsigned n) const { return EmulatedBig(mpz_class((v<<n)&mask())); }
    EmulatedBig operator>>(unsigned n) const { return EmulatedBig(mpz_class(v>>n)); }
    EmulatedBig operator|(const EmulatedBig& b) const { return EmulatedBig(mpz_class(v|b.v)); }
    EmulatedBig operator+(const EmulatedBig& b) const { return EmulatedBig(mpz_class((v+b.v)&mask())); }
    EmulatedBig operator-(const EmulatedBig& b) const { return EmulatedBig(mpz_class((v-b.v)&mask())); }
    bool operator>=(const EmulatedBig& b) const { return v>=b.v; }
    EmulatedBig mul_scalar(uint32_t b) const { return EmulatedBig(mpz_class((v*b)&mask())); }
    EmulatedWide mul_wide(const EmulatedBig& b) const;
    static const mpz_class& rinv(const EmulatedBig& q) {
        static mpz_class previous, inverse;
        if(previous!=q.v) {
            previous=q.v; mpz_class r=mpz_class(1)<<896;
            mpz_invert(inverse.get_mpz_t(),r.get_mpz_t(),q.v.get_mpz_t());
        }
        return inverse;
    }
    EmulatedBig to_montgomery(const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((v<<896)%q.v));
    }
    EmulatedBig from_montgomery(const EmulatedBig& q) const {
        return EmulatedBig(mpz_class(v*rinv(q)%q.v));
    }
    EmulatedBig mul_montgomery(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class(v*b.v*rinv(q)%q.v));
    }
    void store(uint32_t* p,unsigned i) const {
        std::memset(p+i*28,0,112); mpz_export(p+i*28,nullptr,-1,4,0,0,v.get_mpz_t());
    }
    EmulatedBig mul_mod(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((v*b.v)%q.v));
    }
    EmulatedBig add_mod(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((v+b.v)%q.v));
    }
    EmulatedBig sub_mod(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((v-b.v+q.v)%q.v));
    }
};
struct EmulatedWide { EmulatedBig lo,hi; };
inline EmulatedWide EmulatedBig::mul_wide(const EmulatedBig& b) const {
    mpz_class p=v*b.v;
    return {EmulatedBig(mpz_class(p&mask())),EmulatedBig(mpz_class(p>>896))};
}
struct Descriptor { using bigint=EmulatedBig; using modulus=EmulatedBig; };
template<unsigned W,unsigned S> Descriptor operator+(BitWidth<W>,SM<S>){return {};}
inline Descriptor operator+(Descriptor,Thread){return {};}
}
