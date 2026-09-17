#pragma once
// TEST ONLY. Runs the unchanged CUDA kernel bodies sequentially with GMP.
// This checks indexing and algebra; it does NOT validate GPU compilation,
// concurrency, or the cuPQC implementation.
#include <gmpxx.h>
#include <cstdlib>
#include <cstring>
#ifndef FHERMA_TPI
#define FHERMA_TPI 1
#endif
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
            for(threadIdx.x=0;threadIdx.x<threads.x;threadIdx.x+=FHERMA_TPI) f();
}
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
inline const char* cudaGetErrorString(int) { return "host emulation error"; }
inline int cudaMalloc(uint32_t** p,size_t n) { *p=static_cast<uint32_t*>(std::calloc(1,n)); return *p?0:1; }
inline int cudaFree(void* p) { std::free(p); return 0; }
inline int cudaMallocHost(void** p,size_t n) { *p=std::calloc(1,n); return *p?0:1; }
inline int cudaFreeHost(void* p) { std::free(p); return 0; }
inline int cudaMemcpy(void* dst,const void* src,size_t n,int) { std::memcpy(dst,src,n); return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline int cudaGetLastError() { return 0; }
namespace cupqc {
template<unsigned N> struct BitWidth {};
template<unsigned N> struct SM {};
struct Thread {};
struct Warp {};
template<unsigned N> struct TPI {};
struct EmulatedWide;
struct EmulatedBig {
    uint32_t limbs[28]{};
    explicit EmulatedBig(uint32_t x) { limbs[0]=x; }
    EmulatedBig(const uint32_t* p,unsigned i) { std::memcpy(limbs,p+i*28,112); }
    explicit EmulatedBig(mpz_class x) {
        x &= mask(); mpz_export(limbs,nullptr,-1,4,0,0,x.get_mpz_t());
    }
    mpz_class value() const {
        mpz_class x; mpz_import(x.get_mpz_t(),28,-1,4,0,0,limbs); return x;
    }
    static mpz_class mask() { return (mpz_class(1)<<896)-1; }
    const uint32_t& operator[](unsigned k) const { return limbs[k]; }
    uint32_t& operator[](unsigned k) { return limbs[k]; }
    EmulatedBig operator<<(unsigned n) const { return EmulatedBig(mpz_class((value()<<n)&mask())); }
    EmulatedBig operator>>(unsigned n) const { return EmulatedBig(mpz_class(value()>>n)); }
    EmulatedBig operator|(const EmulatedBig& b) const { return EmulatedBig(mpz_class(value()|b.value())); }
    EmulatedBig operator+(const EmulatedBig& b) const { return EmulatedBig(mpz_class((value()+b.value())&mask())); }
    EmulatedBig operator-(const EmulatedBig& b) const { return EmulatedBig(mpz_class((value()-b.value())&mask())); }
    bool operator>=(const EmulatedBig& b) const { return value()>=b.value(); }
    EmulatedBig mul_scalar(uint32_t b) const { return EmulatedBig(mpz_class((value()*b)&mask())); }
    EmulatedWide mul_wide(const EmulatedBig& b) const;
    static const mpz_class& rinv(const EmulatedBig& q) {
        static mpz_class previous, inverse;
        if(previous!=q.value()) {
            previous=q.value(); mpz_class r=mpz_class(1)<<896;
            mpz_invert(inverse.get_mpz_t(),r.get_mpz_t(),q.value().get_mpz_t());
        }
        return inverse;
    }
    EmulatedBig to_montgomery(const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((value()<<896)%q.value()));
    }
    EmulatedBig from_montgomery(const EmulatedBig& q) const {
        return EmulatedBig(mpz_class(value()*rinv(q)%q.value()));
    }
    EmulatedBig mul_montgomery(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class(value()*b.value()*rinv(q)%q.value()));
    }
    void store(uint32_t* p,unsigned i) const {
        std::memcpy(p+i*28,limbs,112);
    }
    EmulatedBig mul_mod(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((value()*b.value())%q.value()));
    }
    EmulatedBig add_mod(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((value()+b.value())%q.value()));
    }
    EmulatedBig sub_mod(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((value()-b.value()+q.value())%q.value()));
    }
};
struct EmulatedWide { EmulatedBig lo,hi; };
inline EmulatedWide EmulatedBig::mul_wide(const EmulatedBig& b) const {
    mpz_class p=value()*b.value();
    return {EmulatedBig(mpz_class(p&mask())),EmulatedBig(mpz_class(p>>896))};
}
struct Descriptor { using bigint=EmulatedBig; using modulus=EmulatedBig; };
template<unsigned W,unsigned S> Descriptor operator+(BitWidth<W>,SM<S>){return {};}
inline Descriptor operator+(Descriptor,Thread){return {};}
inline Descriptor operator+(Descriptor,Warp){return {};}
template<unsigned N> Descriptor operator+(Descriptor,TPI<N>){return {};}
}
