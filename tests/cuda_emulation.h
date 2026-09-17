#pragma once
// TEST ONLY. Runs the unchanged CUDA kernel bodies sequentially with GMP.
// This checks indexing and algebra; it does NOT validate GPU compilation,
// concurrency, or the cuPQC implementation.
#include <gmpxx.h>
#include <cstdlib>
#include <cstring>
#include <coroutine>
#include <exception>
#include <vector>
#include <cassert>
#ifndef FHERMA_TPI
#define FHERMA_TPI 1
#endif
#define __global__
#define __device__
#define __shared__ static
struct dim3 { unsigned x,y,z; dim3(unsigned x_=1,unsigned y_=1,unsigned z_=1):x(x_),y(y_),z(z_){} };
inline dim3 blockIdx,blockDim,threadIdx;
// All participating lanes yield before reading. Alternating slots prevent the
// next shuffle from overwriting values that later lanes have not yet read.
inline uint32_t shuffle_values[2][1024];
inline unsigned shuffle_ordinals[1024];
struct EmulatedShuffleXor {
    uint32_t value; unsigned owner,peer,slot;
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<>) const noexcept { shuffle_values[slot][owner]=value; }
    uint32_t await_resume() const noexcept { return shuffle_values[slot][peer]; }
};
inline EmulatedShuffleXor emulated_shuffle_xor(unsigned mask,uint32_t value,unsigned delta,unsigned width) {
    assert(mask==0xffffffff && width<=32 && width && !(width&(width-1)) && delta<width);
    unsigned owner=threadIdx.x;
    return {value,owner,owner^delta,shuffle_ordinals[owner]++&1};
}
inline uint32_t __umulhi(uint32_t a,uint32_t b) { return uint32_t(uint64_t(a)*b>>32); }
inline uint64_t __umul64hi(uint64_t a,uint64_t b) { return uint64_t((__uint128_t(a)*b)>>64); }
inline unsigned __brev(unsigned x) {
    unsigned y=0; for(unsigned i=0;i<32;++i) { y=(y<<1)|(x&1); x>>=1; } return y;
}
struct EmulatedKernel {
    struct promise_type {
        std::exception_ptr failure;
        EmulatedKernel get_return_object() { return {std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { failure=std::current_exception(); }
    };
    std::coroutine_handle<promise_type> handle;
    EmulatedKernel(std::coroutine_handle<promise_type> h):handle(h){}
    EmulatedKernel(EmulatedKernel&& other):handle(other.handle) { other.handle=nullptr; }
    ~EmulatedKernel() { if(handle) handle.destroy(); }
};
template<class F> void emulate_launch(dim3 grid,dim3 threads,F f) {
    blockDim=threads;
    for(blockIdx.y=0;blockIdx.y<grid.y;++blockIdx.y)
        for(blockIdx.x=0;blockIdx.x<grid.x;++blockIdx.x) {
            std::memset(shuffle_ordinals,0,sizeof(shuffle_ordinals));
            std::vector<EmulatedKernel> tasks;
            for(threadIdx.x=0;threadIdx.x<threads.x;threadIdx.x+=FHERMA_TPI) tasks.push_back(f());
            unsigned active=tasks.size();
            while(active) {
                for(unsigned t=0;t<tasks.size();++t) {
                    auto h=tasks[t].handle;
                    if(h.done()) continue;
                    threadIdx.x=t*FHERMA_TPI;
                    h.resume();
                    if(h.promise().failure) std::rethrow_exception(h.promise().failure);
                    if(h.done()) --active;
                }
            }
        }
}
using cudaError_t=int;
using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
inline const char* cudaGetErrorString(int) { return "host emulation error"; }
template<class T> inline int cudaMalloc(T** p,size_t n) { *p=static_cast<T*>(std::calloc(1,n)); return *p?0:1; }
inline int cudaFree(void* p) { std::free(p); return 0; }
inline int cudaMallocHost(void** p,size_t n) { *p=std::calloc(1,n); return *p?0:1; }
inline int cudaFreeHost(void* p) { std::free(p); return 0; }
inline int cudaHostRegister(void*,size_t,unsigned) { return 0; }
inline int cudaHostUnregister(void*) { return 0; }
inline int cudaMemcpy(void* dst,const void* src,size_t n,int) { std::memcpy(dst,src,n); return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline int cudaGetLastError() { return 0; }
namespace cupqc {
template<unsigned N> struct BitWidth {};
template<unsigned N> struct SM {};
struct Thread {};
struct Warp {};
template<unsigned N> struct TPI {};
template<unsigned Words> struct EmulatedWide;
template<unsigned Words> struct EmulatedBig {
    uint32_t limbs[Words]{};
    explicit EmulatedBig(uint32_t x) { limbs[0]=x; }
    EmulatedBig(const uint32_t* p,unsigned i) { std::memcpy(limbs,p+i*Words,Words*4); }
    explicit EmulatedBig(mpz_class x) {
        x &= mask(); mpz_export(limbs,nullptr,-1,4,0,0,x.get_mpz_t());
    }
    mpz_class value() const {
        mpz_class x; mpz_import(x.get_mpz_t(),Words,-1,4,0,0,limbs); return x;
    }
    static mpz_class mask() { return (mpz_class(1)<<(Words*32))-1; }
    const uint32_t& operator[](unsigned k) const { return limbs[k]; }
    uint32_t& operator[](unsigned k) { return limbs[k]; }
    EmulatedBig operator<<(unsigned n) const { return EmulatedBig(mpz_class((value()<<n)&mask())); }
    EmulatedBig operator>>(unsigned n) const { return EmulatedBig(mpz_class(value()>>n)); }
    EmulatedBig operator|(const EmulatedBig& b) const { return EmulatedBig(mpz_class(value()|b.value())); }
    EmulatedBig operator+(const EmulatedBig& b) const { return EmulatedBig(mpz_class((value()+b.value())&mask())); }
    EmulatedBig operator-(const EmulatedBig& b) const { return EmulatedBig(mpz_class((value()-b.value())&mask())); }
    bool operator>=(const EmulatedBig& b) const { return value()>=b.value(); }
    EmulatedBig mul_scalar(uint32_t b) const { return EmulatedBig(mpz_class((value()*b)&mask())); }
    EmulatedWide<Words> mul_wide(const EmulatedBig& b) const;
    static const mpz_class& rinv(const EmulatedBig& q) {
        static mpz_class previous, inverse;
        if(previous!=q.value()) {
            previous=q.value(); mpz_class r=mpz_class(1)<<(Words*32);
            mpz_invert(inverse.get_mpz_t(),r.get_mpz_t(),q.value().get_mpz_t());
        }
        return inverse;
    }
    EmulatedBig to_montgomery(const EmulatedBig& q) const {
        return EmulatedBig(mpz_class((value()<<(Words*32))%q.value()));
    }
    EmulatedBig from_montgomery(const EmulatedBig& q) const {
        return EmulatedBig(mpz_class(value()*rinv(q)%q.value()));
    }
    EmulatedBig mul_montgomery(const EmulatedBig& b,const EmulatedBig& q) const {
        return EmulatedBig(mpz_class(value()*b.value()*rinv(q)%q.value()));
    }
    void store(uint32_t* p,unsigned i) const {
        std::memcpy(p+i*Words,limbs,Words*4);
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
template<unsigned Words> struct EmulatedWide { EmulatedBig<Words> lo,hi; };
template<unsigned Words>
inline EmulatedWide<Words> EmulatedBig<Words>::mul_wide(const EmulatedBig<Words>& b) const {
    mpz_class p=value()*b.value();
    return {EmulatedBig<Words>(mpz_class(p&mask())),EmulatedBig<Words>(mpz_class(p>>(Words*32)))};
}
template<unsigned W> struct Descriptor { using bigint=EmulatedBig<W/32>; using modulus=bigint; };
template<unsigned W,unsigned S> Descriptor<W> operator+(BitWidth<W>,SM<S>){return {};}
template<unsigned W> Descriptor<W> operator+(Descriptor<W>,Thread){return {};}
template<unsigned W> Descriptor<W> operator+(Descriptor<W>,Warp){return {};}
template<unsigned W,unsigned N> Descriptor<W> operator+(Descriptor<W>,TPI<N>){return {};}
}
