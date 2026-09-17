#ifndef FHERMA_KARATSUBA
#define FHERMA_KARATSUBA 1
#endif
#ifndef FHERMA_PIPELINE_OUTPUT
#define FHERMA_PIPELINE_OUTPUT 4
#endif
#ifndef FHERMA_PIPELINE_INPUT
#define FHERMA_PIPELINE_INPUT 4
#endif
#ifndef FHERMA_PREFAULT_OUTPUT
#define FHERMA_PREFAULT_OUTPUT 1
#endif
#ifndef FHERMA_SMALL_TABLES
#define FHERMA_SMALL_TABLES 1
#endif
#ifndef FHERMA_TAIL_TABLES
#define FHERMA_TAIL_TABLES 1
#endif
#ifndef FHERMA_FUSED_PREPARE
#define FHERMA_FUSED_PREPARE 0
#endif
#ifndef FHERMA_SPECIAL_ADD_SUB
#define FHERMA_SPECIAL_ADD_SUB 0
#endif
#ifndef FHERMA_HOST_PROFILE
#define FHERMA_HOST_PROFILE 0
#endif
#ifndef FHERMA_FUSED_TAIL
#define FHERMA_FUSED_TAIL 1
#endif
#ifndef FHERMA_MIN_BLOCKS
#define FHERMA_MIN_BLOCKS 0
#endif
#ifndef FHERMA_SPIN_COPY
#define FHERMA_SPIN_COPY 1
#endif
#ifndef FHERMA_DIRECT_OUTPUT
#define FHERMA_DIRECT_OUTPUT 0
#endif
#ifndef FHERMA_GRAPH
#define FHERMA_GRAPH 1
#endif
#ifndef FHERMA_MONTGOMERY
#define FHERMA_MONTGOMERY 0
#endif
#ifndef FHERMA_SOA
#define FHERMA_SOA 1
#endif
#ifndef FHERMA_PROFILE
#define FHERMA_PROFILE 0
#endif

#ifndef FHERMA_TPI
#define FHERMA_TPI 1
#endif
#ifndef FHERMA_PINNED
#define FHERMA_PINNED 1
#endif
#ifndef FHERMA_FUSED_SMALL
#define FHERMA_FUSED_SMALL 1
#endif
#ifndef FHERMA_NUMA
#define FHERMA_NUMA 1
#endif
#ifndef FHERMA_REGISTER_INPUTS
#define FHERMA_REGISTER_INPUTS 0
#endif
#ifndef FHERMA_PARALLEL_COPY
#define FHERMA_PARALLEL_COPY 1
#endif
#ifndef FHERMA_STREAM_COPY
#define FHERMA_STREAM_COPY 1
#endif
#ifndef FHERMA_COPY_THREADS
#define FHERMA_COPY_THREADS 8
#endif
#ifndef FHERMA_PARALLEL_OUTPUT
#define FHERMA_PARALLEL_OUTPUT 1
#endif
#ifndef FHERMA_CONSTANT_MODULUS
#define FHERMA_CONSTANT_MODULUS 1
#endif
#ifndef FHERMA_SKIP_IDENTITY
#define FHERMA_SKIP_IDENTITY 1
#endif

// Exact negacyclic NTT baseline. All device modular arithmetic uses cuPQC.
#include "fherma.h"
#include "wide_host.h"
#include <cupqc/bigint.hpp>
#include <cuda_runtime.h>
#include "host_affinity.h"
#include "host_copy_pool.h"
#include "input_pipeline.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace {
constexpr unsigned L=28;
#if FHERMA_TPI == 1
using BI=decltype(cupqc::BitWidth<L*32>()+cupqc::SM<800>()+cupqc::Thread());
#else
static_assert(FHERMA_MONTGOMERY && !FHERMA_SOA,"cooperative experiment uses Montgomery and native layout");
using BI=decltype(cupqc::BitWidth<L*32>()+cupqc::SM<800>()+cupqc::TPI<FHERMA_TPI>()+cupqc::Warp());
#endif
using Big=typename BI::bigint;
__device__ Big load_coeff(const uint32_t* p,unsigned i,unsigned n) {
#if FHERMA_SOA
    Big x(uint32_t(0));
    #pragma unroll
    for(unsigned k=0;k<L;++k) x[k]=p[k*n+i];
    return x;
#else
    return Big(p,i);
#endif
}
__device__ void store_coeff(const Big& x,uint32_t* p,unsigned i,unsigned n) {
#if FHERMA_SOA
    #pragma unroll
    for(unsigned k=0;k<L;++k) p[k*n+i]=x[k];
#else
    x.store(p,i);
#endif
}
#if FHERMA_MONTGOMERY
using Mod=typename BI::modulus;
__device__ Big encode(const Big& x,const Mod& q) { return x.to_montgomery(q); }
__device__ Big decode(const Big& x,const Mod& q) { return x.from_montgomery(q); }
__device__ Big multiply(const Big& a,const Big& b,const Mod& q) { return a.mul_montgomery(b,q); }
#else
using Mod=Big;
__device__ Big encode(const Big& x,const Mod&) { return x; }
__device__ Big decode(const Big& x,const Mod&) { return x; }
#if FHERMA_KARATSUBA
using HalfBI=decltype(cupqc::BitWidth<448>()+cupqc::SM<800>()+cupqc::Thread());
using Half=typename HalfBI::bigint;
struct SplitProduct {Big lo,hi;};
// Exact for a,b<2^868. The middle coefficient a0*b1+a1*b0 is <2^869,
// so its calculation modulo 2^896 is exact even if z0+z2 wraps first.
__device__ SplitProduct karatsuba_product(const Big& a,const Big& b) {
    Half a0(uint32_t(0)),a1(uint32_t(0)),b0(uint32_t(0)),b1(uint32_t(0));
    #pragma unroll
    for(unsigned k=0;k<14;++k) {a0[k]=a[k];a1[k]=a[k+14];b0[k]=b[k];b1[k]=b[k+14];}
    bool a_positive=a1>=a0,b_positive=b0>=b1;
    Half da=a_positive ? a1-a0 : a0-a1,db=b_positive ? b0-b1 : b1-b0;
    auto z0=a0.mul_wide(b0),z2=a1.mul_wide(b1);
    SplitProduct result{Big(uint32_t(0)),Big(uint32_t(0))};
    #pragma unroll
    for(unsigned k=0;k<14;++k) {
        result.lo[k]=z0.lo[k];result.lo[k+14]=z0.hi[k];
        result.hi[k]=z2.lo[k];result.hi[k+14]=z2.hi[k];
    }
    Big middle=result.lo+result.hi;
    auto delta=da.mul_wide(db);
    Big difference(uint32_t(0));
    #pragma unroll
    for(unsigned k=0;k<14;++k) {difference[k]=delta.lo[k];difference[k+14]=delta.hi[k];}
    middle=a_positive==b_positive ? middle+difference : middle-difference;
    uint64_t carry=0;
    #pragma unroll
    for(unsigned k=0;k<14;++k) {
        uint64_t word=uint64_t(result.lo[k+14])+middle[k]+carry;
        result.lo[k+14]=uint32_t(word);carry=word>>32;
    }
    #pragma unroll
    for(unsigned k=0;k<28;++k) {
        uint64_t word=uint64_t(result.hi[k])+(k<14 ? uint32_t(middle[k+14]) : 0u)+carry;
        result.hi[k]=uint32_t(word);carry=word>>32;
    }
    return result;
}
#endif
// For q = 2^868-c, c < 2^28, a full product folds twice without division.
// At each fold all intermediates fit the 896-bit cuPQC storage width.
// The modulus shape is checked once on the host during setup.
__device__ Big multiply(const Big& a,const Big& b,const Big& q) {
    const uint32_t c=uint32_t(0)-q[0];
#if FHERMA_KARATSUBA
    auto p=karatsuba_product(a,b);
#else
    auto p=a.mul_wide(b);
#endif
    // Extract z>>868 before overwriting the low half. Coefficients use
    // 27 full words and four bits of word 27.
    uint32_t h0=(p.lo[27]>>4)|(p.hi[0]<<28);
    uint64_t carry=0;
    #pragma unroll
    for(unsigned k=0;k<28;++k) {
        uint32_t h=k==0 ? h0 : ((p.hi[k-1]>>4)|(p.hi[k]<<28));
        uint32_t low=k==27 ? (p.lo[k]&15u) : uint32_t(p.lo[k]);
        uint64_t v=uint64_t(h)*c+low+carry;
        p.lo[k]=uint32_t(v); carry=v>>32;
    }
    carry=uint64_t(p.lo[27]>>4)*c;
    p.lo[27]=p.lo[27]&15u;
    #pragma unroll
    for(unsigned k=0;k<28;++k) {
        uint64_t v=uint64_t(p.lo[k])+carry;
        p.lo[k]=uint32_t(v); carry=v>>32;
    }
    if(p.lo>=q) p.lo=p.lo-q;
    return p.lo;
}
#endif
__device__ Big butterfly_add(const Big& a,const Big& b,const Mod& q) {
#if FHERMA_SPECIAL_ADD_SUB && !FHERMA_MONTGOMERY
    Big sum=a+b;
    uint32_t folded=(sum[27]>>4)*(uint32_t(0)-q[0]);
    sum[27]=sum[27]&15u;
    sum=sum+Big(folded);
    // After folding, sum<2^868. Only the c-sized interval below 2^868
    // can still need subtraction; compare the fixed all-one upper words.
    uint32_t upper=sum[27]|0xfffffff0u;
    #pragma unroll
    for(unsigned k=1;k<27;++k) upper &= uint32_t(sum[k]);
    if(upper==0xffffffffu && sum[0]>=q[0]) sum=sum-q;
    return sum;
#else
    return a.add_mod(b,q);
#endif
}
__device__ Big butterfly_sub(const Big& a,const Big& b,const Mod& q) {
#if FHERMA_SPECIAL_ADD_SUB && !FHERMA_MONTGOMERY
    Big difference=a-b;
    uint32_t correction=(difference[27]>>31)*(uint32_t(0)-q[0]);
    difference[27]=difference[27]&15u;
    return difference-Big(correction);
#else
    return a.sub_mod(b,q);
#endif
}
__device__ Mod load_modulus(const uint32_t* qp) {
#if FHERMA_CONSTANT_MODULUS && !FHERMA_MONTGOMERY
    // init has checked these words, so they need no global loads or registers.
    Big q(uint32_t(0)); q[0]=qp[0];
    #pragma unroll
    for(unsigned k=1;k<27;++k) q[k]=0xffffffffu;
    q[27]=15u;
    return q;
#else
    return Mod(qp,0);
#endif
}
void check(cudaError_t e,const char* op) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));
}
struct RegisteredInput {
    void* ptr=nullptr;
    bool pin(const uint32_t* data,size_t bytes) {
        auto p=const_cast<uint32_t*>(data);
        if(cudaHostRegister(p,bytes,0)==cudaSuccess) { ptr=p; return true; }
        cudaGetLastError(); // Registration is optional; staged copies are the fallback.
        return false;
    }
    void release() { if(ptr) { check(cudaHostUnregister(ptr),"unregister input"); ptr=nullptr; } }
    ~RegisteredInput() { if(ptr) cudaHostUnregister(ptr); }
};
// One owned vector is prepared ahead. Every run replenishes it inside the
// measured call, while the GPU can execute. Returned storage is unregistered
// before ownership passes to the caller; no caller memory is retained.
struct NextOutput {
    std::vector<uint32_t> data;
    bool registered=false;
    double reserve_us=0,prefault_us=0,zero_us=0,register_us=0;
    void prepare(size_t words,HostCopyPool* copy=nullptr) {
#if FHERMA_HOST_PROFILE
        using Clock=std::chrono::steady_clock;
        auto start=Clock::now();
#endif
        // Keep a page of spare capacity between registered allocations.
        data.reserve(words+1024);
#if FHERMA_HOST_PROFILE
        auto reserved=Clock::now();
#endif
        if(FHERMA_PREFAULT_OUTPUT && copy) copy->prefault(data.data(),words*4);
#if FHERMA_HOST_PROFILE
        auto touched=Clock::now();
#endif
        data.resize(words);
#if FHERMA_HOST_PROFILE
        auto zeroed=Clock::now();
#endif
        check(cudaHostRegister(data.data(),words*4,0),"register next output");
        registered=true;
#if FHERMA_HOST_PROFILE
        auto pinned=Clock::now();
        auto us=[](auto a,auto b) { return std::chrono::duration<double,std::micro>(b-a).count(); };
        reserve_us=us(start,reserved); prefault_us=us(reserved,touched);
        zero_us=us(touched,zeroed); register_us=us(zeroed,pinned);
#endif
    }
    void take(std::vector<uint32_t>& out,RegisteredInput& registration) {
        if(!registered) throw std::runtime_error("output buffer unavailable after a failed run");
        out.swap(data); registration.ptr=out.data(); registered=false;
    }
    ~NextOutput() { if(registered) cudaHostUnregister(data.data()); }
};
struct State {
#if FHERMA_DIRECT_OUTPUT
    NextOutput next_output;
#endif
#if FHERMA_PARALLEL_COPY
    HostCopyPool copy;
#endif
#if FHERMA_GRAPH
    cudaStream_t stream=nullptr;
    cudaGraphExec_t graph=nullptr;
#if FHERMA_PIPELINE_OUTPUT>1
    cudaEvent_t output_ready[FHERMA_PIPELINE_OUTPUT]{};
#endif
#endif
    uint32_t n=0, logn=0;
    uint32_t *q=nullptr,*roots=nullptr,*twist=nullptr,*inv_twist=nullptr,*scale=nullptr;
    uint32_t *input=nullptr,*ab=nullptr,*c=nullptr;
    uint32_t *tail_twist=nullptr,*tail_inv_twist=nullptr;
    uint32_t *small_twist=nullptr,*small_inv_twist=nullptr;
#if FHERMA_PINNED
    uint32_t *host_input=nullptr,*host_output=nullptr;
#endif
#if FHERMA_PROFILE
    cudaEvent_t events[8]{};
#endif
    ~State() {
#if FHERMA_GRAPH
        if(graph) cudaGraphExecDestroy(graph);
        if(stream) cudaStreamDestroy(stream);
#if FHERMA_PIPELINE_OUTPUT>1
        for(auto event:output_ready) if(event) cudaEventDestroy(event);
#endif
#endif
        cudaFree(q); cudaFree(roots); cudaFree(twist); cudaFree(inv_twist);
        cudaFree(scale); cudaFree(input); cudaFree(ab); cudaFree(c);
        cudaFree(tail_twist); cudaFree(tail_inv_twist);
        cudaFree(small_twist); cudaFree(small_inv_twist);
#if FHERMA_PINNED
        cudaFreeHost(host_input); cudaFreeHost(host_output);
#endif
#if FHERMA_PROFILE
        for(auto e:events) if(e) cudaEventDestroy(e);
#endif
    }
};
void mark(State& s,unsigned i) {
#if FHERMA_PROFILE
    check(cudaEventRecord(s.events[i]),"profile record");
#else
    (void)s; (void)i;
#endif
}
__device__ Big pow_small(Big x,unsigned e,const Mod& q) {
    Big y=encode(Big(uint32_t(1)),q);
    while(e) { if(e&1) y=multiply(y,x,q); e>>=1; if(e) x=multiply(x,x,q); }
    return y;
}
__global__ void make_tables(const uint32_t* qp,const uint32_t* roots,
                           uint32_t* twist,uint32_t* inv_twist,uint32_t* scale,unsigned n) {
    unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(i>=n) return;
    const Mod q=load_modulus(qp);
    const Big psi=encode(Big(roots,0),q), invpsi=encode(Big(roots,1),q), invn=encode(Big(roots,2),q);
    auto a=pow_small(psi,i,q), b=pow_small(invpsi,i,q);
    store_coeff(a,twist,i,n); store_coeff(b,inv_twist,i,n);
    store_coeff(multiply(b,invn,q),scale,i,n);
}
__global__ void prepare(const uint32_t* input,uint32_t* ab,const uint32_t* twist,
                        const uint32_t* qp,unsigned n,unsigned logn) {
    unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(i>=n) return;
    unsigned poly=blockIdx.y, j=__brev(i)>>(32-logn);
    const Mod q=load_modulus(qp);
    const Big x=encode(Big(input,poly*n+i),q), t=load_coeff(twist,i,n);
    store_coeff(multiply(x,t,q),ab+poly*n*L,j,n);
}
#if FHERMA_MIN_BLOCKS && FHERMA_TPI==1
__global__ __launch_bounds__(128,FHERMA_MIN_BLOCKS)
#else
__global__
#endif
void stage(uint32_t* values,const uint32_t* table,const uint32_t* qp,
                      unsigned n,unsigned half,unsigned table_stride) {
    unsigned k=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(k>=n/2) return;
    unsigned j=k&(half-1), i=2*(k-j)+j;
    values+=blockIdx.y*n*L;
    const Mod q=load_modulus(qp);
    const Big u=load_coeff(values,i,n), v=load_coeff(values,i+half,n), tw=load_coeff(table,j*table_stride,n);
    auto t=(FHERMA_SKIP_IDENTITY && j==0) ? v : multiply(v,tw,q);
    store_coeff(butterfly_add(u,t,q),values,i,n); store_coeff(butterfly_sub(u,t,q),values,i+half,n);
}
// Keep the first eight radix-2 stages in shared memory. Every block handles
// an independent contiguous tile; no inter-block synchronization is needed.
#if FHERMA_MIN_BLOCKS && FHERMA_TPI==1
__global__ __launch_bounds__(128,FHERMA_MIN_BLOCKS)
#else
__global__
#endif
void small_stages(uint32_t* values,const uint32_t* table,const uint32_t* qp,unsigned n,
                  const uint32_t* raw_input,unsigned logn,const uint32_t* compact_table) {
    __shared__ uint32_t tile[256*L];
    unsigned t=threadIdx.x/FHERMA_TPI, base=blockIdx.x*256;
    values+=blockIdx.y*n*L;
    const Mod q=load_modulus(qp);
    if(raw_input) {
        for(unsigned offset=0;offset<2;++offset) {
            unsigned local=2*t+offset, i=__brev(base+local)>>(32-logn);
            const Big x=encode(Big(raw_input,blockIdx.y*n+i),q),tw=load_coeff(table,i,n);
            store_coeff(multiply(x,tw,q),tile,local,256);
        }
    } else {
        store_coeff(load_coeff(values,base+2*t,n),tile,2*t,256);
        store_coeff(load_coeff(values,base+2*t+1,n),tile,2*t+1,256);
    }
    __syncthreads();
    unsigned table_stride=FHERMA_SMALL_TABLES ? 256 : n;
    for(unsigned half=1;half<256;half*=2,table_stride>>=1) {
        unsigned j=t&(half-1), i=2*(t-j)+j;
        const Big u=load_coeff(tile,i,256), v=load_coeff(tile,i+half,256);
        const Big tw=FHERMA_SMALL_TABLES ? load_coeff(compact_table,j*table_stride,256) : load_coeff(table,j*table_stride,n);
        const Big m=(FHERMA_SKIP_IDENTITY && j==0) ? v : multiply(v,tw,q);
        store_coeff(butterfly_add(u,m,q),tile,i,256);
        store_coeff(butterfly_sub(u,m,q),tile,i+half,256);
        __syncthreads();
    }
    store_coeff(load_coeff(tile,2*t,256),values,base+2*t,n);
    store_coeff(load_coeff(tile,2*t+1,256),values,base+2*t+1,n);
}
// Transpose each limb plane: [128 rows][256 columns] -> [256][128].
// Each warp performs contiguous reads and writes; padding avoids bank conflicts.
__global__ void transpose_tail(const uint32_t* source,uint32_t* dest,unsigned n) {
    __shared__ uint32_t tile[32*33];
    unsigned plane=blockIdx.x/32, block=blockIdx.x%32;
    unsigned row=block/8*32, column=block%8*32;
    unsigned x=threadIdx.x%32,y=threadIdx.x/32;
    source+=(blockIdx.y*L+plane)*n;
    dest+=(blockIdx.y*L+plane)*n;
    for(unsigned dy=0;dy<32;dy+=8)
        tile[(y+dy)*33+x]=source[(row+y+dy)*256+column+x];
    __syncthreads();
    for(unsigned dy=0;dy<32;dy+=8)
        dest[(column+y+dy)*128+row+x]=tile[x*33+y+dy];
}
__global__ void make_small_tables(const uint32_t* forward,const uint32_t* inverse,
                                  uint32_t* out_forward,uint32_t* out_inverse,unsigned n) {
    unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(i>=256) return;
    store_coeff(load_coeff(forward,i*(n/256),n),out_forward,i,256);
    store_coeff(load_coeff(inverse,i*(n/256),n),out_inverse,i,256);
}
// Stage-major twiddles, with consecutive j within each fixed low-bit column.
// The conventional table becomes strided after the coefficient transpose.
__global__ void make_tail_tables(const uint32_t* forward,const uint32_t* inverse,
                                 uint32_t* out_forward,uint32_t* out_inverse,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=32512) return;
    unsigned half=1;
    while(i>=256*(2*half-1)) half*=2;
    unsigned within=i-256*(half-1),column=within/half,j=within%half;
    unsigned exponent=(column+256*j)*(128/half);
    store_coeff(load_coeff(forward,exponent,n),out_forward,i,n);
    store_coeff(load_coeff(inverse,exponent,n),out_inverse,i,n);
}
// The seven remaining stages are independent for each low eight-bit column.
// After the transpose, two entire columns fit one shared-memory tile.
__global__ void tail_stages(uint32_t* values,const uint32_t* table,const uint32_t* qp,unsigned n) {
    __shared__ uint32_t tile[256*L];
    unsigned t=threadIdx.x, base=blockIdx.x*256;
    unsigned column=2*blockIdx.x+t/64, k=t%64, offset=(t/64)*128;
    values+=blockIdx.y*n*L;
    const Mod q=load_modulus(qp);
    store_coeff(load_coeff(values,base+2*t,n),tile,2*t,256);
    store_coeff(load_coeff(values,base+2*t+1,n),tile,2*t+1,256);
    __syncthreads();
    unsigned table_stride=n/256;
    for(unsigned half=1;half<128;half*=2,table_stride>>=1) {
        unsigned j=k&(half-1), i=offset+2*(k-j)+j;
        unsigned exponent=(column+256*j)*table_stride;
        const Big u=load_coeff(tile,i,256),v=load_coeff(tile,i+half,256);
        unsigned twiddle_i=FHERMA_TAIL_TABLES ? (256*(half-1)+column*half+j) : exponent;
        const Big tw=load_coeff(table,twiddle_i,n);
        const Big m=(FHERMA_SKIP_IDENTITY && column==0 && j==0) ? v : multiply(v,tw,q);
        store_coeff(butterfly_add(u,m,q),tile,i,256);
        store_coeff(butterfly_sub(u,m,q),tile,i+half,256);
        __syncthreads();
    }
    store_coeff(load_coeff(tile,2*t,256),values,base+2*t,n);
    store_coeff(load_coeff(tile,2*t+1,256),values,base+2*t+1,n);
}
__device__ unsigned frequency_index(unsigned i,unsigned n) {
    return (FHERMA_FUSED_TAIL && n==32768) ? ((i&255)*128+(i>>8)) : i;
}
__global__ void product(const uint32_t* ab,uint32_t* c,const uint32_t* qp,
                        unsigned n,unsigned logn) {
    unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(i>=n) return;
    const Mod q=load_modulus(qp);
    unsigned source_i=frequency_index(i,n);
    const Big a=load_coeff(ab,source_i,n), b=load_coeff(ab+n*L,source_i,n);
    store_coeff(multiply(a,b,q),c,__brev(i)>>(32-logn),n);
}
__global__ void finish(const uint32_t* c,uint32_t* out,const uint32_t* scale,const uint32_t* qp,unsigned n) {
    unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(i>=n) return;
    const Mod q=load_modulus(qp);
    const Big x=load_coeff(c,frequency_index(i,n),n), s=load_coeff(scale,i,n);
    decode(multiply(x,s,q),q).store(out,i);
}
void launch_ntt(State& s,cudaStream_t stream=nullptr) {
    dim3 full((s.n*FHERMA_TPI+127)/128,2), halves((s.n/2*FHERMA_TPI+127)/128,2);
    bool fused_prepare=FHERMA_FUSED_PREPARE && FHERMA_FUSED_SMALL && s.n>=256;
    if(!fused_prepare) prepare<<<full,128,0,stream>>>(s.input,s.ab,s.twist,s.q,s.n,s.logn);
    mark(s,2);
    unsigned first=1;
    if(FHERMA_FUSED_SMALL && s.n>=256) {
        dim3 tiles(s.n/256,2);
        small_stages<<<tiles,128*FHERMA_TPI,0,stream>>>(s.ab,s.twist,s.q,s.n,fused_prepare?s.input:nullptr,s.logn,s.small_twist);
        first=256;
    }
    uint32_t* forward_values=s.ab;
    if(FHERMA_FUSED_TAIL && s.n==32768) {
        dim3 planes(32*L,2),tiles(128,2);
        transpose_tail<<<planes,256,0,stream>>>(s.ab,s.input,s.n);
        tail_stages<<<tiles,128,0,stream>>>(s.input,FHERMA_TAIL_TABLES?s.tail_twist:s.twist,s.q,s.n);
        forward_values=s.input;
    } else {
        for(unsigned half=first;half<s.n;half*=2) stage<<<halves,128,0,stream>>>(s.ab,s.twist,s.q,s.n,half,s.n/half);
    }
    mark(s,3);
    product<<<full.x,128,0,stream>>>(forward_values,s.c,s.q,s.n,s.logn);
    mark(s,4);
    if(FHERMA_FUSED_SMALL && s.n>=256) {
        unsigned tiles=s.n/256;
        small_stages<<<tiles,128*FHERMA_TPI,0,stream>>>(s.c,s.inv_twist,s.q,s.n,nullptr,s.logn,s.small_inv_twist);
    }
    uint32_t* inverse_values=s.c;
    if(FHERMA_FUSED_TAIL && s.n==32768) {
        transpose_tail<<<32*L,256,0,stream>>>(s.c,s.ab,s.n);
        tail_stages<<<128,128,0,stream>>>(s.ab,FHERMA_TAIL_TABLES?s.tail_inv_twist:s.inv_twist,s.q,s.n);
        inverse_values=s.ab;
    } else {
        for(unsigned half=first;half<s.n;half*=2) stage<<<halves.x,128,0,stream>>>(s.c,s.inv_twist,s.q,s.n,half,s.n/half);
    }
    mark(s,5);
    finish<<<full.x,128,0,stream>>>(inverse_values,s.input,s.scale,s.q,s.n);
    mark(s,6);
    check(cudaGetLastError(),"NTT launch");
}
} // namespace

void* fherma_init(const fherma::Point& p) {
    static_assert(!FHERMA_FUSED_TAIL || (FHERMA_SOA && FHERMA_TPI==1 && FHERMA_FUSED_SMALL),"tail fusion uses single-thread SoA and eight initial stages");
    if(p.N<2 || (p.N&(p.N-1)) || p.N>32768 || p.W!=868 || p.L!=L || p.q.data.size()!=L)
        throw std::runtime_error("coverage: power-of-two 2<=N<=32768, W=868, L=28");
    if(FHERMA_NUMA) pin_near_gpu();
#ifdef __CUDACC__
    int pageable=0,host_tables=0,managed=0;
    cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,0);
    cudaDeviceGetAttribute(&host_tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,0);
    cudaDeviceGetAttribute(&managed,cudaDevAttrConcurrentManagedAccess,0);
    std::fprintf(stderr,"MEMORY_CAPS pageable=%d host_tables=%d concurrent_managed=%d\n",pageable,host_tables,managed);
#endif
#if !FHERMA_MONTGOMERY
    uint32_t delta=uint32_t(0)-p.q.data[0];
    bool special=delta>0 && delta<0x10000000u && p.q.data[27]==15u;
    for(unsigned k=1;k<27;++k) special=special && p.q.data[k]==0xffffffffu;
    if(!special) throw std::runtime_error("coverage: q=2^868-c with 0<c<2^28");
#endif
    auto s=std::make_unique<State>(); s->n=p.N; s->logn=__builtin_ctz(p.N);
    auto root=host_wide::roots(p.q.data,p.N);
    std::vector<uint32_t> packed=root.psi;
    packed.insert(packed.end(),root.invpsi.begin(),root.invpsi.end());
    packed.insert(packed.end(),root.invn.begin(),root.invn.end());
    auto alloc=[](uint32_t** ptr,size_t coeffs) { check(cudaMalloc(ptr,coeffs*L*4),"cudaMalloc"); };
    alloc(&s->q,1); alloc(&s->roots,3); alloc(&s->twist,p.N); alloc(&s->inv_twist,p.N);
    alloc(&s->scale,p.N); alloc(&s->input,2*p.N); alloc(&s->ab,2*p.N); alloc(&s->c,p.N);
#if FHERMA_PINNED
    check(cudaMallocHost(reinterpret_cast<void**>(&s->host_input),size_t(2)*p.N*L*4),"pinned inputs");
    check(cudaMallocHost(reinterpret_cast<void**>(&s->host_output),size_t(p.N)*L*4),"pinned output");
    std::memset(s->host_input,0,size_t(2)*p.N*L*4);
    std::memset(s->host_output,0,size_t(p.N)*L*4);
#endif
#if FHERMA_DIRECT_OUTPUT
    static_assert(FHERMA_PINNED && !FHERMA_REGISTER_INPUTS,"direct output requires staged pinned inputs");
    s->next_output.prepare(size_t(p.N)*L);
#endif
    check(cudaMemcpy(s->q,p.q.data.data(),L*4,cudaMemcpyHostToDevice),"copy q");
    check(cudaMemcpy(s->roots,packed.data(),3*L*4,cudaMemcpyHostToDevice),"copy roots");
    make_tables<<<(p.N*FHERMA_TPI+127)/128,128>>>(s->q,s->roots,s->twist,s->inv_twist,s->scale,p.N);
    check(cudaGetLastError(),"table launch");
    if(FHERMA_SMALL_TABLES && FHERMA_FUSED_SMALL && p.N>=256) {
        alloc(&s->small_twist,256); alloc(&s->small_inv_twist,256);
        make_small_tables<<<2*FHERMA_TPI,128>>>(s->twist,s->inv_twist,s->small_twist,s->small_inv_twist,p.N);
        check(cudaGetLastError(),"small table launch");
    }
    if(FHERMA_FUSED_TAIL && FHERMA_TAIL_TABLES && p.N==32768) {
        alloc(&s->tail_twist,p.N); alloc(&s->tail_inv_twist,p.N);
        make_tail_tables<<<254,128>>>(s->twist,s->inv_twist,s->tail_twist,s->tail_inv_twist,p.N);
        check(cudaGetLastError(),"tail table launch");
    }
    check(cudaDeviceSynchronize(),"table setup");
#if FHERMA_PROFILE
    for(auto& e:s->events) check(cudaEventCreate(&e),"profile create");
#endif
#if FHERMA_GRAPH
    static_assert(FHERMA_PINNED && !FHERMA_PROFILE && !FHERMA_REGISTER_INPUTS,"graph uses fixed pinned buffers without diagnostic events");
#if FHERMA_PIPELINE_OUTPUT>1
    static_assert(!FHERMA_DIRECT_OUTPUT && FHERMA_PARALLEL_COPY && FHERMA_PARALLEL_OUTPUT,
                  "output pipeline requires staged parallel output");
    static_assert(FHERMA_PIPELINE_OUTPUT<=32,"output segments must fit the smallest supported point");
    for(auto& event:s->output_ready) check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"output segment event");
#endif
    check(cudaStreamCreateWithFlags(&s->stream,cudaStreamNonBlocking),"graph stream");
    check(cudaStreamBeginCapture(s->stream,cudaStreamCaptureModeGlobal),"begin capture");
    size_t bytes=size_t(s->n)*L*4;
#if FHERMA_PIPELINE_INPUT<=1
    check(cudaMemcpyAsync(s->input,s->host_input,2*bytes,cudaMemcpyHostToDevice,s->stream),"capture H2D");
#endif
    launch_ntt(*s,s->stream);
#if !FHERMA_DIRECT_OUTPUT && FHERMA_PIPELINE_OUTPUT<=1
    check(cudaMemcpyAsync(s->host_output,s->input,bytes,cudaMemcpyDeviceToHost,s->stream),"capture D2H");
#endif
    cudaGraph_t graph=nullptr;
    check(cudaStreamEndCapture(s->stream,&graph),"end capture");
    auto status=cudaGraphInstantiateWithFlags(&s->graph,graph,0);
    cudaGraphDestroy(graph); check(status,"instantiate graph");
    check(cudaGraphUpload(s->graph,s->stream),"upload graph");
    check(cudaStreamSynchronize(s->stream),"graph ready");
#endif
    return s.release();
}
fherma::Outputs fherma_run(void* opaque,const fherma::Inputs& in) {
    auto& s=*static_cast<State*>(opaque); size_t words=size_t(s.n)*L, bytes=words*4;
    if(in.a.data.size()!=words || in.b.data.size()!=words) throw std::runtime_error("input size");
#if FHERMA_GRAPH
    try {
#if FHERMA_HOST_PROFILE
    using HostClock=std::chrono::steady_clock;
    auto pack_start=HostClock::now();
#endif
#if FHERMA_PIPELINE_INPUT>1
    static_assert(FHERMA_PARALLEL_COPY,"input pipeline needs the persistent copy pool");
    copy_input_pipeline<FHERMA_PIPELINE_INPUT>(s.copy,s.host_input,in.a.data.data(),in.b.data.data(),words,
        [&](size_t begin,const uint32_t* a,const uint32_t* b,size_t count) {
            check(cudaMemcpyAsync(s.input+begin,a,count*4,cudaMemcpyHostToDevice,s.stream),"pipeline A H2D");
            check(cudaMemcpyAsync(s.input+words+begin,b,count*4,cudaMemcpyHostToDevice,s.stream),"pipeline B H2D");
        });
#elif FHERMA_PARALLEL_COPY
    s.copy.inputs(s.host_input,in.a.data.data(),in.b.data.data(),bytes);
#else
    std::memcpy(s.host_input,in.a.data.data(),bytes);
    std::memcpy(s.host_input+words,in.b.data.data(),bytes);
#endif
#if FHERMA_HOST_PROFILE
    auto pack_end=HostClock::now();
    HostClock::time_point enqueued,prepared,synced,released;
#endif
    fherma::Outputs out; out.c.shape={s.n,L};
#if FHERMA_DIRECT_OUTPUT
    RegisteredInput output_registration;
    s.next_output.take(out.c.data,output_registration);
    try {
        check(cudaGraphLaunch(s.graph,s.stream),"execute graph");
        check(cudaMemcpyAsync(out.c.data.data(),s.input,bytes,cudaMemcpyDeviceToHost,s.stream),"direct output D2H");
#if FHERMA_HOST_PROFILE
        enqueued=HostClock::now();
#endif
#if FHERMA_PARALLEL_COPY
        s.next_output.prepare(words,&s.copy);
#else
        s.next_output.prepare(words);
#endif
#if FHERMA_HOST_PROFILE
        prepared=HostClock::now();
#endif
        check(cudaStreamSynchronize(s.stream),"direct output ready");
#if FHERMA_HOST_PROFILE
        synced=HostClock::now();
#endif
    } catch(...) { cudaStreamSynchronize(s.stream); throw; }
    output_registration.release();
#if FHERMA_HOST_PROFILE
    released=HostClock::now();
    auto us=[](auto start,auto end) { return std::chrono::duration<double,std::micro>(end-start).count(); };
    std::fprintf(stderr,"HOST_GRAPH_US pack=%.3f enqueue=%.3f prepare_output=%.3f wait=%.3f unregister=%.3f\n",
                 us(pack_start,pack_end),us(pack_end,enqueued),us(enqueued,prepared),us(prepared,synced),us(synced,released));
    std::fprintf(stderr,"HOST_ALLOC_US reserve=%.3f prefault=%.3f zero=%.3f register=%.3f\n",
                 s.next_output.reserve_us,s.next_output.prefault_us,s.next_output.zero_us,s.next_output.register_us);
#endif
#else
    check(cudaGraphLaunch(s.graph,s.stream),"execute graph");
#if FHERMA_PIPELINE_OUTPUT>1
    for(unsigned part=0;part<FHERMA_PIPELINE_OUTPUT;++part) {
        size_t begin=words*part/FHERMA_PIPELINE_OUTPUT,end=words*(part+1)/FHERMA_PIPELINE_OUTPUT;
        check(cudaMemcpyAsync(s.host_output+begin,s.input+begin,(end-begin)*4,cudaMemcpyDeviceToHost,s.stream),"pipeline D2H");
        check(cudaEventRecord(s.output_ready[part],s.stream),"record output segment ready");
    }
#endif
#if FHERMA_HOST_PROFILE
    enqueued=HostClock::now();
#endif
#if FHERMA_PARALLEL_COPY && FHERMA_PARALLEL_OUTPUT
    out.c.data.reserve(words);
    if(FHERMA_PREFAULT_OUTPUT) s.copy.prefault(out.c.data.data(),bytes);
    out.c.data.resize(words);
#endif
#if FHERMA_HOST_PROFILE
    prepared=HostClock::now();
#endif
#if FHERMA_PIPELINE_OUTPUT>1
    check(cudaEventSynchronize(s.output_ready[0]),"first output segment ready");
#else
    check(cudaStreamSynchronize(s.stream),"graph result ready");
#endif
#if FHERMA_HOST_PROFILE
    synced=HostClock::now();
#endif
#if FHERMA_PIPELINE_OUTPUT>1
    for(unsigned part=0;part<FHERMA_PIPELINE_OUTPUT;++part) {
        check(cudaEventSynchronize(s.output_ready[part]),"output segment ready");
        size_t begin=words*part/FHERMA_PIPELINE_OUTPUT,end=words*(part+1)/FHERMA_PIPELINE_OUTPUT;
        s.copy.output(out.c.data.data()+begin,s.host_output+begin,(end-begin)*4);
    }
#elif FHERMA_PARALLEL_COPY && FHERMA_PARALLEL_OUTPUT
    s.copy.output(out.c.data.data(),s.host_output,bytes);
#else
    out.c.data.assign(s.host_output,s.host_output+words);
#endif
#if FHERMA_HOST_PROFILE
    released=HostClock::now();
    auto us=[](auto start,auto end) { return std::chrono::duration<double,std::micro>(end-start).count(); };
    std::fprintf(stderr,"HOST_STAGED_US pack=%.3f enqueue=%.3f allocate=%.3f wait=%.3f copy=%.3f\n",
                 us(pack_start,pack_end),us(pack_end,enqueued),us(enqueued,prepared),us(prepared,synced),us(synced,released));
#endif
#endif
    return out;
    } catch(...) { cudaStreamSynchronize(s.stream); throw; }
#else
    mark(s,0);
#if FHERMA_PINNED
#if FHERMA_PROFILE
    auto pack_start=std::chrono::steady_clock::now();
#endif
    RegisteredInput registered_a,registered_b;
    bool registered=false;
    if(FHERMA_REGISTER_INPUTS && bytes>=1048576)
        registered=registered_a.pin(in.a.data.data(),bytes) && registered_b.pin(in.b.data.data(),bytes);
    if(!registered) {
        registered_a.release(); registered_b.release();
#if FHERMA_PARALLEL_COPY
        s.copy.inputs(s.host_input,in.a.data.data(),in.b.data.data(),bytes);
#else
        std::memcpy(s.host_input,in.a.data.data(),bytes);
        std::memcpy(s.host_input+words,in.b.data.data(),bytes);
#endif
    }
#if FHERMA_PROFILE
    double pack_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-pack_start).count();
#endif
    if(registered) {
        check(cudaMemcpy(s.input,in.a.data.data(),bytes,cudaMemcpyHostToDevice),"registered a H2D");
        check(cudaMemcpy(s.input+words,in.b.data.data(),bytes,cudaMemcpyHostToDevice),"registered b H2D");
    } else {
        check(cudaMemcpy(s.input,s.host_input,2*bytes,cudaMemcpyHostToDevice),"pinned inputs H2D");
    }
#else
    check(cudaMemcpy(s.input,in.a.data.data(),bytes,cudaMemcpyHostToDevice),"copy a");
    check(cudaMemcpy(s.input+words,in.b.data.data(),bytes,cudaMemcpyHostToDevice),"copy b");
#endif
    mark(s,1);
    launch_ntt(s);
    fherma::Outputs out; out.c.shape={s.n,L};
#if FHERMA_DIRECT_OUTPUT
    RegisteredInput output_registration;
    s.next_output.take(out.c.data,output_registration);
    check(cudaMemcpy(out.c.data.data(),s.input,bytes,cudaMemcpyDeviceToHost),"direct output D2H");
#if FHERMA_PARALLEL_COPY
    s.next_output.prepare(words,&s.copy);
#else
    s.next_output.prepare(words);
#endif
    output_registration.release();
    registered_a.release(); registered_b.release();
#if FHERMA_PROFILE
    double unpack_us=0; // This variant is measured without the host-copy diagnostic.
#endif
#elif FHERMA_PINNED
#if FHERMA_PARALLEL_COPY && FHERMA_PARALLEL_OUTPUT
    // Allocate while the GPU executes the already-enqueued NTT kernels.
    out.c.data.reserve(words);
    if(FHERMA_PREFAULT_OUTPUT) s.copy.prefault(out.c.data.data(),bytes);
    out.c.data.resize(words);
#endif
    check(cudaMemcpy(s.host_output,s.input,bytes,cudaMemcpyDeviceToHost),"pinned output D2H");
#if FHERMA_PROFILE
    auto unpack_start=std::chrono::steady_clock::now();
#endif
#if FHERMA_PARALLEL_COPY && FHERMA_PARALLEL_OUTPUT
    s.copy.output(out.c.data.data(),s.host_output,bytes);
#else
    out.c.data.assign(s.host_output,s.host_output+words);
#endif
#if FHERMA_PROFILE
    double unpack_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-unpack_start).count();
#endif
    // Do not retain registrations or input pointers beyond this timed call.
    registered_a.release(); registered_b.release();
#else
    out.c.data.resize(words);
    check(cudaMemcpy(out.c.data.data(),s.input,bytes,cudaMemcpyDeviceToHost),"copy output / synchronize");
#endif
    mark(s,7);
#if FHERMA_PROFILE
    check(cudaEventSynchronize(s.events[7]),"profile synchronize");
    const char* names[]={"h2d","prepare","forward","product","inverse","finish","d2h"};
    std::fprintf(stderr,"PROFILE_US");
    for(unsigned i=0;i<7;++i) {
        float ms=0; check(cudaEventElapsedTime(&ms,s.events[i],s.events[i+1]),"profile elapsed");
        std::fprintf(stderr," %s=%.3f",names[i],ms*1000);
    }
    std::fprintf(stderr,"\n");
#if FHERMA_PINNED
    std::fprintf(stderr,"HOST_US pack=%.3f unpack=%.3f\n",pack_us,unpack_us);
#endif
#endif
    return out;
#endif
}
void fherma_free(void* state) { delete static_cast<State*>(state); }
