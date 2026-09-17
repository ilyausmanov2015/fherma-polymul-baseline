#ifndef FHERMA_MONTGOMERY
#define FHERMA_MONTGOMERY 0
#endif
#ifndef FHERMA_SOA
#define FHERMA_SOA 1
#endif
#ifndef FHERMA_PROFILE
#define FHERMA_PROFILE 1
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

// Exact negacyclic NTT baseline. All device modular arithmetic uses cuPQC.
#include "fherma.h"
#include "wide_host.h"
#include <cupqc/bigint.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <cstdio>
#include <cstring>

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
// For q = 2^868-c, c < 2^28, a full product folds twice without division.
// At each fold all intermediates fit the 896-bit cuPQC storage width.
// The modulus shape is checked once on the host during setup.
__device__ Big multiply(const Big& a,const Big& b,const Big& q) {
    const uint32_t c=uint32_t(0)-q[0];
    auto p=a.mul_wide(b);
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
void check(cudaError_t e,const char* op) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));
}
struct State {
    uint32_t n=0, logn=0;
    uint32_t *q=nullptr,*roots=nullptr,*twist=nullptr,*inv_twist=nullptr,*scale=nullptr;
    uint32_t *input=nullptr,*ab=nullptr,*c=nullptr;
#if FHERMA_PINNED
    uint32_t *host_input=nullptr,*host_output=nullptr;
#endif
#if FHERMA_PROFILE
    cudaEvent_t events[8]{};
#endif
    ~State() { cudaFree(q); cudaFree(roots); cudaFree(twist); cudaFree(inv_twist);
        cudaFree(scale); cudaFree(input); cudaFree(ab); cudaFree(c);
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
    const Mod q(qp,0);
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
    const Mod q(qp,0);
    const Big x=encode(Big(input,poly*n+i),q), t=load_coeff(twist,i,n);
    store_coeff(multiply(x,t,q),ab+poly*n*L,j,n);
}
__global__ void stage(uint32_t* values,const uint32_t* table,const uint32_t* qp,
                      unsigned n,unsigned half) {
    unsigned k=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(k>=n/2) return;
    unsigned j=k&(half-1), i=2*(k-j)+j;
    values+=blockIdx.y*n*L;
    const Mod q(qp,0);
    const Big u=load_coeff(values,i,n), v=load_coeff(values,i+half,n), tw=load_coeff(table,j*(n/half),n);
    auto t=multiply(v,tw,q);
    store_coeff(u.add_mod(t,q),values,i,n); store_coeff(u.sub_mod(t,q),values,i+half,n);
}
// Keep the first eight radix-2 stages in shared memory. Every block handles
// an independent contiguous tile; no inter-block synchronization is needed.
__global__ void small_stages(uint32_t* values,const uint32_t* table,const uint32_t* qp,unsigned n) {
    __shared__ uint32_t tile[256*L];
    unsigned t=threadIdx.x/FHERMA_TPI, base=blockIdx.x*256;
    values+=blockIdx.y*n*L;
    const Mod q(qp,0);
    store_coeff(load_coeff(values,base+2*t,n),tile,2*t,256);
    store_coeff(load_coeff(values,base+2*t+1,n),tile,2*t+1,256);
    __syncthreads();
    for(unsigned half=1;half<256;half*=2) {
        unsigned j=t&(half-1), i=2*(t-j)+j;
        const Big u=load_coeff(tile,i,256), v=load_coeff(tile,i+half,256);
        const Big tw=load_coeff(table,j*(n/half),n);
        const Big m=multiply(v,tw,q);
        store_coeff(u.add_mod(m,q),tile,i,256);
        store_coeff(u.sub_mod(m,q),tile,i+half,256);
        __syncthreads();
    }
    store_coeff(load_coeff(tile,2*t,256),values,base+2*t,n);
    store_coeff(load_coeff(tile,2*t+1,256),values,base+2*t+1,n);
}
__global__ void product(const uint32_t* ab,uint32_t* c,const uint32_t* qp,
                        unsigned n,unsigned logn) {
    unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(i>=n) return;
    const Mod q(qp,0);
    const Big a=load_coeff(ab,i,n), b=load_coeff(ab+n*L,i,n);
    store_coeff(multiply(a,b,q),c,__brev(i)>>(32-logn),n);
}
__global__ void finish(const uint32_t* c,uint32_t* out,const uint32_t* scale,const uint32_t* qp,unsigned n) {
    unsigned i=(blockIdx.x*blockDim.x+threadIdx.x)/FHERMA_TPI;
    if(i>=n) return;
    const Mod q(qp,0);
    const Big x=load_coeff(c,i,n), s=load_coeff(scale,i,n);
    decode(multiply(x,s,q),q).store(out,i);
}
} // namespace

void* fherma_init(const fherma::Point& p) {
    if(p.N<2 || (p.N&(p.N-1)) || p.N>32768 || p.W!=868 || p.L!=L || p.q.data.size()!=L)
        throw std::runtime_error("coverage: power-of-two 2<=N<=32768, W=868, L=28");
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
    check(cudaMemcpy(s->q,p.q.data.data(),L*4,cudaMemcpyHostToDevice),"copy q");
    check(cudaMemcpy(s->roots,packed.data(),3*L*4,cudaMemcpyHostToDevice),"copy roots");
    make_tables<<<(p.N*FHERMA_TPI+127)/128,128>>>(s->q,s->roots,s->twist,s->inv_twist,s->scale,p.N);
    check(cudaGetLastError(),"table launch"); check(cudaDeviceSynchronize(),"table setup");
#if FHERMA_PROFILE
    for(auto& e:s->events) check(cudaEventCreate(&e),"profile create");
#endif
    return s.release();
}
fherma::Outputs fherma_run(void* opaque,const fherma::Inputs& in) {
    auto& s=*static_cast<State*>(opaque); size_t words=size_t(s.n)*L, bytes=words*4;
    if(in.a.data.size()!=words || in.b.data.size()!=words) throw std::runtime_error("input size");
    mark(s,0);
#if FHERMA_PINNED
    std::memcpy(s.host_input,in.a.data.data(),bytes);
    std::memcpy(s.host_input+words,in.b.data.data(),bytes);
    check(cudaMemcpy(s.input,s.host_input,2*bytes,cudaMemcpyHostToDevice),"pinned inputs H2D");
#else
    check(cudaMemcpy(s.input,in.a.data.data(),bytes,cudaMemcpyHostToDevice),"copy a");
    check(cudaMemcpy(s.input+words,in.b.data.data(),bytes,cudaMemcpyHostToDevice),"copy b");
#endif
    mark(s,1);
    dim3 full((s.n*FHERMA_TPI+127)/128,2), halves((s.n/2*FHERMA_TPI+127)/128,2);
    prepare<<<full,128>>>(s.input,s.ab,s.twist,s.q,s.n,s.logn);
    mark(s,2);
    unsigned first=1;
    if(FHERMA_FUSED_SMALL && s.n>=256) {
        dim3 tiles(s.n/256,2);
        small_stages<<<tiles,128*FHERMA_TPI>>>(s.ab,s.twist,s.q,s.n);
        first=256;
    }
    for(unsigned half=first;half<s.n;half*=2) stage<<<halves,128>>>(s.ab,s.twist,s.q,s.n,half);
    mark(s,3);
    product<<<full.x,128>>>(s.ab,s.c,s.q,s.n,s.logn);
    mark(s,4);
    if(FHERMA_FUSED_SMALL && s.n>=256) {
        unsigned tiles=s.n/256;
        small_stages<<<tiles,128*FHERMA_TPI>>>(s.c,s.inv_twist,s.q,s.n);
    }
    for(unsigned half=first;half<s.n;half*=2) stage<<<halves.x,128>>>(s.c,s.inv_twist,s.q,s.n,half);
    mark(s,5);
    finish<<<full.x,128>>>(s.c,s.input,s.scale,s.q,s.n);
    mark(s,6);
    check(cudaGetLastError(),"NTT launch");
    fherma::Outputs out; out.c.shape={s.n,L};
#if FHERMA_PINNED
    check(cudaMemcpy(s.host_output,s.input,bytes,cudaMemcpyDeviceToHost),"pinned output D2H");
    out.c.data.assign(s.host_output,s.host_output+words);
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
#endif
    return out;
}
void fherma_free(void* state) { delete static_cast<State*>(state); }
