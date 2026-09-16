// Exact negacyclic NTT baseline. All device modular arithmetic uses cuPQC.
#include "fherma.h"
#include "wide_host.h"
#include <cupqc/bigint.hpp>
#include <cuda_runtime.h>
#include <memory>

namespace {
constexpr unsigned L=28;
using BI=decltype(cupqc::BitWidth<L*32>()+cupqc::SM<800>()+cupqc::Thread());
using Big=typename BI::bigint;
void check(cudaError_t e,const char* op) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));
}
struct State {
    uint32_t n=0, logn=0;
    uint32_t *q=nullptr,*roots=nullptr,*twist=nullptr,*inv_twist=nullptr,*scale=nullptr;
    uint32_t *input=nullptr,*ab=nullptr,*c=nullptr;
    ~State() { cudaFree(q); cudaFree(roots); cudaFree(twist); cudaFree(inv_twist);
        cudaFree(scale); cudaFree(input); cudaFree(ab); cudaFree(c); }
};
__device__ Big pow_small(Big x,unsigned e,const Big& q) {
    Big y(uint32_t(1));
    while(e) { if(e&1) y=y.mul_mod(x,q); e>>=1; if(e) x=x.mul_mod(x,q); }
    return y;
}
__global__ void make_tables(const uint32_t* qp,const uint32_t* roots,
                           uint32_t* twist,uint32_t* inv_twist,uint32_t* scale,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const Big q(qp,0), psi(roots,0), invpsi(roots,1), invn(roots,2);
    auto a=pow_small(psi,i,q), b=pow_small(invpsi,i,q);
    a.store(twist,i); b.store(inv_twist,i); b.mul_mod(invn,q).store(scale,i);
}
__global__ void prepare(const uint32_t* input,uint32_t* ab,const uint32_t* twist,
                        const uint32_t* qp,unsigned n,unsigned logn) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    unsigned poly=blockIdx.y, j=__brev(i)>>(32-logn);
    const Big q(qp,0), x(input,poly*n+i), t(twist,i);
    x.mul_mod(t,q).store(ab,poly*n+j);
}
__global__ void stage(uint32_t* values,const uint32_t* table,const uint32_t* qp,
                      unsigned n,unsigned half) {
    unsigned k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=n/2) return;
    unsigned j=k&(half-1), i=2*(k-j)+j+blockIdx.y*n;
    const Big q(qp,0), u(values,i), v(values,i+half), tw(table,j*(n/half));
    auto t=v.mul_mod(tw,q);
    u.add_mod(t,q).store(values,i); u.sub_mod(t,q).store(values,i+half);
}
__global__ void product(const uint32_t* ab,uint32_t* c,const uint32_t* qp,
                        unsigned n,unsigned logn) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const Big q(qp,0), a(ab,i), b(ab,n+i);
    a.mul_mod(b,q).store(c,__brev(i)>>(32-logn));
}
__global__ void finish(uint32_t* c,const uint32_t* scale,const uint32_t* qp,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const Big q(qp,0), x(c,i), s(scale,i);
    x.mul_mod(s,q).store(c,i);
}
} // namespace

void* fherma_init(const fherma::Point& p) {
    if(p.N<2 || (p.N&(p.N-1)) || p.N>32768 || p.W!=868 || p.L!=L || p.q.data.size()!=L)
        throw std::runtime_error("coverage: power-of-two 2<=N<=32768, W=868, L=28");
    auto s=std::make_unique<State>(); s->n=p.N; s->logn=__builtin_ctz(p.N);
    auto root=host_wide::roots(p.q.data,p.N);
    std::vector<uint32_t> packed=root.psi;
    packed.insert(packed.end(),root.invpsi.begin(),root.invpsi.end());
    packed.insert(packed.end(),root.invn.begin(),root.invn.end());
    auto alloc=[](uint32_t** ptr,size_t coeffs) { check(cudaMalloc(ptr,coeffs*L*4),"cudaMalloc"); };
    alloc(&s->q,1); alloc(&s->roots,3); alloc(&s->twist,p.N); alloc(&s->inv_twist,p.N);
    alloc(&s->scale,p.N); alloc(&s->input,2*p.N); alloc(&s->ab,2*p.N); alloc(&s->c,p.N);
    check(cudaMemcpy(s->q,p.q.data.data(),L*4,cudaMemcpyHostToDevice),"copy q");
    check(cudaMemcpy(s->roots,packed.data(),3*L*4,cudaMemcpyHostToDevice),"copy roots");
    make_tables<<<(p.N+127)/128,128>>>(s->q,s->roots,s->twist,s->inv_twist,s->scale,p.N);
    check(cudaGetLastError(),"table launch"); check(cudaDeviceSynchronize(),"table setup");
    return s.release();
}
fherma::Outputs fherma_run(void* opaque,const fherma::Inputs& in) {
    auto& s=*static_cast<State*>(opaque); size_t words=size_t(s.n)*L, bytes=words*4;
    if(in.a.data.size()!=words || in.b.data.size()!=words) throw std::runtime_error("input size");
    check(cudaMemcpy(s.input,in.a.data.data(),bytes,cudaMemcpyHostToDevice),"copy a");
    check(cudaMemcpy(s.input+words,in.b.data.data(),bytes,cudaMemcpyHostToDevice),"copy b");
    dim3 full((s.n+127)/128,2), halves((s.n/2+127)/128,2);
    prepare<<<full,128>>>(s.input,s.ab,s.twist,s.q,s.n,s.logn);
    for(unsigned half=1;half<s.n;half*=2) stage<<<halves,128>>>(s.ab,s.twist,s.q,s.n,half);
    product<<<full.x,128>>>(s.ab,s.c,s.q,s.n,s.logn);
    for(unsigned half=1;half<s.n;half*=2) stage<<<halves.x,128>>>(s.c,s.inv_twist,s.q,s.n,half);
    finish<<<full.x,128>>>(s.c,s.scale,s.q,s.n);
    check(cudaGetLastError(),"NTT launch");
    fherma::Outputs out; out.c.shape={s.n,L}; out.c.data.resize(words);
    check(cudaMemcpy(out.c.data.data(),s.c,bytes,cudaMemcpyDeviceToHost),"copy output / synchronize");
    return out;
}
void fherma_free(void* state) { delete static_cast<State*>(state); }
