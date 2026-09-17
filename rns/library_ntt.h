#pragma once
// Optional cuPQC staged NTT backend. All tables are parameter-only.
#include <cupqc/ntt.hpp>
namespace rns_library {
#ifndef FHERMA_LIBRARY_FUSED
#define FHERMA_LIBRARY_FUSED 1
#endif
constexpr unsigned N=32768,M=256,K=N/M,Threads=128;
using Forward=decltype(cupqc::Algorithm<cupqc::algorithm::NTT>()+
    cupqc::Direction<cupqc::nttDirection::FORWARD>()+cupqc::Precision<uint32_t>()+
    cupqc::Size<N>()+cupqc::SubSize<M>()+cupqc::Block()+cupqc::BlockDim<Threads>());
using Inverse=decltype(cupqc::Algorithm<cupqc::algorithm::NTT>()+
    cupqc::Direction<cupqc::nttDirection::INVERSE>()+cupqc::Precision<uint32_t>()+
    cupqc::Size<N>()+cupqc::SubSize<M>()+cupqc::Block()+cupqc::BlockDim<Threads>());
using Scheme=cupqc::nttConst<uint32_t>;
__global__ void make_tables(uint32_t* forward,uint32_t* inverse,const rns::SmallMod* mods,
                             const rns::Twiddle* twist,const rns::Twiddle* inverse_twist) {
    unsigned pi=blockIdx.y,p=mods[pi].p;
    Forward().make_twiddles(forward+pi*N,p,twist[pi*N+2].value);
    Inverse().make_twiddles(inverse+pi*N,p,inverse_twist[pi*N+2].value);
}
__global__ void convert_tables(uint32_t* forward,uint32_t* inverse,const rns::SmallMod* mods) {
    unsigned pi=blockIdx.y,p=mods[pi].p;
    Forward().transform_twiddles_to_mont(forward+pi*N,p);
    Inverse().transform_twiddles_to_mont(inverse+pi*N,p);
}
__global__ void forward_first(uint32_t* values,const uint32_t* twiddles,const Scheme* schemes) {
    __shared__ uint32_t tile[K];
    unsigned pi=blockIdx.y%rns::PrimeCount;values+=blockIdx.y*N;
    auto scheme=schemes[pi];
    Forward().stage_1_load_to_mont(tile,values,blockIdx.x,scheme);
    __syncthreads();
    Forward().stage_1_execute(tile,twiddles+pi*N,scheme.p);
    __syncthreads();
    Forward().stage_1_store(tile,values,blockIdx.x);
}
__global__ void forward_second(uint32_t* values,const uint32_t* twiddles,const Scheme* schemes) {
    __shared__ uint32_t tile[M];
    unsigned pi=blockIdx.y%rns::PrimeCount;values+=blockIdx.y*N;
    auto scheme=schemes[pi];
    Forward().stage_2_load(tile,values,blockIdx.x);
    __syncthreads();
    Forward().stage_2_execute(tile,twiddles+pi*N,scheme.p);
    __syncthreads();
    Forward().stage_2_store_from_mont(tile,values,blockIdx.x,scheme);
}
__global__ void product(const uint32_t* ab,uint32_t* c,const rns::SmallMod* mods) {
    unsigned pi=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    auto modulus=mods[pi];
    uint64_t v=uint64_t(ab[pi*N+i])*ab[(pi+rns::PrimeCount)*N+i];
    uint64_t quotient=__umul64hi(v,modulus.reciprocal);
    uint32_t result=uint32_t(v-quotient*modulus.p);
    c[pi*N+i]=result>=modulus.p ? result-modulus.p : result;
}
__global__ void inverse_first(uint32_t* values,const uint32_t* twiddles,const Scheme* schemes) {
    __shared__ uint32_t tile[M];
    unsigned pi=blockIdx.y;values+=pi*N;
    auto scheme=schemes[pi];
    Inverse().stage_1_load_to_mont(tile,values,blockIdx.x,scheme);
    __syncthreads();
    Inverse().stage_1_execute(tile,twiddles+pi*N,scheme.p);
    __syncthreads();
    Inverse().stage_1_store(tile,values,blockIdx.x);
}
__global__ void inverse_second(uint32_t* values,const uint32_t* twiddles,const Scheme* schemes) {
    __shared__ uint32_t tile[K];
    unsigned pi=blockIdx.y;values+=pi*N;
    auto scheme=schemes[pi];
    Inverse().stage_2_load(tile,values,blockIdx.x);
    __syncthreads();
    // The CRT scale already contains N^-1, so leave this inverse unscaled.
    Inverse().stage_2_execute(tile,twiddles+pi*N,scheme.p,uint32_t(1));
    __syncthreads();
    Inverse().stage_2_store_from_mont(tile,values,blockIdx.x,scheme);
}
// The last forward and first inverse stage use the same M-element spectral
// group. Keep both operands and their pointwise product in shared memory.
__global__ void fused_middle(const uint32_t* ab,uint32_t* c,const uint32_t* forward,
                              const uint32_t* inverse,const Scheme* schemes) {
    __shared__ uint32_t a[M],b[M];
    unsigned pi=blockIdx.y;auto scheme=schemes[pi];
    Forward().stage_2_load(a,ab+pi*N,blockIdx.x);
    Forward().stage_2_load(b,ab+(pi+rns::PrimeCount)*N,blockIdx.x);
    __syncthreads();
    Forward().stage_2_execute(a,forward+pi*N,scheme.p);
    __syncthreads();
    Forward().stage_2_execute(b,forward+pi*N,scheme.p);
    __syncthreads();
    for(unsigned i=threadIdx.x;i<M;i+=Threads) {
        uint32_t x=a[i],y=b[i];
        // Accommodate lazy residues before the exact Montgomery product.
        if(x>=scheme.p) x-=scheme.p;if(x>=scheme.p) x-=scheme.p;
        if(y>=scheme.p) y-=scheme.p;if(y>=scheme.p) y-=scheme.p;
        uint64_t product=uint64_t(x)*y;
        uint32_t factor=uint32_t(product)*scheme.n_dash;
        uint32_t value=uint32_t((product+uint64_t(factor)*scheme.p)>>32);
        a[i]=value>=scheme.p ? value-scheme.p : value;
    }
    __syncthreads();
    Inverse().stage_1_execute(a,inverse+pi*N,scheme.p);
    __syncthreads();
    Inverse().stage_1_store(a,c+pi*N,blockIdx.x);
}
inline void require(cudaError_t error,const char* operation) {
    if(error!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
struct Tables {
    uint32_t *forward=nullptr,*inverse=nullptr;
    Scheme* schemes=nullptr;
    ~Tables() {cudaFree(forward);cudaFree(inverse);cudaFree(schemes);}
    void init(const std::vector<rns::SmallMod>& host_mods,const rns::SmallMod* mods,
              const rns::Twiddle* twist,const rns::Twiddle* inverse_twist) {
        std::vector<Scheme> parameters;
        for(auto modulus:host_mods) parameters.emplace_back(modulus.p);
        require(cudaMalloc(&schemes,parameters.size()*sizeof(Scheme)),"allocate cuPQC NTT schemes");
        require(cudaMemcpy(schemes,parameters.data(),parameters.size()*sizeof(Scheme),cudaMemcpyHostToDevice),"upload cuPQC NTT schemes");
        require(cudaMalloc(&forward,size_t(rns::PrimeCount)*N*4),"allocate cuPQC forward table");
        require(cudaMalloc(&inverse,size_t(rns::PrimeCount)*N*4),"allocate cuPQC inverse table");
        dim3 grid(1,rns::PrimeCount);
        make_tables<<<grid,1>>>(forward,inverse,mods,twist,inverse_twist);
        require(cudaGetLastError(),"make cuPQC NTT tables");
        convert_tables<<<grid,Threads>>>(forward,inverse,mods);
        require(cudaGetLastError(),"convert cuPQC NTT tables");
        require(cudaDeviceSynchronize(),"cuPQC NTT tables ready");
    }
    void launch(uint32_t* ab,uint32_t* c,const rns::SmallMod* mods,cudaStream_t stream) {
        dim3 first(M,2*rns::PrimeCount),second(K,2*rns::PrimeCount);
        dim3 products(N/Threads,rns::PrimeCount),inverse1(K,rns::PrimeCount),inverse2(M,rns::PrimeCount);
        forward_first<<<first,Threads,0,stream>>>(ab,forward,schemes);
#if FHERMA_LIBRARY_FUSED
        fused_middle<<<inverse1,Threads,0,stream>>>(ab,c,forward,inverse,schemes);
#else
        forward_second<<<second,Threads,0,stream>>>(ab,forward,schemes);
        product<<<products,Threads,0,stream>>>(ab,c,mods);
        inverse_first<<<inverse1,Threads,0,stream>>>(c,inverse,schemes);
#endif
        inverse_second<<<inverse2,Threads,0,stream>>>(c,inverse,schemes);
    }
};
} // namespace rns_library
