#pragma once
#include "rns/tensor_setup.h"
#include <cublas_v2.h>
#include <mma.h>
#ifndef FHERMA_TENSOR_WMMA
#define FHERMA_TENSOR_WMMA 1
#endif
namespace rns_tensor {
__global__ void sum_rows(const uint32_t* input,int32_t* sums,unsigned n,unsigned begin,unsigned count) {
    unsigned t=blockIdx.x*blockDim.x+threadIdx.x,coefficient=begin+t/4,lane=t&3;
    if(t>=count*4) return;
    unsigned poly=blockIdx.y;int sum=0;
    #pragma unroll
    for(unsigned limb=lane;limb<rns::AbiWords;limb+=4)
        sum=__dp4a(int(input[(poly*n+coefficient)*rns::AbiWords+limb]^0x80808080u),0x01010101,sum);
    sum+=__shfl_xor_sync(0xffffffffu,sum,1,4);
    sum+=__shfl_xor_sync(0xffffffffu,sum,2,4);
    if(lane==0) sums[poly*n+coefficient]=sum;
}
__global__ void matrix_residues(const uint32_t* input,const int8_t* weights,const int32_t* sums,
                                uint32_t* ab,const int32_t* corrections,const rns::SmallMod* mods,
                                const rns::Twiddle* twists,unsigned n,unsigned begin) {
    namespace wmma=nvcuda::wmma;
    __shared__ __align__(32) uint32_t shared_a[4*64],shared_b[4*64];
    __shared__ __align__(32) int shared_c[4*16*24];
    constexpr unsigned Groups=(rns::PrimeCount+3)/4;
    unsigned warp=threadIdx.x/32,lane=threadIdx.x&31;
    unsigned poly=blockIdx.y/Groups,group=blockIdx.y%Groups;
    unsigned base=begin+blockIdx.x*64+warp*16;
    wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,int> c;
    wmma::fill_fragment(c,0);
    #pragma unroll
    for(unsigned k=0;k<RealK;k+=16) {
        #pragma unroll
        for(unsigned word=lane;word<64;word+=32) {
            unsigned row=word/4,inner=word%4;
            shared_a[warp*64+word]=input[(poly*n+base+row)*rns::AbiWords+k/4+inner]^0x80808080u;
            shared_b[warp*64+word]=reinterpret_cast<const uint32_t*>(weights)[(group*16+row)*(K/4)+k/4+inner];
        }
        __syncwarp();
        wmma::load_matrix_sync(a,reinterpret_cast<const signed char*>(shared_a+warp*64),16);
        wmma::load_matrix_sync(b,reinterpret_cast<const signed char*>(shared_b+warp*64),16);
        wmma::mma_sync(c,a,b,c);
        __syncwarp();
    }
    wmma::store_matrix_sync(shared_c+warp*16*24,c,24,wmma::mem_row_major);
    __syncthreads();
    unsigned row=lane&15,i=base+row;int32_t row_sum=sums[poly*n+i];
    #pragma unroll
    for(unsigned within=lane/16;within<4;within+=2) {
        unsigned pi=group*4+within;
        if(pi<rns::PrimeCount) {
            uint64_t exact=0;
            #pragma unroll
            for(unsigned byte=0;byte<4;++byte) {
                int32_t recovered=shared_c[(warp*16+row)*24+within*4+byte]+128*row_sum+corrections[pi*4+byte];
                exact+=uint64_t(recovered)<<(8*byte);
            }
            auto modulus=mods[pi];
            uint64_t quotient=__umul64hi(exact,modulus.reciprocal);
            uint32_t value=uint32_t(exact-quotient*modulus.p);
            if(value>=modulus.p) value-=modulus.p;
            auto twist=twists[pi*n+i];uint32_t qhat=__umulhi(value,twist.shoup);
            value=value*twist.value-qhat*modulus.p;
            if(value>=modulus.p) value-=modulus.p;
            ab[(poly*rns::PrimeCount+pi)*n+i]=value;
        }
    }
}
__global__ void pack(const uint32_t* input,int8_t* packed,unsigned n,unsigned begin,unsigned count) {
    unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count*K) return;
    unsigned coefficient=begin+index/K,byte=index%K,poly=blockIdx.y;
    const auto* source=reinterpret_cast<const uint8_t*>(input)+(poly*n+coefficient)*RealK;
    packed[(poly*n+coefficient)*K+byte]=byte<RealK ? int8_t(int(source[byte])-128) : int8_t(0);
}
__global__ void reduce(const int32_t* products,uint32_t* ab,const int32_t* corrections,
                        const rns::SmallMod* mods,const rns::Twiddle* twists,
                        unsigned n,unsigned begin,unsigned count) {
    unsigned i=begin+blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=begin+count) return;
    unsigned pi=blockIdx.y%rns::PrimeCount,poly=blockIdx.y/rns::PrimeCount;
    const auto* dot=products+poly*n*Columns+i;
    int32_t row_sum=dot[SumColumn*n];uint64_t exact=0;
    #pragma unroll
    for(unsigned byte=0;byte<4;++byte) {
        unsigned col=4*pi+byte;
        int32_t recovered=dot[col*n]+128*row_sum+corrections[col];
        exact+=uint64_t(recovered)<<(8*byte);
    }
    auto modulus=mods[pi];
    uint64_t quotient=__umul64hi(exact,modulus.reciprocal);
    uint32_t value=uint32_t(exact-quotient*modulus.p);
    if(value>=modulus.p) value-=modulus.p;
    auto twist=twists[pi*n+i];uint32_t qhat=__umulhi(value,twist.shoup);
    value=value*twist.value-qhat*modulus.p;
    if(value>=modulus.p) value-=modulus.p;
    ab[blockIdx.y*n+i]=value;
}
inline void require_cuda(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
inline void require_blas(cublasStatus_t status,const char* operation) {
    if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(operation)+": "+std::to_string(int(status)));
}
struct Converter {
    cublasHandle_t handle=nullptr;
    int8_t *packed=nullptr,*weights=nullptr;
    int32_t *products=nullptr,*corrections=nullptr,*row_sums=nullptr;
    void* workspace=nullptr;
    static constexpr size_t WorkspaceBytes=4*1024*1024;
    ~Converter() {
        if(handle) cublasDestroy(handle);
        cudaFree(packed);cudaFree(weights);cudaFree(products);cudaFree(corrections);cudaFree(workspace);cudaFree(row_sums);
    }
    void init(unsigned n,const std::vector<rns::SmallMod>& mods) {
        auto table=make_weights(mods);
        require_cuda(cudaMalloc(&weights,2*sizeof(table.matrix)),"allocate tensor weights");
        require_cuda(cudaMalloc(&corrections,sizeof(table.correction)),"allocate tensor corrections");
        require_cuda(cudaMemcpy(weights,table.matrix.data(),sizeof(table.matrix),cudaMemcpyHostToDevice),"upload tensor weights");
        require_cuda(cudaMemcpy(weights+K*Columns,table.matrix.data(),sizeof(table.matrix),cudaMemcpyHostToDevice),"upload second tensor weights");
        require_cuda(cudaMemcpy(corrections,table.correction.data(),sizeof(table.correction),cudaMemcpyHostToDevice),"upload tensor corrections");
#if FHERMA_TENSOR_WMMA
        require_cuda(cudaMalloc(&row_sums,size_t(2)*n*4),"allocate tensor row sums");
#else
        require_cuda(cudaMalloc(&packed,size_t(2)*n*K),"allocate tensor input");
        require_cuda(cudaMalloc(&products,size_t(2)*n*Columns*4),"allocate tensor products");
        require_cuda(cudaMalloc(&workspace,WorkspaceBytes),"allocate tensor workspace");
        require_blas(cublasCreate(&handle),"create cuBLAS converter");
        require_blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST),"set integer scalar pointer mode");
        bind(nullptr);
#endif
    }
    void bind(cudaStream_t stream) {
        if(!handle) return;
        require_blas(cublasSetStream(handle,stream),"bind tensor converter stream");
        require_blas(cublasSetWorkspace(handle,workspace,WorkspaceBytes),"set tensor converter workspace");
    }
    void launch(const uint32_t* input,uint32_t* ab,const rns::SmallMod* mods,const rns::Twiddle* twists,
                unsigned n,unsigned begin,unsigned count,cudaStream_t stream) {
#if FHERMA_TENSOR_WMMA
        if(count%64) throw std::runtime_error("WMMA conversion requires chunks divisible by 64");
        dim3 sums_grid(count*4/256,2),matrix_grid(count/64,2*((rns::PrimeCount+3)/4));
        sum_rows<<<sums_grid,256,0,stream>>>(input,row_sums,n,begin,count);
        matrix_residues<<<matrix_grid,128,0,stream>>>(input,weights,row_sums,ab,corrections,mods,twists,n,begin);
#else
        dim3 byte_grid((count*K+255)/256,2),residue_grid((count+127)/128,2*rns::PrimeCount);
        pack<<<byte_grid,256,0,stream>>>(input,packed,n,begin,count);
        int32_t alpha=1,beta=0;
        require_blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
            count,Columns,K,&alpha,packed+size_t(begin)*K,CUDA_R_8I,K,size_t(n)*K,
            weights,CUDA_R_8I,K,K*Columns,&beta,products+begin,CUDA_R_32I,n,size_t(n)*Columns,
            2,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"exact int8 residue GEMM");
        reduce<<<residue_grid,128,0,stream>>>(products,ab,corrections,mods,twists,n,begin,count);
#endif
        require_cuda(cudaGetLastError(),"tensor residue kernels");
    }
};
} // namespace rns_tensor
