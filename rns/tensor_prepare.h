#pragma once
#include "rns/tensor_setup.h"
#include <cublas_v2.h>
namespace rns_tensor {
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
    int32_t *products=nullptr,*corrections=nullptr;
    void* workspace=nullptr;
    static constexpr size_t WorkspaceBytes=4*1024*1024;
    ~Converter() {
        if(handle) cublasDestroy(handle);
        cudaFree(packed);cudaFree(weights);cudaFree(products);cudaFree(corrections);cudaFree(workspace);
    }
    void init(unsigned n,const std::vector<rns::SmallMod>& mods) {
        auto table=make_weights(mods);
        require_cuda(cudaMalloc(&packed,size_t(2)*n*K),"allocate tensor input");
        require_cuda(cudaMalloc(&weights,2*sizeof(table.matrix)),"allocate tensor weights");
        require_cuda(cudaMalloc(&products,size_t(2)*n*Columns*4),"allocate tensor products");
        require_cuda(cudaMalloc(&corrections,sizeof(table.correction)),"allocate tensor corrections");
        require_cuda(cudaMemcpy(weights,table.matrix.data(),sizeof(table.matrix),cudaMemcpyHostToDevice),"upload tensor weights");
        require_cuda(cudaMemcpy(weights+K*Columns,table.matrix.data(),sizeof(table.matrix),cudaMemcpyHostToDevice),"upload second tensor weights");
        require_cuda(cudaMemcpy(corrections,table.correction.data(),sizeof(table.correction),cudaMemcpyHostToDevice),"upload tensor corrections");
        require_cuda(cudaMalloc(&workspace,WorkspaceBytes),"allocate tensor workspace");
        require_blas(cublasCreate(&handle),"create cuBLAS converter");
        require_blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST),"set integer scalar pointer mode");
        bind(nullptr);
    }
    void bind(cudaStream_t stream) {
        require_blas(cublasSetStream(handle,stream),"bind tensor converter stream");
        require_blas(cublasSetWorkspace(handle,workspace,WorkspaceBytes),"set tensor converter workspace");
    }
    void launch(const uint32_t* input,uint32_t* ab,const rns::SmallMod* mods,const rns::Twiddle* twists,
                unsigned n,unsigned begin,unsigned count,cudaStream_t stream) {
        dim3 byte_grid((count*K+255)/256,2),residue_grid((count+127)/128,2*rns::PrimeCount);
        pack<<<byte_grid,256,0,stream>>>(input,packed,n,begin,count);
        int32_t alpha=1,beta=0;
        require_blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
            count,Columns,K,&alpha,packed+size_t(begin)*K,CUDA_R_8I,K,size_t(n)*K,
            weights,CUDA_R_8I,K,K*Columns,&beta,products+begin,CUDA_R_32I,n,size_t(n)*Columns,
            2,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"exact int8 residue GEMM");
        reduce<<<residue_grid,128,0,stream>>>(products,ab,corrections,mods,twists,n,begin,count);
        require_cuda(cudaGetLastError(),"tensor residue kernels");
    }
};
} // namespace rns_tensor
