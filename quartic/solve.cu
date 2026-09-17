#ifndef FHERMA_OUTPUT_SPARE
#define FHERMA_OUTPUT_SPARE 1
#endif
#ifndef FHERMA_OUTPUT_APPEND
#define FHERMA_OUTPUT_APPEND 0
#endif
#ifndef FHERMA_PAIRED_INPUT
#define FHERMA_PAIRED_INPUT 1
#endif
#ifndef FHERMA_HOST_PROFILE
#define FHERMA_HOST_PROFILE 0
#endif
#ifndef FHERMA_HUGE_OUTPUT
#define FHERMA_HUGE_OUTPUT 0
#endif
#ifndef FHERMA_FIRST_CPU
#define FHERMA_FIRST_CPU 1
#endif
#ifndef FHERMA_MAIN_OUTPUT
#define FHERMA_MAIN_OUTPUT 1
#endif
#ifndef FHERMA_DIF_FORWARD
#define FHERMA_DIF_FORWARD 1
#endif
#ifndef FHERMA_COPY_ACKS
#define FHERMA_COPY_ACKS 1
#endif
#ifndef FHERMA_SHUFFLE_TAIL
#define FHERMA_SHUFFLE_TAIL 1
#endif
#ifndef FHERMA_FUSED_PRODUCT
#define FHERMA_FUSED_PRODUCT 1
#endif
#ifndef FHERMA_SHARED_SWIZZLE
#define FHERMA_SHARED_SWIZZLE 1
#endif
#ifndef FHERMA_RNS_RADIX8
#define FHERMA_RNS_RADIX8 1
#endif
#ifndef FHERMA_MAPPED_OUTPUT
#define FHERMA_MAPPED_OUTPUT 0
#endif
#ifndef FHERMA_MAPPED_INPUT
#define FHERMA_MAPPED_INPUT 0
#endif
#ifndef FHERMA_QUARTIC_COMPACT_SCALE
#define FHERMA_QUARTIC_COMPACT_SCALE 0
#endif
#ifndef FHERMA_OVERLAP_PREPARE
#define FHERMA_OVERLAP_PREPARE 1
#endif
#ifndef FHERMA_CRT_TILED_OUTPUT
#define FHERMA_CRT_TILED_OUTPUT 1
#endif
#ifndef FHERMA_RNS_TILED_PREPARE
#define FHERMA_RNS_TILED_PREPARE 1
#endif
#ifndef FHERMA_INPUT_WC
#define FHERMA_INPUT_WC 0
#endif
#ifndef FHERMA_STREAM_OUTPUT
#define FHERMA_STREAM_OUTPUT 0
#endif
#ifndef FHERMA_RNS_FUSED_TRANSPOSE
#define FHERMA_RNS_FUSED_TRANSPOSE 1
#endif
#ifndef FHERMA_PIPELINE_INPUT
#define FHERMA_PIPELINE_INPUT 4
#endif
#ifndef FHERMA_PIPELINE_OUTPUT
#define FHERMA_PIPELINE_OUTPUT 4
#endif
#ifndef FHERMA_RNS_RADIX4
#define FHERMA_RNS_RADIX4 1
#endif
#ifndef FHERMA_RNS_GROUPED
#define FHERMA_RNS_GROUPED 1
#endif
#ifndef FHERMA_CRT_PARTS
#define FHERMA_CRT_PARTS 4
#endif
#ifndef FHERMA_RNS_TAIL
#define FHERMA_RNS_TAIL 1
#endif
#ifndef FHERMA_PROFILE
#define FHERMA_PROFILE 0
#endif
#ifndef FHERMA_GRAPH
#define FHERMA_GRAPH 1
#endif
#ifndef FHERMA_COPY_THREADS
#define FHERMA_COPY_THREADS 16
#endif
#ifndef FHERMA_OUTPUT_THREADS
#define FHERMA_OUTPUT_THREADS (FHERMA_COPY_THREADS<8 ? FHERMA_COPY_THREADS : 8)
#endif
#ifndef FHERMA_SPIN_COPY
#define FHERMA_SPIN_COPY 1
#endif
#ifndef FHERMA_STREAM_COPY
#define FHERMA_STREAM_COPY 1
#endif
#include "fherma.h"
#include "quartic/host_setup.h"
#include <cupqc/bigint.hpp>
#include <cuda_runtime.h>
#include "host_affinity.h"
#include "host_copy_pool.h"
#include "host_output_memory.h"
#include "input_pipeline.h"
#include <memory>
#include <cstring>
#include <chrono>

namespace {
using namespace quartic;
using WideBI=decltype(cupqc::BitWidth<WideWords*32>()+cupqc::SM<800>()+cupqc::Thread());
using AbiBI=decltype(cupqc::BitWidth<AbiWords*32>()+cupqc::SM<800>()+cupqc::Thread());
constexpr unsigned CrtOutputs=128/FHERMA_CRT_PARTS;
static_assert(FHERMA_CRT_PARTS==4,"quartic CRT uses one warp per component");
using Wide=typename WideBI::bigint;
using Big=typename AbiBI::bigint;
__device__ uint32_t add_mod(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t sum=a+b;return sum>=p ? sum-p : sum;
}
__device__ uint32_t sub_mod(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t difference=a-b;return a<b ? difference+p : difference;
}
__device__ uint32_t shoup(uint32_t a,Twiddle w,uint32_t p) {
    uint32_t quotient=__umulhi(a,w.shoup);
    uint32_t result=a*w.value-quotient*p;
    return result>=p ? result-p : result;
}
__device__ uint32_t multiply_mod(uint32_t a,uint32_t b,const SmallMod& modulus) {
    uint64_t product=uint64_t(a)*b;
    uint64_t quotient=__umul64hi(product,modulus.reciprocal);
    uint32_t result=uint32_t(product-quotient*modulus.p);
    return result>=modulus.p ? result-modulus.p : result;
}
template<bool Lazy> __device__ inline uint32_t ntt_add(uint32_t a,uint32_t b,uint32_t p) {
    return add_mod(a,b,Lazy ? 2*p : p);
}
template<bool Lazy> __device__ inline uint32_t ntt_sub(uint32_t a,uint32_t b,uint32_t p) {
    return sub_mod(a,b,Lazy ? 2*p : p);
}
template<bool Lazy> __device__ inline uint32_t ntt_shoup(uint32_t a,Twiddle w,uint32_t p) {
    uint32_t result=a*w.value-__umulhi(a,w.shoup)*p;
    if constexpr(Lazy) return result; // 0 <= result < 2p; 4p fits uint32_t.
    return result>=p ? result-p : result;
}
// ABI coefficients are AoS. Transpose once so all 16 residue transforms
// read a limb plane in contiguous warp-wide transactions.
__global__ void transpose_inputs(const uint32_t* input,uint32_t* output,unsigned n,unsigned begin=0,unsigned count=0,bool packed=false) {
    __shared__ uint32_t tile[32*33];
    unsigned x=threadIdx.x&31,y=threadIdx.x>>5,base=begin+blockIdx.x*32,end=count ? begin+count : n;
    input+=blockIdx.y*(packed ? count : n)*AbiWords;output+=blockIdx.y*n*AbiWords;
    for(unsigned dy=0;dy<32;dy+=8)
        if(x<AbiWords && base+y+dy<end) tile[(y+dy)*33+x]=input[(base+y+dy-(packed ? begin : 0))*AbiWords+x];
    __syncthreads();
    for(unsigned dy=0;dy<32;dy+=8)
        if(y+dy<AbiWords && base+x<end) output[(y+dy)*n+base+x]=tile[x*33+y+dy];
}
__device__ inline void prepare_coefficient(const uint32_t* input,unsigned stride,uint32_t* ab,
    const Twiddle* twists,const Twiddle* powers,const Roots* roots,unsigned n,unsigned logn,
    unsigned i,unsigned pi,unsigned poly,uint32_t p,bool natural) {
    uint32_t z[4];
    #pragma unroll
    for(unsigned component=0;component<4;++component) {
        unsigned shift=PartBits*component,base=shift/32,bits=shift%32;
        uint32_t even=0,odd=0;
        #pragma unroll
        for(unsigned limb=0;limb<PartWords;++limb) {
            uint32_t word=input[(base+limb)*stride]>>bits;
            if(bits) word|=input[(base+limb+1)*stride]<<(32-bits);
            if(limb==PartWords-1) word&=(uint32_t(1)<<25)-1;
            uint32_t term=shoup(word,powers[pi*PartWords+limb],p);
            if(limb&1) odd=add_mod(odd,term,p);else even=add_mod(even,term,p);
        }
        z[component]=shoup(add_mod(even,odd,p),roots[pi].powers[component],p);
    }
    uint32_t u=add_mod(z[0],z[2],p),v=sub_mod(z[0],z[2],p);
    uint32_t sum=add_mod(z[1],z[3],p),difference=shoup(sub_mod(z[1],z[3],p),roots[pi].i,p);
    uint32_t mixed[4]={add_mod(u,sum,p),add_mod(v,difference,p),sub_mod(u,sum,p),sub_mod(v,difference,p)};
    unsigned output_i=natural ? i : (__brev(i)>>(32-logn));
    #pragma unroll
    for(unsigned channel=0;channel<4;++channel)
        ab[(poly*PrimeCount+channel*ModCount+pi)*n+output_i]=shoup(mixed[channel],twists[pi*n+i],p);
}
__global__ void prepare_rns(const uint32_t* input,uint32_t* ab,const SmallMod* mods,
                             const Twiddle* twists,const Twiddle* powers,const Roots* roots,unsigned n,unsigned logn,
                             unsigned begin=0,unsigned count=0,bool natural=false) {
    unsigned i=begin+blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(count ? begin+count : n)) return;
    unsigned pi=blockIdx.y%ModCount,poly=blockIdx.y/ModCount;
    prepare_coefficient(input+poly*n*AbiWords+i,n,ab,twists,powers,roots,n,logn,i,pi,poly,mods[pi].p,natural);
}
// Four warps share one AoS tile, each warp evaluating a different prime.
__global__ void prepare_grouped(const uint32_t* input,uint32_t* ab,const SmallMod* mods,
                             const Twiddle* twists,const Twiddle* powers,const Roots* roots,unsigned n,unsigned logn,
                             unsigned begin=0,unsigned count=0,bool natural=false,bool packed=false) {
    __shared__ uint32_t tile[32*29];
    unsigned t=threadIdx.x,lane=t&31,warp=t>>5;
    unsigned pi=(blockIdx.y%(ModCount/4))*4+warp,poly=blockIdx.y/(ModCount/4);
    unsigned base=begin+blockIdx.x*32,end=count ? begin+count : n;
    input+=poly*(packed ? count : n)*AbiWords;
    #pragma unroll
    for(unsigned index=t;index<32*AbiWords;index+=128) {
        unsigned row=index/AbiWords,word=index%AbiWords;
        if(base+row<end) tile[row*29+word]=input[(base+row-(packed ? begin : 0))*AbiWords+word];
    }
    __syncthreads();
    unsigned i=base+lane;
    if(i>=end) return;
    prepare_coefficient(tile+lane*29,1,ab,twists,powers,roots,n,logn,i,pi,poly,mods[pi].p,natural);
}
// The chunk converter writes natural order while H2D is still active.
// Permute only after all chunks are ready, using coalesced loads and stores.
__global__ void reverse_rns(const uint32_t* source,uint32_t* destination,unsigned n) {
    __shared__ uint32_t tile[32*33];
    unsigned x=threadIdx.x&31,y0=threadIdx.x>>5;
    source+=blockIdx.y*n;destination+=blockIdx.y*n;
    #pragma unroll
    for(unsigned y=y0;y<32;y+=8) tile[y*33+x]=source[(y<<10)+(blockIdx.x<<5)+x];
    __syncthreads();
    unsigned reverse_x=__brev(x)>>27,reverse_middle=__brev(blockIdx.x)>>27;
    #pragma unroll
    for(unsigned y=y0;y<32;y+=8)
        destination[(y<<10)+(reverse_middle<<5)+x]=tile[reverse_x*33+(__brev(y)>>27)];
}
// For N=2^15, split the index into three groups of five bits. Fix the
// middle group and transpose the outer groups through padded shared memory.
// Both the coefficient reads and the bit-reversed residue writes now use
// consecutive 32-word warp transactions.

// Four warps reuse 32 complete coefficients. Padding to 29 words avoids
// shared-memory bank conflicts when a warp reads the same limb of 32 inputs.

__device__ inline unsigned small_index(unsigned x) {
#if FHERMA_SHARED_SWIZZLE && FHERMA_RNS_RADIX8
    unsigned high=(x>>5)&7;
    return x^high^((high&6)<<2);
#elif FHERMA_SHARED_SWIZZLE && FHERMA_RNS_RADIX4
    return x^(((x>>5)&3)*5)^((x>>2)&16);
#else
    return x;
#endif
}
template<bool Product> __device__ inline uint32_t small_value(
    const uint32_t* values,const uint32_t* paired,unsigned local,unsigned begin,unsigned channel,
    unsigned n,unsigned logn,const SmallMod& modulus) {
    if constexpr(!Product) return values[local];
    else {
        unsigned frequency=__brev(begin+local)>>(32-logn);
        unsigned index=FHERMA_DIF_FORWARD && n==32768 ? begin+local :
            (FHERMA_RNS_TAIL && n==32768 ? (frequency%Tile)*TailRows+frequency/Tile : frequency);
        return multiply_mod(paired[channel*n+index],paired[(PrimeCount+channel)*n+index],modulus);
    }
}
template<bool Product,bool Lazy> __global__ void small_rns(uint32_t* values,const SmallMod* mods,const Twiddle* tables,unsigned n,const uint32_t* paired=nullptr,unsigned logn=0) {
    __shared__ uint32_t tile[Tile];
    unsigned t=threadIdx.x,prime_i=blockIdx.y%ModCount;
    values+=blockIdx.y*n+blockIdx.x*Tile;
    tables+=prime_i*Tile;
    SmallMod modulus=mods[prime_i];uint32_t p=modulus.p;
#if FHERMA_RNS_RADIX8
    #pragma unroll
    for(unsigned k=0;k<8;++k) tile[small_index(t+k*Tile/8)]=small_value<Product>(values,paired,t+k*Tile/8,blockIdx.x*Tile,blockIdx.y,n,logn,modulus);
    __syncthreads();
    for(unsigned half=1;half*8<=Tile;half*=8) {
        unsigned j=t&(half-1),i=8*(t-j)+j;uint32_t x[8];
        #pragma unroll
        for(unsigned k=0;k<8;++k) x[k]=tile[small_index(i+k*half)];
        #pragma unroll
        for(unsigned step=1;step<8;step*=2) {
            #pragma unroll
            for(unsigned lane=0;lane<step;++lane) {
                unsigned exponent=(j+lane*half)*(Tile/(step*half));
                Twiddle w=tables[exponent];
                #pragma unroll
                for(unsigned k=lane;k<8;k+=2*step) {
                    uint32_t u=x[k],v=x[k+step];
                    if(exponent) v=ntt_shoup<Lazy>(v,w,p);
                    x[k]=ntt_add<Lazy>(u,v,p);x[k+step]=ntt_sub<Lazy>(u,v,p);
                }
            }
        }
        if(half*8==Tile) {
            #pragma unroll
            for(unsigned k=0;k<8;++k) values[i+k*half]=x[k];
        } else {
            #pragma unroll
            for(unsigned k=0;k<8;++k) tile[small_index(i+k*half)]=x[k];
            __syncthreads();
        }
    }
    // 1024 needs one more binary stage; 4096 has four complete radix-8 groups.
    if(Tile==1024) {
        #pragma unroll
        for(unsigned k=0;k<4;++k) {
            unsigned j=t+k*Tile/8;
            uint32_t u=tile[small_index(j)],v=ntt_shoup<Lazy>(tile[small_index(j+Tile/2)],tables[2*j],p);
            values[j]=ntt_add<Lazy>(u,v,p);values[j+Tile/2]=ntt_sub<Lazy>(u,v,p);
        }
    }
#elif FHERMA_RNS_RADIX4
    #pragma unroll
    for(unsigned k=0;k<4;++k) tile[small_index(t+k*Tile/4)]=small_value<Product>(values,paired,t+k*Tile/4,blockIdx.x*Tile,blockIdx.y,n,logn,modulus);
    __syncthreads();
    for(unsigned half=1;half<Tile;half*=4) {
        unsigned j=t&(half-1),i=4*(t-j)+j;
        uint32_t x0=tile[small_index(i)],x1=tile[small_index(i+half)],x2=tile[small_index(i+2*half)],x3=tile[small_index(i+3*half)];
        if(j!=0) {
            Twiddle w=tables[j*(Tile/half)];
            x1=ntt_shoup<Lazy>(x1,w,p);x3=ntt_shoup<Lazy>(x3,w,p);
        }
        uint32_t a0=ntt_add<Lazy>(x0,x1,p),a1=ntt_sub<Lazy>(x0,x1,p);
        uint32_t a2=ntt_add<Lazy>(x2,x3,p),a3=ntt_sub<Lazy>(x2,x3,p);
        if(j!=0) a2=ntt_shoup<Lazy>(a2,tables[j*(Tile/(2*half))],p);
        a3=ntt_shoup<Lazy>(a3,tables[(j+half)*(Tile/(2*half))],p);
        if(half==Tile/4) {
            values[i]=ntt_add<Lazy>(a0,a2,p);values[i+2*half]=ntt_sub<Lazy>(a0,a2,p);
            values[i+half]=ntt_add<Lazy>(a1,a3,p);values[i+3*half]=ntt_sub<Lazy>(a1,a3,p);
        } else {
            tile[small_index(i)]=ntt_add<Lazy>(a0,a2,p);tile[small_index(i+2*half)]=ntt_sub<Lazy>(a0,a2,p);
            tile[small_index(i+half)]=ntt_add<Lazy>(a1,a3,p);tile[small_index(i+3*half)]=ntt_sub<Lazy>(a1,a3,p);
            __syncthreads();
        }
    }
#else
    tile[small_index(t)]=small_value<Product>(values,paired,t,blockIdx.x*Tile,blockIdx.y,n,logn,modulus);tile[small_index(t+Tile/2)]=small_value<Product>(values,paired,t+Tile/2,blockIdx.x*Tile,blockIdx.y,n,logn,modulus);
    __syncthreads();
    unsigned stride=Tile;
    for(unsigned half=1;half<Tile;half*=2,stride>>=1) {
        unsigned j=t&(half-1),i=2*(t-j)+j;
        uint32_t u=tile[small_index(i)],v=ntt_shoup<Lazy>(tile[small_index(i+half)],tables[j*stride],p);
        tile[small_index(i)]=ntt_add<Lazy>(u,v,p);tile[small_index(i+half)]=ntt_sub<Lazy>(u,v,p);
        __syncthreads();
    }
    values[t]=tile[small_index(t)];values[t+Tile/2]=tile[small_index(t+Tile/2)];
#endif
}
// DIF forward accepts natural-order input and leaves bit-reversed frequencies.
// The inverse DIT consumes this order directly, including the pointwise product.
__device__ inline unsigned small_dif_index(unsigned x) {
    return x^(((x>>5)&7)<<1)^((x>>5)&1)^((x>>3)&16);
}
template<bool Lazy> __global__ void small_dif_rns(uint32_t* values,const SmallMod* mods,const Twiddle* tables,unsigned n) {
    static_assert(!FHERMA_DIF_FORWARD || (Tile==1024 && FHERMA_RNS_TAIL),"DIF requires 1024-element tiles and the tail path");
    __shared__ uint32_t tile[Tile];
    unsigned t=threadIdx.x,pi=blockIdx.y%ModCount;
    values+=blockIdx.y*n+blockIdx.x*Tile;tables+=pi*Tile;
    uint32_t p=mods[pi].p;
    #pragma unroll
    for(unsigned k=0;k<8;++k) tile[small_dif_index(t+k*Tile/8)]=values[t+k*Tile/8];
    __syncthreads();
    for(unsigned half=Tile/8;half;half/=8) {
        unsigned j=t&(half-1),i=8*(t-j)+j;uint32_t x[8];
        #pragma unroll
        for(unsigned k=0;k<8;++k) x[k]=tile[small_dif_index(i+k*half)];
        #pragma unroll
        for(unsigned step=4;step;step/=2) {
            #pragma unroll
            for(unsigned lane=0;lane<step;++lane) {
                unsigned exponent=(j+lane*half)*(Tile/(step*half));
                Twiddle w=tables[exponent];
                #pragma unroll
                for(unsigned k=lane;k<8;k+=2*step) {
                    uint32_t u=x[k],v=x[k+step];
                    x[k]=ntt_add<Lazy>(u,v,p);x[k+step]=ntt_sub<Lazy>(u,v,p);
                    if(exponent) x[k+step]=ntt_shoup<Lazy>(x[k+step],w,p);
                }
            }
        }
        #pragma unroll
        for(unsigned k=0;k<8;++k) tile[small_dif_index(i+k*half)]=x[k];
        __syncthreads();
    }
    #pragma unroll
    for(unsigned k=0;k<4;++k) {
        unsigned i=2*(t+k*Tile/8);
        uint32_t u=tile[small_dif_index(i)],v=tile[small_dif_index(i+1)];
        values[i]=ntt_add<Lazy>(u,v,p);values[i+1]=ntt_sub<Lazy>(u,v,p);
    }
}
// The five large DIF stages operate on a column of the [32][1024] layout.
// Restore this layout before the contiguous small DIF tiles, using two shared
// transposes and register shuffles between butterflies.
template<bool Lazy> __global__ void tail_dif_rns(const uint32_t* source,uint32_t* destination,const SmallMod* mods,
                             const Twiddle* tables,unsigned n) {
    __shared__ uint32_t tile[TailColumns*(TailRows+1)];
    unsigned t=threadIdx.x,pi=blockIdx.y%ModCount,column_base=blockIdx.x*TailColumns;
    source+=blockIdx.y*n;destination+=blockIdx.y*n;tables+=pi*n;
    #pragma unroll
    for(unsigned index=t;index<256;index+=128) {
        unsigned row=index/TailColumns,column=index%TailColumns;
        tile[column*(TailRows+1)+row]=source[row*Tile+column_base+column];
    }
    __syncthreads();
    unsigned column=column_base+t/(TailRows/2),k=t%(TailRows/2),offset=(t/(TailRows/2))*(TailRows+1);
    uint32_t p=mods[pi].p,u=tile[offset+k],v=tile[offset+k+TailRows/2];
    uint32_t lower=ntt_add<Lazy>(u,v,p),upper=ntt_shoup<Lazy>(ntt_sub<Lazy>(u,v,p),tables[Tile*(TailRows/2-1)+column*(TailRows/2)+k],p);
    #pragma unroll
    for(unsigned half=TailRows/4;half;half/=2) {
        uint32_t peer_lower=__shfl_xor_sync(0xffffffff,lower,half,TailRows/2);
        uint32_t peer_upper=__shfl_xor_sync(0xffffffff,upper,half,TailRows/2);
        u=(k&half) ? peer_upper : lower;v=(k&half) ? upper : peer_lower;
        unsigned j=k&(half-1);
        lower=ntt_add<Lazy>(u,v,p);upper=ntt_shoup<Lazy>(ntt_sub<Lazy>(u,v,p),tables[Tile*(half-1)+column*half+j],p);
    }
    tile[offset+2*k]=lower;tile[offset+2*k+1]=upper;
    __syncthreads();
    #pragma unroll
    for(unsigned index=t;index<256;index+=128) {
        unsigned row=index/TailColumns,column=index%TailColumns;
        destination[row*Tile+column_base+column]=tile[column*(TailRows+1)+row];
    }
}
template<bool Lazy> __global__ void stage_rns(uint32_t* values,const SmallMod* mods,const Twiddle* tables,
                           unsigned n,unsigned half,unsigned stride) {
    unsigned k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=n/2) return;
    unsigned prime_i=blockIdx.y%ModCount,j=k&(half-1),i=2*(k-j)+j;
    uint32_t p=mods[prime_i].p;
    values+=blockIdx.y*n;
    uint32_t u=values[i],v=ntt_shoup<Lazy>(values[i+half],tables[prime_i*n+j*stride],p);
    values[i]=ntt_add<Lazy>(u,v,p);values[i+half]=ntt_sub<Lazy>(u,v,p);
}
// [prime][TailRows][Tile] -> [prime][Tile][TailRows], padded transpose.
__global__ void transpose_rns(const uint32_t* source,uint32_t* destination,unsigned n) {
    __shared__ uint32_t tile[32*33];
    unsigned x=threadIdx.x&31,y=threadIdx.x>>5,column=blockIdx.x*32;
    source+=blockIdx.y*n;destination+=blockIdx.y*n;
    for(unsigned dy=0;dy<32;dy+=8) if(y+dy<TailRows) tile[(y+dy)*33+x]=source[(y+dy)*Tile+column+x];
    __syncthreads();
    for(unsigned dy=0;dy<32;dy+=8) if(x<TailRows) destination[(column+y+dy)*TailRows+x]=tile[x*33+y+dy];
}
template<bool Lazy> __global__ void tail_rns(uint32_t* values,const SmallMod* mods,const Twiddle* tables,unsigned n) {
    __shared__ uint32_t tile[256];
    unsigned t=threadIdx.x,prime_i=blockIdx.y%ModCount;
    unsigned column=blockIdx.x*TailColumns+t/(TailRows/2),k=t%(TailRows/2),offset=(t/(TailRows/2))*TailRows;
    values+=blockIdx.y*n+blockIdx.x*256;tables+=prime_i*n;
    uint32_t p=mods[prime_i].p;
    tile[t]=values[t];tile[t+128]=values[t+128];
    __syncthreads();
    for(unsigned half=1;half<TailRows;half*=2) {
        unsigned j=k&(half-1),i=offset+2*(k-j)+j;
        uint32_t u=tile[i],v=ntt_shoup<Lazy>(tile[i+half],tables[Tile*(half-1)+column*half+j],p);
        tile[i]=ntt_add<Lazy>(u,v,p);tile[i+half]=ntt_sub<Lazy>(u,v,p);
        __syncthreads();
    }
    values[t]=tile[t];values[t+128]=tile[t+128];
}
// Transpose adjacent columns while executing the remaining tail stages.
template<bool Lazy> __global__ void fused_tail_rns(const uint32_t* source,uint32_t* destination,const SmallMod* mods,
                               const Twiddle* tables,unsigned n) {
    __shared__ uint32_t tile[TailColumns*(TailRows+1)];
    unsigned t=threadIdx.x,prime_i=blockIdx.y%ModCount,column_base=blockIdx.x*TailColumns;
    source+=blockIdx.y*n;destination+=blockIdx.y*n+blockIdx.x*256;tables+=prime_i*n;
    #pragma unroll
    for(unsigned index=t;index<256;index+=128) {
        unsigned row=index/TailColumns,column=index%TailColumns;
        tile[column*(TailRows+1)+row]=source[row*Tile+column_base+column];
    }
    __syncthreads();
    unsigned column=column_base+t/(TailRows/2),k=t%(TailRows/2),offset=(t/(TailRows/2))*(TailRows+1);
    uint32_t p=mods[prime_i].p;
#if FHERMA_SHUFFLE_TAIL
    uint32_t u=tile[offset+2*k];
    uint32_t v=ntt_shoup<Lazy>(tile[offset+2*k+1],tables[column],p);
    uint32_t lower=ntt_add<Lazy>(u,v,p),upper=ntt_sub<Lazy>(u,v,p);
    #pragma unroll
    for(unsigned half=2;half<TailRows;half*=2) {
        // Redistribute the preceding butterfly pairs entirely in registers.
        uint32_t peer_lower=__shfl_xor_sync(0xffffffff,lower,half/2,TailRows/2);
        uint32_t peer_upper=__shfl_xor_sync(0xffffffff,upper,half/2,TailRows/2);
        u=(k&(half/2)) ? peer_upper : lower;
        v=(k&(half/2)) ? upper : peer_lower;
        unsigned j=k&(half-1);
        v=ntt_shoup<Lazy>(v,tables[Tile*(half-1)+column*half+j],p);
        lower=ntt_add<Lazy>(u,v,p);upper=ntt_sub<Lazy>(u,v,p);
    }
    unsigned output=(t/(TailRows/2))*TailRows+k;
    destination[output]=lower;destination[output+TailRows/2]=upper;
#else
    for(unsigned half=1;half<TailRows;half*=2) {
        unsigned j=k&(half-1),i=offset+2*(k-j)+j;
        uint32_t u=tile[i],v=ntt_shoup<Lazy>(tile[i+half],tables[Tile*(half-1)+column*half+j],p);
        tile[i]=ntt_add<Lazy>(u,v,p);tile[i+half]=ntt_sub<Lazy>(u,v,p);
        __syncthreads();
    }
    destination[2*t]=tile[offset+2*k];destination[2*t+1]=tile[offset+2*k+1];
#endif
}
__device__ unsigned natural_index(unsigned i,unsigned n) {
    return FHERMA_RNS_TAIL && n==32768 ? ((i&(TailRows-1))*Tile+i/TailRows) : i;
}
__global__ void product_rns(const uint32_t* ab,uint32_t* c,const SmallMod* mods,unsigned n,unsigned logn) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    unsigned prime_i=blockIdx.y;
    uint32_t a=ab[prime_i*n+i],b=ab[(PrimeCount+prime_i)*n+i];
    unsigned index=FHERMA_DIF_FORWARD && n==32768 ? i : (__brev(natural_index(i,n))>>(32-logn));
    c[prime_i*n+index]=multiply_mod(a,b,mods[prime_i%ModCount]);
}
// Extract a word of the exact 512-bit magnitude shifted by 217*j.
__device__ uint32_t shifted_word(const Wide& value,unsigned word,unsigned shift) {
    unsigned base=shift/32,bits=shift%32;
    if(word<base) return 0;
    unsigned index=word-base;
    uint32_t result=index<WideWords ? uint32_t(value[index])<<bits : 0;
    if(bits && index>0 && index-1<WideWords) result|=uint32_t(value[index-1])>>(32-bits);
    return result;
}
template<unsigned Component> __device__ Big fold_component(const Wide& magnitude,const Big& q) {
    constexpr unsigned shift=PartBits*Component;Big low(uint32_t(0)),high(uint32_t(0));
    #pragma unroll
    for(unsigned word=0;word<AbiWords;++word) {
        low[word]=shifted_word(magnitude,word,shift);
        high[word]=(shifted_word(magnitude,word+27,shift)>>4)|(shifted_word(magnitude,word+28,shift)<<28);
    }
    low[27]=uint32_t(low[27])&15u;
    Big folded=low+high.mul_scalar(uint32_t(0)-uint32_t(q[0]));
    if(folded>=q) folded=folded-q;
    return folded;
}
template<bool Lazy> __global__ void reconstruct_rns(const uint32_t* residues,uint32_t* output,const SmallMod* mods,
                                const Twiddle* scales,const uint32_t* bases,const Roots* roots,
                                const uint32_t* q_words,const uint32_t* product,unsigned n) {
    static_assert(CrtOutputs==32,"one warp per quartic component");
    __shared__ uint32_t partial[128*29];
    unsigned lane=threadIdx.x&31,component=threadIdx.x/32,i=blockIdx.x*32+lane;
    Wide accumulator(uint32_t(0));uint64_t fraction=0;uint32_t alpha=0;
    #pragma unroll 1
    for(unsigned pi=0;i<n && pi<ModCount;++pi) {
        auto modulus=mods[pi];uint32_t p=modulus.p;
        uint32_t a=residues[pi*n+i],b=residues[(ModCount+pi)*n+i];
        uint32_t c=residues[(2*ModCount+pi)*n+i],d=residues[(3*ModCount+pi)*n+i],mixed;
        if constexpr(Lazy) {
            if(a>=p) a-=p;if(b>=p) b-=p;if(c>=p) c-=p;if(d>=p) d-=p;
        }
        if(component==0 || component==2) {
            uint32_t u=add_mod(a,c,p),v=add_mod(b,d,p);
            mixed=component==0 ? add_mod(u,v,p) : sub_mod(u,v,p);
        } else {
            uint32_t u=sub_mod(a,c,p),v=shoup(sub_mod(b,d,p),roots[pi].i,p);
            mixed=component==1 ? sub_mod(u,v,p) : add_mod(u,v,p);
        }
        #if FHERMA_QUARTIC_COMPACT_SCALE
        uint32_t normalized=shoup(mixed,roots[pi].inverse_powers[component],p);
        uint32_t t=shoup(normalized,scales[pi*n+i],p);
#else
        uint32_t t=shoup(mixed,scales[(component*ModCount+pi)*n+i],p);
#endif
        uint64_t term=uint64_t(t)*modulus.reciprocal,next=fraction+term;
        alpha+=next<fraction;fraction=next;
        accumulator=accumulator+Wide(bases,pi).mul_scalar(t);
    }
    uint32_t nearest=alpha+uint32_t(fraction>>63);
    Wide correction=Wide(product,0).mul_scalar(nearest);
    bool negative=!(accumulator>=correction);
    Wide magnitude=negative ? correction-accumulator : accumulator-correction;
    const Big q(q_words,0);Big answer(uint32_t(0));
    switch(component) {
        case 0:answer=fold_component<0>(magnitude,q);break;
        case 1:answer=fold_component<1>(magnitude,q);break;
        case 2:answer=fold_component<2>(magnitude,q);break;
        default:answer=fold_component<3>(magnitude,q);break;
    }
    if(negative) answer=Big(uint32_t(0)).sub_mod(answer,q);
    #pragma unroll
    for(unsigned word=0;word<AbiWords;++word) partial[threadIdx.x*29+word]=answer[word];
    __syncthreads();
    if(component==0) {
        answer=Big(uint32_t(0));
        #pragma unroll
        for(unsigned part=0;part<4;++part) {
            Big contribution(uint32_t(0));
            #pragma unroll
            for(unsigned word=0;word<AbiWords;++word) contribution[word]=partial[(part*32+lane)*29+word];
            answer=answer.add_mod(contribution,q);
        }
        #pragma unroll
        for(unsigned word=0;word<AbiWords;++word) partial[lane*29+word]=answer[word];
    }
    __syncthreads();
    for(unsigned word=threadIdx.x;word<32*AbiWords;word+=128) {
        unsigned coefficient=word/AbiWords,limb=word%AbiWords,index=blockIdx.x*32+coefficient;
        if(index<n) output[natural_index(index,n)*AbiWords+limb]=partial[coefficient*29+limb];
    }
}
void check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
struct State {
    HostCopyPool copy;
    // Only zero-initialized storage is retained. Every run allocates a fresh
    // replacement inside its timed call; returned result ownership is unique.
    std::vector<uint32_t> spare_output;
    unsigned n=0,logn=0;
    bool overlap_input=false,lazy_ntt=false;
    uint32_t *input=nullptr,*input_soa=nullptr,*ab=nullptr,*c=nullptr,*bases=nullptr,*scratch=nullptr;
    uint32_t *q=nullptr,*product=nullptr,*host_input=nullptr,*host_output=nullptr;
    SmallMod* mods=nullptr; Roots* roots=nullptr;
    Twiddle *forward=nullptr,*inverse=nullptr,*scale=nullptr,*small_forward=nullptr,*small_inverse=nullptr,*tail_forward=nullptr,*tail_inverse=nullptr,*input_powers=nullptr;
#if FHERMA_GRAPH
    cudaStream_t stream=nullptr;cudaGraphExec_t graph=nullptr;
    cudaStream_t transfer_stream=nullptr;
    cudaEvent_t input_ready[FHERMA_PIPELINE_INPUT]{};
    cudaGraphExec_t prepare_graph[FHERMA_PIPELINE_INPUT]{};
#if FHERMA_PIPELINE_OUTPUT>1
    cudaEvent_t output_ready[FHERMA_PIPELINE_OUTPUT]{};
#endif
#endif
#if FHERMA_PROFILE
    cudaEvent_t events[8]{};
#endif
    ~State() {
#if FHERMA_GRAPH
        if(graph) cudaGraphExecDestroy(graph);
        for(auto executable:prepare_graph) if(executable) cudaGraphExecDestroy(executable);
        if(stream) cudaStreamDestroy(stream);
        if(transfer_stream) cudaStreamDestroy(transfer_stream);
        for(auto event:input_ready) if(event) cudaEventDestroy(event);
#if FHERMA_PIPELINE_OUTPUT>1
        for(auto event:output_ready) if(event) cudaEventDestroy(event);
#endif
#endif
#if FHERMA_PROFILE
        for(auto event:events) if(event) cudaEventDestroy(event);
#endif
        cudaFree(input);cudaFree(input_soa);cudaFree(ab);cudaFree(c);cudaFree(bases);cudaFree(scratch);
        cudaFree(q);cudaFree(product);cudaFree(mods);cudaFree(forward);cudaFree(inverse);
        cudaFree(roots);cudaFree(scale);cudaFree(small_forward);cudaFree(small_inverse);cudaFree(tail_forward);cudaFree(tail_inverse);cudaFree(input_powers);
        cudaFreeHost(host_input);cudaFreeHost(host_output);
    }
};
void mark(State& s,unsigned i,cudaStream_t stream=nullptr) {
#if FHERMA_PROFILE
#if FHERMA_GRAPH
    // Ordinary captured events represent dependencies without a timestamp.
    // Explicit external nodes execute the event record on every replay.
    check(cudaEventRecordWithFlags(s.events[i],stream,cudaEventRecordExternal),"RNS graph profile mark");
#else
    check(cudaEventRecord(s.events[i],stream),"RNS profile mark");
#endif
#else
    (void)s;(void)i;(void)stream;
#endif
}
template<class T> void upload(T*& destination,const std::vector<T>& source) {
    if(source.empty()) return;
    check(cudaMalloc(&destination,source.size()*sizeof(T)),"allocate parameter table");
    check(cudaMemcpy(destination,source.data(),source.size()*sizeof(T),cudaMemcpyHostToDevice),"upload parameter table");
}
void launch_input_chunk(State& s,unsigned begin,unsigned count,cudaStream_t stream=nullptr) {
    dim3 abi_tiles((count+31)/32,2),residues((count+127)/128,2*ModCount);
    bool paired=FHERMA_MAPPED_INPUT || FHERMA_PAIRED_INPUT;
    const uint32_t* source=FHERMA_MAPPED_INPUT ? s.host_input : s.input;
    if(paired) source+=size_t(2)*begin*AbiWords;
    if(FHERMA_RNS_GROUPED) {
        dim3 groups((count+31)/32,2*(ModCount/4));
        prepare_grouped<<<groups,128,0,stream>>>(source,s.ab,s.mods,s.forward,s.input_powers,s.roots,s.n,s.logn,begin,count,true,paired);
    } else {
    transpose_inputs<<<abi_tiles,256,0,stream>>>(source,s.input_soa,s.n,begin,count,paired);
    prepare_rns<<<residues,128,0,stream>>>(s.input_soa,s.ab,s.mods,s.forward,s.input_powers,s.roots,s.n,s.logn,begin,count,true);
    }
    check(cudaGetLastError(),"RNS input chunk kernels");
}
template<bool Lazy> void launch_rns_impl(State& s,cudaStream_t stream=nullptr) {
    dim3 full((s.n+127)/128,2*PrimeCount),half((s.n/2+127)/128,2*PrimeCount);
    bool dif_forward=FHERMA_DIF_FORWARD && s.n==32768;
    uint32_t* prepared=s.ab;
    if(s.overlap_input) {
      if(!dif_forward) {
        dim3 permutation(32,2*PrimeCount);
        reverse_rns<<<permutation,256,0,stream>>>(s.ab,s.scratch,s.n);
        prepared=s.scratch;
      }
    } else {
        dim3 abi_tiles((s.n+31)/32,2),residues((s.n+127)/128,2*ModCount);
        if(FHERMA_RNS_GROUPED) {
            dim3 groups((s.n+31)/32,2*(ModCount/4));
            prepare_grouped<<<groups,128,0,stream>>>(s.input,s.ab,s.mods,s.forward,s.input_powers,s.roots,s.n,s.logn,0,0,dif_forward);
        } else {
        transpose_inputs<<<abi_tiles,256,0,stream>>>(s.input,s.input_soa,s.n);
        prepare_rns<<<residues,128,0,stream>>>(s.input_soa,s.ab,s.mods,s.forward,s.input_powers,s.roots,s.n,s.logn,0,0,dif_forward);
        }
    }
    mark(s,2,stream);
    unsigned first=1;
    uint32_t* forward_values=prepared;
    if(dif_forward) {
        dim3 tails(128,2*PrimeCount),tiles(s.n/Tile,2*PrimeCount);
        tail_dif_rns<Lazy><<<tails,128,0,stream>>>(prepared,s.scratch,s.mods,s.tail_forward,s.n);
        small_dif_rns<Lazy><<<tiles,Tile/8,0,stream>>>(s.scratch,s.mods,s.small_forward,s.n);
        forward_values=s.scratch;
        first=Tile;
    } else {
    if(s.n>=Tile) {
        dim3 tiles(s.n/Tile,2*PrimeCount);
        small_rns<false,Lazy><<<tiles,Tile/(FHERMA_RNS_RADIX8?8:(FHERMA_RNS_RADIX4?4:2)),0,stream>>>(prepared,s.mods,s.small_forward,s.n);
        first=Tile;
    }
    if(FHERMA_RNS_TAIL && s.n==32768) {
        dim3 transposes(Tile/32,2*PrimeCount),tails(128,2*PrimeCount);
        uint32_t* transposed=s.overlap_input ? s.ab : s.scratch;
        if(FHERMA_RNS_FUSED_TRANSPOSE) {
            fused_tail_rns<Lazy><<<tails,128,0,stream>>>(prepared,transposed,s.mods,s.tail_forward,s.n);
        } else {
            transpose_rns<<<transposes,256,0,stream>>>(prepared,transposed,s.n);
            tail_rns<Lazy><<<tails,128,0,stream>>>(transposed,s.mods,s.tail_forward,s.n);
        }
        forward_values=transposed;
    } else for(unsigned h=first;h<s.n;h*=2)
        stage_rns<Lazy><<<half,128,0,stream>>>(s.ab,s.mods,s.forward,s.n,h,s.n/h);
    }
    mark(s,3,stream);
    dim3 inverse_full(full.x,PrimeCount),inverse_half(half.x,PrimeCount);
    bool fused_product=FHERMA_FUSED_PRODUCT && s.n>=Tile;
    if(!fused_product) product_rns<<<inverse_full,128,0,stream>>>(forward_values,s.c,s.mods,s.n,s.logn);
    mark(s,4,stream);
    if(s.n>=Tile) {
        dim3 tiles(s.n/Tile,PrimeCount);
        if(fused_product)
            small_rns<true,Lazy><<<tiles,Tile/(FHERMA_RNS_RADIX8?8:(FHERMA_RNS_RADIX4?4:2)),0,stream>>>(s.c,s.mods,s.small_inverse,s.n,forward_values,s.logn);
        else
            small_rns<false,Lazy><<<tiles,Tile/(FHERMA_RNS_RADIX8?8:(FHERMA_RNS_RADIX4?4:2)),0,stream>>>(s.c,s.mods,s.small_inverse,s.n);
    }
    uint32_t* inverse_values=s.c;
    if(FHERMA_RNS_TAIL && s.n==32768) {
        dim3 transposes(Tile/32,PrimeCount),tails(128,PrimeCount);
        if(FHERMA_RNS_FUSED_TRANSPOSE) {
            fused_tail_rns<Lazy><<<tails,128,0,stream>>>(s.c,s.ab,s.mods,s.tail_inverse,s.n);
        } else {
            transpose_rns<<<transposes,256,0,stream>>>(s.c,s.ab,s.n);
            tail_rns<Lazy><<<tails,128,0,stream>>>(s.ab,s.mods,s.tail_inverse,s.n);
        }
        inverse_values=s.ab;
    } else for(unsigned h=first;h<s.n;h*=2)
        stage_rns<Lazy><<<inverse_half,128,0,stream>>>(s.c,s.mods,s.inverse,s.n,h,s.n/h);
    mark(s,5,stream);
    unsigned crt_blocks=(s.n+CrtOutputs-1)/CrtOutputs;
    reconstruct_rns<Lazy><<<crt_blocks,128,0,stream>>>(inverse_values,FHERMA_MAPPED_OUTPUT ? s.host_output : s.input,s.mods,s.scale,s.bases,s.roots,
                                            s.q,s.product,s.n);
    mark(s,6,stream);
    check(cudaGetLastError(),"RNS kernels");
}
void launch_rns(State& s,cudaStream_t stream=nullptr) {
    if(s.lazy_ntt) launch_rns_impl<true>(s,stream);
    else launch_rns_impl<false>(s,stream);
}
} // namespace

void* fherma_init(const fherma::Point& p) {
    if(p.N<2 || (p.N&(p.N-1)) || p.N>32768 || p.W!=868 || p.L!=quartic::AbiWords || p.q.data.size()!=quartic::AbiWords)
        throw std::runtime_error("RNS coverage: power-of-two 2<=N<=32768, W=868, L=28");
    uint32_t delta=uint32_t(0)-p.q.data[0];
    bool special=delta>0 && delta<0x10000000u && p.q.data[27]==15u;
    for(unsigned k=1;k<27;++k) special=special && p.q.data[k]==0xffffffffu;
    if(!special) throw std::runtime_error("RNS coverage: q=2^868-c, c<2^28");
    static_assert(!FHERMA_PROFILE || (FHERMA_PIPELINE_INPUT<=1 && FHERMA_PIPELINE_OUTPUT<=1),
                  "profile arithmetic separately from overlapped host transfers");
    static_assert(FHERMA_PIPELINE_OUTPUT<=32,"output segments fit the smallest point");
    static_assert(!FHERMA_MAPPED_OUTPUT || FHERMA_PIPELINE_OUTPUT==1,"mapped output waits for the complete CRT kernel");
    static_assert(!(FHERMA_OUTPUT_APPEND && FHERMA_OUTPUT_SPARE),"choose one output construction experiment");
    pin_near_gpu();
#if FHERMA_HOST_PROFILE && defined(__linux__)
    std::string thp_policy;
    std::getline(std::ifstream("/sys/kernel/mm/transparent_hugepage/enabled"),thp_policy);
    std::fprintf(stderr,"OUTPUT_MEMORY huge_hint=%d thp=%s\n",FHERMA_HUGE_OUTPUT,thp_policy.c_str());
#endif
#ifdef __CUDACC__
    int device=0,pageable=0,host_tables=0,concurrent=0,direct=0;
    check(cudaGetDevice(&device),"query active GPU");
    check(cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,device),"query pageable access");
    check(cudaDeviceGetAttribute(&host_tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,device),"query host page tables");
    check(cudaDeviceGetAttribute(&concurrent,cudaDevAttrConcurrentManagedAccess,device),"query concurrent managed access");
    check(cudaDeviceGetAttribute(&direct,cudaDevAttrDirectManagedMemAccessFromHost,device),"query direct host access");
    std::fprintf(stderr,"MEMORY_CAPS pageable=%d host_tables=%d concurrent=%d direct=%d\n",pageable,host_tables,concurrent,direct);
    if(FHERMA_MAPPED_INPUT || FHERMA_MAPPED_OUTPUT) {
        int unified=0;
        check(cudaDeviceGetAttribute(&unified,cudaDevAttrUnifiedAddressing,device),"query UVA for pinned input");
        if(!unified) throw std::runtime_error("mapped pinned buffers require unified addressing");
    }
#endif
    auto s=std::make_unique<State>();s->n=p.N;s->logn=__builtin_ctz(p.N);
    s->overlap_input=FHERMA_OVERLAP_PREPARE && FHERMA_PIPELINE_INPUT>1 && FHERMA_RNS_TAIL && p.N==32768;
    static_assert(!FHERMA_OVERLAP_PREPARE || !FHERMA_PROFILE,"chunk prepare cannot use diagnostic GPU events");
    static_assert(FHERMA_PIPELINE_INPUT>0 && (32768%FHERMA_PIPELINE_INPUT)==0,"input chunks must contain whole coefficients");
    auto constants=quartic::setup(p.N,delta);
    s->lazy_ntt=FHERMA_LAZY_NTT && constants.mods[0].p<(uint32_t(1)<<30);
    upload(s->input_powers,constants.input_powers);upload(s->roots,constants.roots);
    upload(s->mods,constants.mods);upload(s->bases,constants.bases);
    upload(s->product,constants.product);upload(s->q,p.q.data);
    constexpr unsigned ScaleCount=FHERMA_QUARTIC_COMPACT_SCALE ? quartic::ModCount : quartic::PrimeCount;
    constants.scale.resize(size_t(ScaleCount)*p.N);
    if(FHERMA_RNS_TAIL && p.N==32768) {
        auto original=constants.scale;
        for(unsigned pi=0;pi<ScaleCount;++pi)
            for(unsigned i=0;i<p.N;++i)
                constants.scale[pi*p.N+i]=original[pi*p.N+(i&(TailRows-1))*Tile+i/TailRows];
    }
    upload(s->forward,constants.forward);upload(s->inverse,constants.inverse);upload(s->scale,constants.scale);
    upload(s->small_forward,constants.small_forward);upload(s->small_inverse,constants.small_inverse);
    if(FHERMA_RNS_TAIL && p.N==32768) {
        std::vector<quartic::Twiddle> forward_tail(size_t(quartic::ModCount)*p.N),inverse_tail(forward_tail.size());
        for(unsigned pi=0;pi<quartic::ModCount;++pi)
            for(unsigned half=1;half<TailRows;half*=2)
                for(unsigned col=0;col<Tile;++col)
                    for(unsigned j=0;j<half;++j) {
                        unsigned index=pi*p.N+Tile*(half-1)+col*half+j;
                        unsigned source=pi*p.N+(col+Tile*j)*(TailRows/half);
                        forward_tail[index]=constants.forward[source];inverse_tail[index]=constants.inverse[source];
                    }
        upload(s->tail_forward,forward_tail);upload(s->tail_inverse,inverse_tail);
        check(cudaMalloc(&s->scratch,size_t(2)*quartic::PrimeCount*p.N*4),"RNS transpose scratch");
    }
    size_t bytes=size_t(p.N)*quartic::AbiWords*4;
    check(cudaMalloc(&s->input,2*bytes),"allocate ABI buffers");
    if(!FHERMA_RNS_GROUPED) check(cudaMalloc(&s->input_soa,2*bytes),"allocate limb-plane inputs");
    check(cudaMalloc(&s->ab,size_t(2)*quartic::PrimeCount*p.N*4),"allocate RNS operands");
    check(cudaMalloc(&s->c,size_t(quartic::PrimeCount)*p.N*4),"allocate RNS inverse");
#if defined(__CUDACC__) && FHERMA_INPUT_WC
    check(cudaHostAlloc(reinterpret_cast<void**>(&s->host_input),2*bytes,cudaHostAllocWriteCombined),"write-combined RNS inputs");
#else
    check(cudaMallocHost(reinterpret_cast<void**>(&s->host_input),2*bytes),"pinned RNS inputs");
#endif
    check(cudaMallocHost(reinterpret_cast<void**>(&s->host_output),bytes),"pinned RNS output");
    std::memset(s->host_input,0,2*bytes);std::memset(s->host_output,0,bytes);
#if FHERMA_PROFILE
    for(auto& event:s->events) check(cudaEventCreate(&event),"RNS profile event");
#endif
#if FHERMA_GRAPH
    if(s->overlap_input) {
        check(cudaStreamCreateWithFlags(&s->transfer_stream,cudaStreamNonBlocking),"RNS input transfer stream");
        for(auto& event:s->input_ready) check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"RNS input segment event");
    }
#if FHERMA_PIPELINE_OUTPUT>1
    for(auto& event:s->output_ready) check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"RNS output segment event");
#endif
    check(cudaStreamCreateWithFlags(&s->stream,cudaStreamNonBlocking),"RNS stream");
    if(s->overlap_input) {
        for(unsigned part=0;part<FHERMA_PIPELINE_INPUT;++part) {
            check(cudaStreamBeginCapture(s->stream,cudaStreamCaptureModeGlobal),"capture RNS input chunk");
            launch_input_chunk(*s,s->n*part/FHERMA_PIPELINE_INPUT,s->n/FHERMA_PIPELINE_INPUT,s->stream);
            cudaGraph_t definition=nullptr;
            check(cudaStreamEndCapture(s->stream,&definition),"finish RNS input chunk capture");
            auto status=cudaGraphInstantiateWithFlags(&s->prepare_graph[part],definition,0);
            cudaGraphDestroy(definition);check(status,"instantiate RNS input chunk");
            check(cudaGraphUpload(s->prepare_graph[part],s->stream),"upload RNS input chunk");
        }
        check(cudaStreamSynchronize(s->stream),"RNS input graphs ready");
    }
    check(cudaStreamBeginCapture(s->stream,cudaStreamCaptureModeGlobal),"capture RNS graph");
    mark(*s,0,s->stream);
#if FHERMA_PIPELINE_INPUT<=1
    check(cudaMemcpyAsync(s->input,s->host_input,2*bytes,cudaMemcpyHostToDevice,s->stream),"capture RNS H2D");
#endif
    mark(*s,1,s->stream);
    launch_rns(*s,s->stream);
#if FHERMA_PIPELINE_OUTPUT<=1 && !FHERMA_MAPPED_OUTPUT
    check(cudaMemcpyAsync(s->host_output,s->input,bytes,cudaMemcpyDeviceToHost,s->stream),"capture RNS D2H");
#endif
    mark(*s,7,s->stream);
    cudaGraph_t definition=nullptr;
    check(cudaStreamEndCapture(s->stream,&definition),"finish RNS capture");
    auto status=cudaGraphInstantiateWithFlags(&s->graph,definition,0);
    cudaGraphDestroy(definition);check(status,"instantiate RNS graph");
    check(cudaGraphUpload(s->graph,s->stream),"upload RNS graph");
    check(cudaStreamSynchronize(s->stream),"RNS graph ready");
#endif
    return s.release();
}
fherma::Outputs fherma_run(void* opaque,const fherma::Inputs& input) {
    auto& s=*static_cast<State*>(opaque);
    size_t words=size_t(s.n)*quartic::AbiWords,bytes=words*4;
    if(input.a.data.size()!=words || input.b.data.size()!=words) throw std::runtime_error("RNS input size");
    try {
#if FHERMA_PROFILE || FHERMA_HOST_PROFILE
    auto pack_start=std::chrono::steady_clock::now();
#endif
#if FHERMA_GRAPH && FHERMA_PIPELINE_INPUT>1
    unsigned input_part=0;
    copy_input_pipeline<FHERMA_PIPELINE_INPUT>(s.copy,s.host_input,input.a.data.data(),input.b.data.data(),words,
        [&](size_t begin,const uint32_t* a,const uint32_t* b,size_t count) {
            if(FHERMA_MAPPED_INPUT && s.overlap_input) {
                check(cudaGraphLaunch(s.prepare_graph[input_part],s.stream),"execute mapped input chunk");
                ++input_part;
                return;
            }
            auto transfer=s.overlap_input ? s.transfer_stream : s.stream;
            if(FHERMA_PAIRED_INPUT && s.overlap_input) {
                check(cudaMemcpyAsync(s.input+2*begin,a,2*count*4,cudaMemcpyHostToDevice,transfer),"RNS paired pipeline H2D");
            } else {
                check(cudaMemcpyAsync(s.input+begin,a,count*4,cudaMemcpyHostToDevice,transfer),"RNS pipeline A H2D");
                check(cudaMemcpyAsync(s.input+words+begin,b,count*4,cudaMemcpyHostToDevice,transfer),"RNS pipeline B H2D");
            }
            if(s.overlap_input) {
                check(cudaEventRecord(s.input_ready[input_part],transfer),"RNS input segment uploaded");
                check(cudaStreamWaitEvent(s.stream,s.input_ready[input_part],0),"RNS prepare waits for input segment");
                check(cudaGraphLaunch(s.prepare_graph[input_part],s.stream),"execute RNS input chunk");
            }
            ++input_part;
        });
#else
    if((FHERMA_MAPPED_INPUT || FHERMA_PAIRED_INPUT) && s.overlap_input)
        copy_input_pipeline<FHERMA_PIPELINE_INPUT>(s.copy,s.host_input,input.a.data.data(),input.b.data.data(),words,
            [](size_t,const uint32_t*,const uint32_t*,size_t) {});
    else s.copy.inputs(s.host_input,input.a.data.data(),input.b.data.data(),bytes);
#endif
#if FHERMA_PROFILE || FHERMA_HOST_PROFILE
    auto pack_end=std::chrono::steady_clock::now();
#endif
    fherma::Outputs output;output.c.shape={s.n,quartic::AbiWords};
    if(FHERMA_OUTPUT_SPARE) output.c.data.swap(s.spare_output);
#if FHERMA_GRAPH
        check(cudaGraphLaunch(s.graph,s.stream),"execute RNS graph");
#if FHERMA_PIPELINE_OUTPUT>1
        for(unsigned part=0;part<FHERMA_PIPELINE_OUTPUT;++part) {
            size_t begin=words*part/FHERMA_PIPELINE_OUTPUT,end=words*(part+1)/FHERMA_PIPELINE_OUTPUT;
            check(cudaMemcpyAsync(s.host_output+begin,s.input+begin,(end-begin)*4,cudaMemcpyDeviceToHost,s.stream),"RNS pipeline D2H");
            check(cudaEventRecord(s.output_ready[part],s.stream),"record RNS output ready");
        }
#endif
#else
        mark(s,0);
        if(!(FHERMA_MAPPED_INPUT && s.overlap_input)) check(cudaMemcpy(s.input,s.host_input,2*bytes,cudaMemcpyHostToDevice),"RNS H2D");
        mark(s,1);
        if(s.overlap_input) for(unsigned part=0;part<FHERMA_PIPELINE_INPUT;++part)
            launch_input_chunk(s,s.n*part/FHERMA_PIPELINE_INPUT,s.n/FHERMA_PIPELINE_INPUT);
        launch_rns(s);
#endif
#if FHERMA_HOST_PROFILE
        auto alloc_start=std::chrono::steady_clock::now();
#endif
        auto& allocation=FHERMA_OUTPUT_SPARE ? s.spare_output : output.c.data;
        allocation.reserve(words);
        if(FHERMA_HUGE_OUTPUT) advise_output_hugepages(allocation.data(),bytes);
#if FHERMA_HOST_PROFILE
        auto reserved_at=std::chrono::steady_clock::now();
#endif
        s.copy.prefault(allocation.data(),bytes);
#if FHERMA_HOST_PROFILE
        auto faulted_at=std::chrono::steady_clock::now();
#endif
        if(!FHERMA_OUTPUT_APPEND) allocation.resize(words);
        // The first call pays for both its own storage and the next empty one.
        if(FHERMA_OUTPUT_SPARE && output.c.data.size()!=words) {
            output.c.data.reserve(words);
            s.copy.prefault(output.c.data.data(),bytes);
            output.c.data.resize(words);
        }
#if FHERMA_HOST_PROFILE
        auto alloc_end=std::chrono::steady_clock::now();
#endif
#if FHERMA_GRAPH && FHERMA_PIPELINE_OUTPUT>1
        check(cudaEventSynchronize(s.output_ready[0]),"first RNS output segment ready");
#elif FHERMA_GRAPH
        check(cudaStreamSynchronize(s.stream),"RNS output ready");
#else
        if(!FHERMA_MAPPED_OUTPUT) check(cudaMemcpy(s.host_output,s.input,bytes,cudaMemcpyDeviceToHost),"RNS D2H");
        mark(s,7);
        check(cudaDeviceSynchronize(),"RNS diagnostic events ready");
#endif
#if FHERMA_PROFILE || FHERMA_HOST_PROFILE
    auto unpack_start=std::chrono::steady_clock::now();
#endif
#if FHERMA_GRAPH && FHERMA_PIPELINE_OUTPUT>1
    for(unsigned part=0;part<FHERMA_PIPELINE_OUTPUT;++part) {
        check(cudaEventSynchronize(s.output_ready[part]),"RNS output segment ready");
        size_t begin=words*part/FHERMA_PIPELINE_OUTPUT,end=words*(part+1)/FHERMA_PIPELINE_OUTPUT;
        if(FHERMA_OUTPUT_APPEND)
            output.c.data.insert(output.c.data.end(),s.host_output+begin,s.host_output+end);
        else s.copy.output(output.c.data.data()+begin,s.host_output+begin,(end-begin)*4);
    }
#else
    if(FHERMA_OUTPUT_APPEND)
        output.c.data.insert(output.c.data.end(),s.host_output,s.host_output+words);
    else s.copy.output(output.c.data.data(),s.host_output,bytes);
#endif
#if FHERMA_PROFILE || FHERMA_HOST_PROFILE
    auto unpack_end=std::chrono::steady_clock::now();
#endif
#if FHERMA_HOST_PROFILE
    auto host_us=[](auto a,auto b) {return std::chrono::duration<double,std::micro>(b-a).count();};
    std::fprintf(stderr,"HOST_STAGED_US pack=%.3f submit=%.3f alloc=%.3f wait=%.3f unpack=%.3f\n",
        host_us(pack_start,pack_end),host_us(pack_end,alloc_start),host_us(alloc_start,alloc_end),
        host_us(alloc_end,unpack_start),host_us(unpack_start,unpack_end));
    std::fprintf(stderr,"HOST_ALLOC_US reserve=%.3f prefault=%.3f zero=%.3f\n",
        host_us(alloc_start,reserved_at),host_us(reserved_at,faulted_at),host_us(faulted_at,alloc_end));
#endif
#if FHERMA_PROFILE
    const char* names[]={"h2d","prepare","forward","product","inverse","finish","d2h"};
    float elapsed[7]{};bool valid=true;
    for(unsigned k=0;k<7;++k) {
        auto status=cudaEventElapsedTime(&elapsed[k],s.events[k],s.events[k+1]);
        if(status!=cudaSuccess) {
            std::fprintf(stderr,"PROFILE_ERROR stage=%s error=%s\n",names[k],cudaGetErrorString(status));
            valid=false;break;
        }
    }
    if(valid) {
        std::fprintf(stderr,"PROFILE_US");
        for(unsigned k=0;k<7;++k) std::fprintf(stderr," %s=%.3f",names[k],1000*elapsed[k]);
    }
    auto us=[](auto a,auto b) {return std::chrono::duration<double,std::micro>(b-a).count();};
    std::fprintf(stderr,"\nHOST_US pack=%.3f unpack=%.3f\n",us(pack_start,pack_end),us(unpack_start,unpack_end));
#endif
    return output;
    } catch(...) {
#if FHERMA_GRAPH
        cudaStreamSynchronize(s.stream);
        if(s.transfer_stream) cudaStreamSynchronize(s.transfer_stream);
#else
        cudaDeviceSynchronize();
#endif
        throw;
    }
}
void fherma_free(void* state) { delete static_cast<State*>(state); }
