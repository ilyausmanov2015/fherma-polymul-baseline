#ifndef FHERMA_RNS_TAIL
#define FHERMA_RNS_TAIL 1
#endif
#ifndef FHERMA_PROFILE
#define FHERMA_PROFILE 1
#endif
#ifndef FHERMA_GRAPH
#define FHERMA_GRAPH 1
#endif
#ifndef FHERMA_COPY_THREADS
#define FHERMA_COPY_THREADS 8
#endif
#ifndef FHERMA_SPIN_COPY
#define FHERMA_SPIN_COPY 1
#endif
#ifndef FHERMA_STREAM_COPY
#define FHERMA_STREAM_COPY 1
#endif
#include "fherma.h"
#include "rns/host_setup.h"
#include <cupqc/bigint.hpp>
#include <cuda_runtime.h>
#include "host_affinity.h"
#include "host_copy_pool.h"
#include <memory>
#include <cstring>
#include <chrono>

namespace {
using namespace rns;
using WideBI=decltype(cupqc::BitWidth<AccumWords*32>()+cupqc::SM<800>()+cupqc::Thread());
using AbiBI=decltype(cupqc::BitWidth<AbiWords*32>()+cupqc::SM<800>()+cupqc::Thread());
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
__global__ void prepare_rns(const uint32_t* input,uint32_t* ab,const SmallMod* mods,
                             const Twiddle* twists,unsigned n,unsigned logn) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    unsigned prime_i=blockIdx.y%PrimeCount,poly=blockIdx.y/PrimeCount;
    SmallMod modulus=mods[prime_i];
    const uint32_t* coefficient=input+(poly*n+i)*AbiWords;
    uint32_t value=0;
    Twiddle base{modulus.base,modulus.base_shoup};
    #pragma unroll
    for(int limb=AbiWords-1;limb>=0;--limb) {
        uint32_t word=coefficient[limb];
        if(word>=modulus.p) word-=modulus.p;
        if(word>=modulus.p) word-=modulus.p;
        value=add_mod(shoup(value,base,modulus.p),word,modulus.p);
    }
    value=shoup(value,twists[prime_i*n+i],modulus.p);
    ab[blockIdx.y*n+(__brev(i)>>(32-logn))]=value;
}
__global__ void small_rns(uint32_t* values,const SmallMod* mods,const Twiddle* tables,unsigned n) {
    __shared__ uint32_t tile[Tile];
    unsigned t=threadIdx.x,prime_i=blockIdx.y%PrimeCount;
    values+=blockIdx.y*n+blockIdx.x*Tile;
    tables+=prime_i*Tile;
    uint32_t p=mods[prime_i].p;
    tile[t]=values[t];tile[t+Tile/2]=values[t+Tile/2];
    __syncthreads();
    unsigned stride=Tile;
    for(unsigned half=1;half<Tile;half*=2,stride>>=1) {
        unsigned j=t&(half-1),i=2*(t-j)+j;
        uint32_t u=tile[i],v=shoup(tile[i+half],tables[j*stride],p);
        tile[i]=add_mod(u,v,p);tile[i+half]=sub_mod(u,v,p);
        __syncthreads();
    }
    values[t]=tile[t];values[t+Tile/2]=tile[t+Tile/2];
}
__global__ void stage_rns(uint32_t* values,const SmallMod* mods,const Twiddle* tables,
                           unsigned n,unsigned half,unsigned stride) {
    unsigned k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=n/2) return;
    unsigned prime_i=blockIdx.y%PrimeCount,j=k&(half-1),i=2*(k-j)+j;
    uint32_t p=mods[prime_i].p;
    values+=blockIdx.y*n;
    uint32_t u=values[i],v=shoup(values[i+half],tables[prime_i*n+j*stride],p);
    values[i]=add_mod(u,v,p);values[i+half]=sub_mod(u,v,p);
}
// [prime][32][1024] -> [prime][1024][32], padded shared transpose.
__global__ void transpose_rns(const uint32_t* source,uint32_t* destination,unsigned n) {
    __shared__ uint32_t tile[32*33];
    unsigned x=threadIdx.x&31,y=threadIdx.x>>5,column=blockIdx.x*32;
    source+=blockIdx.y*n;destination+=blockIdx.y*n;
    for(unsigned dy=0;dy<32;dy+=8) tile[(y+dy)*33+x]=source[(y+dy)*1024+column+x];
    __syncthreads();
    for(unsigned dy=0;dy<32;dy+=8) destination[(column+y+dy)*32+x]=tile[x*33+y+dy];
}
__global__ void tail_rns(uint32_t* values,const SmallMod* mods,const Twiddle* tables,unsigned n) {
    __shared__ uint32_t tile[256];
    unsigned t=threadIdx.x,prime_i=blockIdx.y%PrimeCount;
    unsigned column=blockIdx.x*8+t/16,k=t%16,offset=(t/16)*32;
    values+=blockIdx.y*n+blockIdx.x*256;tables+=prime_i*n;
    uint32_t p=mods[prime_i].p;
    tile[t]=values[t];tile[t+128]=values[t+128];
    __syncthreads();
    for(unsigned half=1;half<32;half*=2) {
        unsigned j=k&(half-1),i=offset+2*(k-j)+j;
        uint32_t u=tile[i],v=shoup(tile[i+half],tables[1024*(half-1)+column*half+j],p);
        tile[i]=add_mod(u,v,p);tile[i+half]=sub_mod(u,v,p);
        __syncthreads();
    }
    values[t]=tile[t];values[t+128]=tile[t+128];
}
__device__ unsigned frequency_index(unsigned i,unsigned n) {
    return FHERMA_RNS_TAIL && n==32768 ? ((i&1023)*32+(i>>10)) : i;
}
__global__ void product_rns(const uint32_t* ab,uint32_t* c,const SmallMod* mods,unsigned n,unsigned logn) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    unsigned prime_i=blockIdx.y;
    unsigned source=frequency_index(i,n);
    uint32_t a=ab[prime_i*n+source],b=ab[(PrimeCount+prime_i)*n+source];
    c[prime_i*n+(__brev(i)>>(32-logn))]=multiply_mod(a,b,mods[prime_i]);
}
// Sum(t_i * (M_i mod q)) is below 57*2^31*q (< 2^905).
// Fold the supported 1024-bit cuPQC accumulator back to the 868-bit ABI.
__device__ Big fold_crt(const Wide& value,const Big& q) {
    uint32_t c=uint32_t(0)-q[0],folded[29];uint64_t carry=0;
    #pragma unroll
    for(unsigned k=0;k<29;++k) {
        uint32_t high=(k+27<AccumWords ? uint32_t(value[k+27])>>4 : 0u)
                     |(k+28<AccumWords ? uint32_t(value[k+28])<<28 : 0u);
        uint32_t low=k<27 ? uint32_t(value[k]) : (k==27 ? (uint32_t(value[k])&15u) : 0u);
        uint64_t v=uint64_t(high)*c+low+carry;
        folded[k]=uint32_t(v);carry=v>>32;
    }
    uint32_t high0=(folded[27]>>4)|(folded[28]<<28),high1=folded[28]>>4;
    folded[27]&=15u;carry=0;
    Big result(uint32_t(0));
    #pragma unroll
    for(unsigned k=0;k<AbiWords;++k) {
        uint64_t addition=k==0 ? uint64_t(high0)*c : (k==1 ? uint64_t(high1)*c : 0);
        uint64_t v=uint64_t(folded[k])+addition+carry;
        result[k]=uint32_t(v);carry=v>>32;
    }
    if(result>=q) result=result-q;
    return result;
}
__global__ void reconstruct_rns(const uint32_t* residues,uint32_t* output,const SmallMod* mods,
                                const Twiddle* scales,const uint32_t* bases,
                                const uint32_t* q_words,const uint32_t* product_mod_q,
                                unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    Wide accumulator(uint32_t(0));
    uint64_t fraction=0;uint32_t alpha=0;
    #pragma unroll 1
    for(unsigned prime_i=0;prime_i<PrimeCount;++prime_i) {
        SmallMod modulus=mods[prime_i];
        // The scale already contains inverse(P/pi mod pi).
        uint32_t t=shoup(residues[prime_i*n+frequency_index(i,n)],scales[prime_i*n+i],modulus.p);
        uint64_t term=uint64_t(t)*modulus.reciprocal,next=fraction+term;
        alpha+=next<fraction;fraction=next;
        accumulator=accumulator+Wide(bases,prime_i).mul_scalar(t);
    }
    // Setup proves |integer convolution| < P/4. The fixed-point error is
    // less than 57*2^31/2^64 < 1/4, so rounding the quotient is exact even
    // for zero, tiny coefficients, and the most negative valid input.
    uint32_t nearest=alpha+uint32_t(fraction>>63);
    const Big q(q_words,0);
    Big answer=fold_crt(accumulator,q);
    Big correction=fold_crt(Wide(product_mod_q,0).mul_scalar(nearest),q);
    answer=answer.sub_mod(correction,q);
    answer.store(output,i);
}
void check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
struct State {
    HostCopyPool copy;
    unsigned n=0,logn=0;
    uint32_t *input=nullptr,*ab=nullptr,*c=nullptr,*bases=nullptr,*scratch=nullptr;
    uint32_t *q=nullptr,*product_mod_q=nullptr,*host_input=nullptr,*host_output=nullptr;
    SmallMod* mods=nullptr;
    Twiddle *forward=nullptr,*inverse=nullptr,*scale=nullptr,*small_forward=nullptr,*small_inverse=nullptr,*tail_forward=nullptr,*tail_inverse=nullptr;
#if FHERMA_GRAPH
    cudaStream_t stream=nullptr;cudaGraphExec_t graph=nullptr;
#endif
#if FHERMA_PROFILE
    cudaEvent_t events[8]{};
#endif
    ~State() {
#if FHERMA_PROFILE
        for(auto event:events) if(event) cudaEventDestroy(event);
#endif
#if FHERMA_GRAPH
        if(graph) cudaGraphExecDestroy(graph);
        if(stream) cudaStreamDestroy(stream);
#endif
        cudaFree(input);cudaFree(ab);cudaFree(c);cudaFree(bases);cudaFree(scratch);
        cudaFree(q);cudaFree(product_mod_q);cudaFree(mods);cudaFree(forward);cudaFree(inverse);
        cudaFree(scale);cudaFree(small_forward);cudaFree(small_inverse);cudaFree(tail_forward);cudaFree(tail_inverse);
        cudaFreeHost(host_input);cudaFreeHost(host_output);
    }
};
void mark(State& s,unsigned i,cudaStream_t stream=nullptr) {
#if FHERMA_PROFILE
    check(cudaEventRecord(s.events[i],stream),"RNS profile mark");
#else
    (void)s;(void)i;(void)stream;
#endif
}
template<class T> void upload(T*& destination,const std::vector<T>& source) {
    if(source.empty()) return;
    check(cudaMalloc(&destination,source.size()*sizeof(T)),"allocate parameter table");
    check(cudaMemcpy(destination,source.data(),source.size()*sizeof(T),cudaMemcpyHostToDevice),"upload parameter table");
}
void launch_rns(State& s,cudaStream_t stream=nullptr) {
    dim3 full((s.n+127)/128,2*PrimeCount),half((s.n/2+127)/128,2*PrimeCount);
    prepare_rns<<<full,128,0,stream>>>(s.input,s.ab,s.mods,s.forward,s.n,s.logn);
    mark(s,2,stream);
    unsigned first=1;
    if(s.n>=Tile) {
        dim3 tiles(s.n/Tile,2*PrimeCount);
        small_rns<<<tiles,Tile/2,0,stream>>>(s.ab,s.mods,s.small_forward,s.n);
        first=Tile;
    }
    uint32_t* forward_values=s.ab;
    if(FHERMA_RNS_TAIL && s.n==32768) {
        dim3 transposes(32,2*PrimeCount),tails(128,2*PrimeCount);
        transpose_rns<<<transposes,256,0,stream>>>(s.ab,s.scratch,s.n);
        tail_rns<<<tails,128,0,stream>>>(s.scratch,s.mods,s.tail_forward,s.n);
        forward_values=s.scratch;
    } else for(unsigned h=first;h<s.n;h*=2)
        stage_rns<<<half,128,0,stream>>>(s.ab,s.mods,s.forward,s.n,h,s.n/h);
    mark(s,3,stream);
    dim3 inverse_full(full.x,PrimeCount),inverse_half(half.x,PrimeCount);
    product_rns<<<inverse_full,128,0,stream>>>(forward_values,s.c,s.mods,s.n,s.logn);
    mark(s,4,stream);
    if(s.n>=Tile) {
        dim3 tiles(s.n/Tile,PrimeCount);
        small_rns<<<tiles,Tile/2,0,stream>>>(s.c,s.mods,s.small_inverse,s.n);
    }
    uint32_t* inverse_values=s.c;
    if(FHERMA_RNS_TAIL && s.n==32768) {
        dim3 transposes(32,PrimeCount),tails(128,PrimeCount);
        transpose_rns<<<transposes,256,0,stream>>>(s.c,s.ab,s.n);
        tail_rns<<<tails,128,0,stream>>>(s.ab,s.mods,s.tail_inverse,s.n);
        inverse_values=s.ab;
    } else for(unsigned h=first;h<s.n;h*=2)
        stage_rns<<<inverse_half,128,0,stream>>>(s.c,s.mods,s.inverse,s.n,h,s.n/h);
    mark(s,5,stream);
    reconstruct_rns<<<full.x,128,0,stream>>>(inverse_values,s.input,s.mods,s.scale,s.bases,
                                            s.q,s.product_mod_q,s.n);
    mark(s,6,stream);
    check(cudaGetLastError(),"RNS kernels");
}
} // namespace

void* fherma_init(const fherma::Point& p) {
    if(p.N<2 || (p.N&(p.N-1)) || p.N>32768 || p.W!=868 || p.L!=rns::AbiWords || p.q.data.size()!=rns::AbiWords)
        throw std::runtime_error("RNS coverage: power-of-two 2<=N<=32768, W=868, L=28");
    uint32_t delta=uint32_t(0)-p.q.data[0];
    bool special=delta>0 && delta<0x10000000u && p.q.data[27]==15u;
    for(unsigned k=1;k<27;++k) special=special && p.q.data[k]==0xffffffffu;
    if(!special) throw std::runtime_error("RNS coverage: q=2^868-c, c<2^28");
    static_assert(!FHERMA_PROFILE || FHERMA_GRAPH,"RNS profiling uses graph events");
    pin_near_gpu();
    auto s=std::make_unique<State>();s->n=p.N;s->logn=__builtin_ctz(p.N);
    auto constants=rns::setup(p.N,p.q.data);
    upload(s->mods,constants.mods);upload(s->bases,constants.bases_mod_q);
    constants.product_mod_q.resize(rns::AccumWords,0);
    upload(s->product_mod_q,constants.product_mod_q);upload(s->q,p.q.data);
    upload(s->forward,constants.forward);upload(s->inverse,constants.inverse);upload(s->scale,constants.scale);
    upload(s->small_forward,constants.small_forward);upload(s->small_inverse,constants.small_inverse);
    if(FHERMA_RNS_TAIL && p.N==32768) {
        std::vector<rns::Twiddle> forward_tail(size_t(rns::PrimeCount)*p.N),inverse_tail(forward_tail.size());
        for(unsigned pi=0;pi<rns::PrimeCount;++pi)
            for(unsigned half=1;half<32;half*=2)
                for(unsigned col=0;col<1024;++col)
                    for(unsigned j=0;j<half;++j) {
                        unsigned index=pi*p.N+1024*(half-1)+col*half+j;
                        unsigned source=pi*p.N+(col+1024*j)*(32/half);
                        forward_tail[index]=constants.forward[source];inverse_tail[index]=constants.inverse[source];
                    }
        upload(s->tail_forward,forward_tail);upload(s->tail_inverse,inverse_tail);
        check(cudaMalloc(&s->scratch,size_t(2)*rns::PrimeCount*p.N*4),"RNS transpose scratch");
    }
    size_t bytes=size_t(p.N)*rns::AbiWords*4;
    check(cudaMalloc(&s->input,2*bytes),"allocate ABI buffers");
    check(cudaMalloc(&s->ab,size_t(2)*rns::PrimeCount*p.N*4),"allocate RNS operands");
    check(cudaMalloc(&s->c,size_t(rns::PrimeCount)*p.N*4),"allocate RNS inverse");
    check(cudaMallocHost(reinterpret_cast<void**>(&s->host_input),2*bytes),"pinned RNS inputs");
    check(cudaMallocHost(reinterpret_cast<void**>(&s->host_output),bytes),"pinned RNS output");
    std::memset(s->host_input,0,2*bytes);std::memset(s->host_output,0,bytes);
#if FHERMA_GRAPH
#if FHERMA_PROFILE
    for(auto& event:s->events) check(cudaEventCreate(&event),"RNS profile event");
#endif
    check(cudaStreamCreateWithFlags(&s->stream,cudaStreamNonBlocking),"RNS stream");
    check(cudaStreamBeginCapture(s->stream,cudaStreamCaptureModeGlobal),"capture RNS graph");
    mark(*s,0,s->stream);
    check(cudaMemcpyAsync(s->input,s->host_input,2*bytes,cudaMemcpyHostToDevice,s->stream),"capture RNS H2D");
    mark(*s,1,s->stream);
    launch_rns(*s,s->stream);
    check(cudaMemcpyAsync(s->host_output,s->input,bytes,cudaMemcpyDeviceToHost,s->stream),"capture RNS D2H");
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
    size_t words=size_t(s.n)*rns::AbiWords,bytes=words*4;
    if(input.a.data.size()!=words || input.b.data.size()!=words) throw std::runtime_error("RNS input size");
#if FHERMA_PROFILE
    auto pack_start=std::chrono::steady_clock::now();
#endif
    s.copy.inputs(s.host_input,input.a.data.data(),input.b.data.data(),bytes);
#if FHERMA_PROFILE
    auto pack_end=std::chrono::steady_clock::now();
#endif
    fherma::Outputs output;output.c.shape={s.n,rns::AbiWords};
    try {
#if FHERMA_GRAPH
        check(cudaGraphLaunch(s.graph,s.stream),"execute RNS graph");
#else
        check(cudaMemcpy(s.input,s.host_input,2*bytes,cudaMemcpyHostToDevice),"RNS H2D");
        launch_rns(s);
#endif
        output.c.data.reserve(words);
        s.copy.prefault(output.c.data.data(),bytes);
        output.c.data.resize(words);
#if FHERMA_GRAPH
        check(cudaStreamSynchronize(s.stream),"RNS output ready");
#else
        check(cudaMemcpy(s.host_output,s.input,bytes,cudaMemcpyDeviceToHost),"RNS D2H");
#endif
    } catch(...) {
#if FHERMA_GRAPH
        cudaStreamSynchronize(s.stream);
#else
        cudaDeviceSynchronize();
#endif
        throw;
    }
#if FHERMA_PROFILE
    auto unpack_start=std::chrono::steady_clock::now();
#endif
    s.copy.output(output.c.data.data(),s.host_output,bytes);
#if FHERMA_PROFILE
    auto unpack_end=std::chrono::steady_clock::now();
    const char* names[]={"h2d","prepare","forward","product","inverse","finish","d2h"};
    std::fprintf(stderr,"PROFILE_US");
    for(unsigned k=0;k<7;++k) {
        float ms=0;check(cudaEventElapsedTime(&ms,s.events[k],s.events[k+1]),"RNS profile time");
        std::fprintf(stderr," %s=%.3f",names[k],1000*ms);
    }
    auto us=[](auto a,auto b) {return std::chrono::duration<double,std::micro>(b-a).count();};
    std::fprintf(stderr,"\nHOST_US pack=%.3f unpack=%.3f\n",us(pack_start,pack_end),us(unpack_start,unpack_end));
#endif
    return output;
}
void fherma_free(void* state) { delete static_cast<State*>(state); }
