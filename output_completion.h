#pragma once
// Optional CUDA stream-to-host completion notification. No input or output
// data is retained here: each cache line holds only a run generation number.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

template<unsigned Parts> class OutputCompletion {
    uint32_t* host_=nullptr;
    CUdeviceptr device_=0;
    uint32_t generation_=0;
    bool supported_=false,graph_=false;
    static void driver_check(CUresult status,const char* operation) {
        if(status==CUDA_SUCCESS) return;
        const char* description=nullptr;
        cuGetErrorString(status,&description);
        throw std::runtime_error(std::string(operation)+": "+(description ? description : "CUDA driver error"));
    }
    static void runtime_check(cudaError_t status,const char* operation) {
        if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
    }
public:
    OutputCompletion()=default;
    OutputCompletion(const OutputCompletion&)=delete;
    ~OutputCompletion() { if(host_) cudaFreeHost(host_); }
    void init(cudaStream_t stream,bool graph=false) {
        graph_=graph;
        runtime_check(cudaHostAlloc(reinterpret_cast<void**>(&host_),Parts*64,cudaHostAllocMapped),"allocate completion flags");
        std::memset(host_,0,Parts*64);
        driver_check(cuMemHostGetDevicePointer(&device_,host_,0),"map completion flags");
        // Probe this optional operation on empty synchronization storage only.
        auto status=cuStreamWriteValue32(stream,device_,0,CU_STREAM_WRITE_VALUE_DEFAULT);
        if(status==CUDA_ERROR_NOT_SUPPORTED) {
            std::fprintf(stderr,"OUTPUT_COMPLETION supported=0\n");return;
        }
        driver_check(status,"probe output completion");
        runtime_check(cudaStreamSynchronize(stream),"completion probe ready");
        if(graph_) {
            runtime_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"probe completion capture");
            status=cuStreamWriteValue32(stream,device_,0,CU_STREAM_WRITE_VALUE_DEFAULT);
            cudaGraph_t definition=nullptr;
            auto ended=cudaStreamEndCapture(stream,&definition);
            if(definition) cudaGraphDestroy(definition);
            if(status==CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED || status==CUDA_ERROR_NOT_SUPPORTED) {
                if(ended!=cudaSuccess && ended!=cudaErrorStreamCaptureInvalidated)
                    runtime_check(ended,"discard unsupported completion capture");
                (void)cudaGetLastError(); // Clear only the expected capture error.
                std::fprintf(stderr,"OUTPUT_COMPLETION supported=0 capture_unsupported=1\n");return;
            }
            driver_check(status,"capture completion probe write");
            runtime_check(ended,"finish completion capture probe");
            generation_=1;
        }
        supported_=true;
        std::fprintf(stderr,"OUTPUT_COMPLETION supported=1 graph=%d\n",graph_);
    }
    bool enabled() const { return supported_; }
    void begin() {
        if(!supported_) return;
        if(graph_) {
            // Every previous flag was observed before the preceding run ended.
            // Reset them before launching the graph, whose writes use value 1.
            for(unsigned part=0;part<Parts;++part) __atomic_store_n(host_+part*16,0u,__ATOMIC_RELEASE);
        } else ++generation_;
    }
    void record(cudaStream_t stream,unsigned part) {
        // Default flags include a system-scope fence for all preceding writes
        // in this stream. Never use NO_MEMORY_BARRIER for host-visible output.
        driver_check(cuStreamWriteValue32(stream,device_+part*64,generation_,CU_STREAM_WRITE_VALUE_DEFAULT),"record output completion");
    }
    void wait(cudaStream_t stream,unsigned part) {
        // The GPU is the sole writer; aligned 32-bit driver stores publish the
        // preceding DMA. An acquire host load prevents reads moving above it.
        for(unsigned spins=0;__atomic_load_n(host_+part*16,__ATOMIC_ACQUIRE)!=generation_;++spins) {
            if(spins==131072) {
                // Bound CPU polling and surface asynchronous GPU failures.
                runtime_check(cudaStreamSynchronize(stream),"output completion slow wait");
                if(__atomic_load_n(host_+part*16,__ATOMIC_ACQUIRE)!=generation_)
                    throw std::runtime_error("output completion flag missing after stream synchronization");
                break;
            }
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
};
