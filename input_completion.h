#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>

// Host-produced readiness flags for a graph's input DMA or mapped prepare nodes. The graph never
// owns or reads caller storage; it only reads our staged input after publication.
template<unsigned Parts> class InputCompletion {
    uint32_t* host_=nullptr;
    CUdeviceptr device_=0;
    static void driver_check(CUresult status,const char* operation) {
        if(status==CUDA_SUCCESS) return;
        const char* description=nullptr;cuGetErrorString(status,&description);
        throw std::runtime_error(std::string(operation)+": "+(description ? description : "CUDA driver error"));
    }
    static void runtime_check(cudaError_t status,const char* operation) {
        if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
    }
public:
    InputCompletion()=default;
    InputCompletion(const InputCompletion&)=delete;
    ~InputCompletion() {if(host_) cudaFreeHost(host_);}
    bool init(cudaStream_t stream) {
        runtime_check(cudaHostAlloc(reinterpret_cast<void**>(&host_),Parts*64,cudaHostAllocMapped),"allocate input flags");
        std::memset(host_,0,Parts*64);
        driver_check(cuMemHostGetDevicePointer(&device_,host_,0),"map input flags");
        // Probe only an already-satisfied wait on synchronization metadata.
        publish(0);
        auto status=cuStreamWaitValue32(stream,device_,1,CU_STREAM_WAIT_VALUE_EQ);
        if(status==CUDA_ERROR_NOT_SUPPORTED) {std::fprintf(stderr,"INPUT_GRAPH supported=0\n");return false;}
        driver_check(status,"probe input wait");
        runtime_check(cudaStreamSynchronize(stream),"input wait probe ready");
        runtime_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"probe input wait capture");
        status=cuStreamWaitValue32(stream,device_,1,CU_STREAM_WAIT_VALUE_EQ);
        cudaGraph_t graph=nullptr;auto ended=cudaStreamEndCapture(stream,&graph);
        if(graph) cudaGraphDestroy(graph);
        if(status==CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED || status==CUDA_ERROR_NOT_SUPPORTED) {
            if(ended!=cudaSuccess && ended!=cudaErrorStreamCaptureInvalidated)
                runtime_check(ended,"discard unsupported input capture");
            (void)cudaGetLastError();
            std::fprintf(stderr,"INPUT_GRAPH supported=0 capture_unsupported=1\n");return false;
        }
        driver_check(status,"capture input wait probe");runtime_check(ended,"finish input wait probe");
        reset();std::fprintf(stderr,"INPUT_GRAPH supported=1\n");return true;
    }
    void reset() {
        for(unsigned part=0;part<Parts;++part) __atomic_store_n(host_+16*part,0u,__ATOMIC_RELEASE);
    }
    void publish(unsigned part) {__atomic_store_n(host_+16*part,1u,__ATOMIC_RELEASE);}
    void release_all() {for(unsigned part=0;part<Parts;++part) publish(part);}
    void capture_wait(cudaStream_t stream,unsigned part) {
        driver_check(cuStreamWaitValue32(stream,device_+64*part,1,CU_STREAM_WAIT_VALUE_EQ),"capture staged input wait");
    }
};
