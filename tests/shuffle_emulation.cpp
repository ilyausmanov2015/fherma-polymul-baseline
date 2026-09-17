#include "tests/cuda_emulation.h"
#include <iostream>
EmulatedKernel shuffle_check(uint32_t* output,unsigned width) {
    unsigned t=threadIdx.x;
    uint32_t x=t+1000*blockIdx.x;
    for(unsigned delta=1;delta<width;delta*=2) {
        uint32_t a=co_await emulated_shuffle_xor(0xffffffff,x,delta,width);
        uint32_t b=co_await emulated_shuffle_xor(0xffffffff,3*x+7,delta,width);
        assert(a==((t^delta)+1000*blockIdx.x));
        assert(b==3*((t^delta)+1000*blockIdx.x)+7);
    }
    output[blockIdx.x*blockDim.x+t]=x;
}
int main() {
    uint32_t output[384]{};
    for(unsigned width: {4u,16u,32u}) {
        emulate_launch(dim3(3),dim3(128),[&] { return shuffle_check(output,width); });
        for(unsigned block=0;block<3;++block) for(unsigned t=0;t<128;++t)
            assert(output[128*block+t]==1000*block+t);
    }
    std::cout<<"Shuffle coroutine scheduling: 3 widths x 3 blocks x 128 lanes passed\n";
}
