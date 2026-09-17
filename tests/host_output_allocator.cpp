#include "host_output_allocator.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

int main() {
    HostOutputAllocator allocator;
    std::vector<std::vector<uint32_t>> retained;
    for(unsigned trial=0;trial<100;++trial) {
        std::vector<uint32_t> output;
        size_t words=trial%2 ? 917504 : trial*19;
        if(trial%4) {
            output.reserve(words);
            auto* storage=reinterpret_cast<char*>(output.data());
            for(size_t byte=0;byte<words*4;byte+=4096) storage[byte]=char(0x7f);
        }
        {
            HostOutputAllocator::Work work(allocator,output,words);
            if(trial%3) work.wait(); // Other iterations exercise the RAII join.
        }
        assert(std::all_of(output.begin(),output.end(),[](auto x) {return x==0;}));
        std::fill(output.begin(),output.end(),trial+1);
        if(trial%10==1) retained.push_back(std::move(output));
    }
    for(unsigned i=0;i<retained.size();++i)
        assert(std::all_of(retained[i].begin(),retained[i].end(),[&](auto x) {return x==10*i+2;}));
    bool caught=false;
    try {
        std::vector<uint32_t> output;
        HostOutputAllocator::Work work(allocator,output,std::numeric_limits<size_t>::max());
        work.wait();
    } catch(const std::length_error&) {caught=true;}
    assert(caught);
    std::vector<uint32_t> output;
    try {
        HostOutputAllocator::Work work(allocator,output,917504);
        throw std::runtime_error("caller failure while allocation is in flight");
    } catch(const std::runtime_error&) {}
    assert(output.size()==917504 && output.back()==0);
    std::puts("100 fresh outputs, retained ownership, worker failure and caller unwind passed");
}
