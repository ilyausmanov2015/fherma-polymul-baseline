#include "host_cpu_topology.h"
#include <cassert>
#include <set>
#include <cstdio>

int main() {
    std::vector<HostCpuCore> cpus;
    for(int cpu=0;cpu<24;++cpu) cpus.push_back({cpu,0,cpu/2});
    auto order=host_worker_cpu_order(cpus,0);
    assert(order.size()==23 && order.back()==1);
    for(int i=0;i<11;++i) assert(order[i]==2*(i+1));
    for(int i=11;i<22;++i) assert(order[i]==2*(i-11)+3);
    assert(std::set<int>(order.begin(),order.end()).size()==23);
    for(int caller=0;caller<24;++caller) {
        order=host_worker_cpu_order(cpus,caller);
        assert(order.size()==23 && order.back()==(caller^1));
        assert(std::find(order.begin(),order.end(),caller)==order.end());
        std::set<int> first_cores;
        for(int i=0;i<11;++i) first_cores.insert(order[i]/2);
        assert(first_cores.size()==11 && !first_cores.count(caller/2));
        std::vector<int> original;
        for(const auto& cpu:cpus) if(cpu.cpu!=caller) original.push_back(cpu.cpu);
        for(unsigned prefix=0;prefix<=26;++prefix) {
            auto hybrid=host_worker_cpu_order(cpus,caller,prefix);
            assert(hybrid.size()==23 && std::set<int>(hybrid.begin(),hybrid.end()).size()==23);
            assert(std::find(hybrid.begin(),hybrid.end(),caller)==hybrid.end());
            for(unsigned i=0;i<std::min(prefix,23u);++i) assert(hybrid[i]==original[i]);
        }
    }
    auto hybrid=host_worker_cpu_order(cpus,0,7);
    for(int i=0;i<7;++i) assert(hybrid[i]==i+1);
    for(int i=7;i<15;++i) assert(hybrid[i]==8+2*(i-7));
    // Sparse cpuset, core IDs repeated on another package, missing metadata.
    cpus={{4,0,2},{5,0,2},{18,0,9},{42,1,2},{43,1,2},{91,-1,-1}};
    assert((host_worker_cpu_order(cpus,4)==std::vector<int>{18,42,91,43,5}));
    assert((host_worker_cpu_order(cpus,4,2)==std::vector<int>{5,18,42,91,43}));
    cpus={{3,-1,-1},{7,-1,-1},{9,-1,-1}};
    assert((host_worker_cpu_order(cpus,7)==std::vector<int>{3,9}));
    assert(host_worker_cpu_order({{7,0,0}},7).empty());
    std::puts("CPU topology: 24 callers, sparse masks, packages and missing metadata passed");
}
