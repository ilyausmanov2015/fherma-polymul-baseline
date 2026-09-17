#pragma once
#include <vector>
#include <algorithm>

struct HostCpuCore {
    int cpu=-1,package=-1,core=-1;
};
inline bool same_host_core(const HostCpuCore& a,const HostCpuCore& b) {
    return a.cpu==b.cpu || (a.package>=0 && a.core>=0 && b.package>=0 && b.core>=0 &&
        a.package==b.package && a.core==b.core);
}
// Only reorder the supplied allowed CPUs. Prefer one thread per physical core,
// fill other cores' SMT siblings next, and use caller siblings only as a last
// resort. Missing topology treats each logical CPU as a separate core.
inline std::vector<int> host_worker_cpu_order(const std::vector<HostCpuCore>& allowed,int caller) {
    HostCpuCore main{caller,-1,-1};
    for(const auto& cpu:allowed) if(cpu.cpu==caller) main=cpu;
    std::vector<HostCpuCore> selected;
    auto used=[&](int cpu) {
        return std::any_of(selected.begin(),selected.end(),[&](const auto& x) {return x.cpu==cpu;});
    };
    for(const auto& cpu:allowed) {
        if(same_host_core(cpu,main)) continue;
        if(std::none_of(selected.begin(),selected.end(),[&](const auto& x) {return same_host_core(x,cpu);}))
            selected.push_back(cpu);
    }
    for(const auto& cpu:allowed) if(!same_host_core(cpu,main) && !used(cpu.cpu)) selected.push_back(cpu);
    for(const auto& cpu:allowed) if(cpu.cpu!=caller && !used(cpu.cpu)) selected.push_back(cpu);
    std::vector<int> result;
    for(const auto& cpu:selected) result.push_back(cpu.cpu);
    return result;
}
