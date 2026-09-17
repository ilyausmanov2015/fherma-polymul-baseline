#pragma once
// Optional NUMA locality: choose an allowed CPU on the GPU's NUMA node.
// This changes only our process placement, before allocating staging buffers.
// No input data is available here. Missing topology leaves placement unchanged.
#include <cstdio>
#ifdef __linux__
#include <sched.h>
#include <fstream>
#include <sstream>
#include <string>
#include <dirent.h>
#endif

inline void pin_near_gpu() {
#ifdef __linux__
    char bus[32]{};
    if(cudaDeviceGetPCIBusId(bus,sizeof(bus),0)!=cudaSuccess) return;
    unsigned domain,b,d,f;
    if(std::sscanf(bus,"%x:%x:%x.%x",&domain,&b,&d,&f)!=4) return;
    char canonical[32]; std::snprintf(canonical,sizeof(canonical),"%04x:%02x:%02x.%x",domain,b,d,f);
    int node=-1;
    std::ifstream(std::string("/sys/bus/pci/devices/")+canonical+"/numa_node")>>node;
    if(node<0) return;
    std::string cpus;
    std::ifstream("/sys/devices/system/node/node"+std::to_string(node)+"/cpulist")>>cpus;
    cpu_set_t allowed,chosen;
    if(sched_getaffinity(0,sizeof(allowed),&allowed)!=0) return;
    CPU_ZERO(&chosen);
    std::stringstream parts(cpus); std::string part;
    int selected=-1, local=0, current=sched_getcpu();
    while(std::getline(parts,part,',')) {
        int first=-1,last=-1;
        int count=std::sscanf(part.c_str(),"%d-%d",&first,&last);
        if(count==1) last=first;
        if(count<1 || first<0) continue;
        for(int cpu=first;cpu<=last && cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&allowed)) {
            ++local;
            CPU_SET(cpu,&chosen);
            if(selected<0 || cpu==current) selected=cpu;
        }
    }
    if(selected<0) return;
    if(sched_setaffinity(0,sizeof(chosen),&chosen)==0)
        std::fprintf(stderr,"PLACEMENT gpu_node=%d cpu=%d allowed_on_node=%d\n",node,selected,local);
#endif
}

// Optional init-only diagnostic for our own helper/worker placement.
inline void report_thread_affinity() {
#ifdef __linux__
    DIR* directory=opendir("/proc/self/task");
    if(!directory) return;
    while(auto* entry=readdir(directory)) {
        if(entry->d_name[0]<'0' || entry->d_name[0]>'9') continue;
        std::ifstream status(std::string("/proc/self/task/")+entry->d_name+"/status");
        std::string line,name,cpus;
        while(std::getline(status,line)) {
            if(line.rfind("Name:",0)==0) name=line.substr(5);
            if(line.rfind("Cpus_allowed_list:",0)==0) cpus=line.substr(18);
        }
        std::fprintf(stderr,"THREAD_AFFINITY tid=%s name=%s cpus=%s\n",entry->d_name,name.c_str(),cpus.c_str());
    }
    closedir(directory);
#endif
}
