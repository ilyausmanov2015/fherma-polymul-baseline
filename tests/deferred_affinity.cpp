#include "host_copy_pool.h"
#include <cassert>
#include <cstdio>

int main() {
#ifdef __linux__
    static_assert(FHERMA_DEFER_MAIN_PIN,"test requires deferred pinning");
    cpu_set_t original,after,child;
    assert(sched_getaffinity(0,sizeof(original),&original)==0);
    {
        HostCopyPool pool;
        assert(sched_getaffinity(0,sizeof(after),&after)==0);
        assert(CPU_EQUAL(&original,&after));
        std::thread helper([&] { assert(sched_getaffinity(0,sizeof(child),&child)==0); });
        helper.join();
        assert(CPU_EQUAL(&original,&child));
        pool.pin_caller();
        assert(sched_getaffinity(0,sizeof(after),&after)==0);
        assert(CPU_COUNT(&after)==1);
        for(int cpu=0;cpu<CPU_SETSIZE;++cpu)
            if(CPU_ISSET(cpu,&after)) assert(CPU_ISSET(cpu,&original));
        unsigned a[19],b[19],out[38];
        for(unsigned i=0;i<19;++i) {a[i]=i;b[i]=100+i;}
        pool.inputs(out,a,b,sizeof(a));
        assert(std::memcmp(out,a,sizeof(a))==0);
        assert(std::memcmp(out+19,b,sizeof(b))==0);
    }
    assert(sched_setaffinity(0,sizeof(original),&original)==0);
    std::puts("deferred pinning: helper inheritance, caller pin and copy passed");
#else
    std::puts("deferred affinity: Linux-only placement check skipped");
#endif
}
