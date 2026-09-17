#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// A per-allocation hint on complete pages owned by the returned std::vector.
// Allocation, advice, prefault and initialization all remain inside run().
inline void advise_output_hugepages(void* storage,size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    long queried=sysconf(_SC_PAGESIZE);
    if(queried<=0) return;
    uintptr_t page=static_cast<uintptr_t>(queried),address=reinterpret_cast<uintptr_t>(storage);
    uintptr_t begin=(address+page-1)/page*page,end=(address+bytes)/page*page;
    if(end>begin) (void)madvise(reinterpret_cast<void*>(begin),end-begin,MADV_HUGEPAGE);
#else
    (void)storage;(void)bytes;
#endif
}
