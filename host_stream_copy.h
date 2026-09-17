#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
__attribute__((target("avx512f")))
inline void stream_copy_avx512(void* target,const void* source,size_t bytes) {
    auto* out=static_cast<char*>(target);
    auto* in=static_cast<const char*>(source);
    size_t prefix=(-reinterpret_cast<uintptr_t>(out))&63;
    if(prefix>bytes) prefix=bytes;
    std::memcpy(out,in,prefix); out+=prefix; in+=prefix; bytes-=prefix;
    while(bytes>=256) {
        auto a=_mm512_loadu_si512(in);
        auto b=_mm512_loadu_si512(in+64);
        auto c=_mm512_loadu_si512(in+128);
        auto d=_mm512_loadu_si512(in+192);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out),a);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+64),b);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+128),c);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+192),d);
        out+=256; in+=256; bytes-=256;
    }
    while(bytes>=64) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out),_mm512_loadu_si512(in));
        out+=64; in+=64; bytes-=64;
    }
    std::memcpy(out,in,bytes);
    _mm_sfence(); // Complete streaming stores before handing the buffer to CUDA.
}
#endif
inline void host_copy_bytes(void* out,const void* in,size_t bytes) {
#if defined(__x86_64__) && defined(__GNUC__) && FHERMA_STREAM_COPY
    if(bytes>=4096 && __builtin_cpu_supports("avx512f")) {
        stream_copy_avx512(out,in,bytes); return;
    }
#endif
    std::memcpy(out,in,bytes);
}
