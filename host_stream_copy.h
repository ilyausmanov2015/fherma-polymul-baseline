#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
__attribute__((target("movdir64b")))
inline void direct_copy64(void* target,const void* source,size_t bytes) {
    auto* out=static_cast<char*>(target);
    auto* in=static_cast<const char*>(source);
    size_t prefix=(-reinterpret_cast<uintptr_t>(out))&63;
    if(prefix>bytes) prefix=bytes;
    std::memcpy(out,in,prefix);out+=prefix;in+=prefix;bytes-=prefix;
    while(bytes>=256) {
        _movdir64b(out,in);_movdir64b(out+64,in+64);
        _movdir64b(out+128,in+128);_movdir64b(out+192,in+192);
        out+=256;in+=256;bytes-=256;
    }
    while(bytes>=64) {_movdir64b(out,in);out+=64;in+=64;bytes-=64;}
    std::memcpy(out,in,bytes);
    _mm_sfence(); // Direct stores use weak ordering, just like streaming stores.
}
template<bool Streaming> __attribute__((target("avx512f")))
inline void vector_copy_avx512(void* target,const void* source,size_t bytes) {
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
        if constexpr(Streaming) {
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out),a);
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out+64),b);
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out+128),c);
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out+192),d);
        } else {
            _mm512_store_si512(reinterpret_cast<__m512i*>(out),a);
            _mm512_store_si512(reinterpret_cast<__m512i*>(out+64),b);
            _mm512_store_si512(reinterpret_cast<__m512i*>(out+128),c);
            _mm512_store_si512(reinterpret_cast<__m512i*>(out+192),d);
        }
        out+=256; in+=256; bytes-=256;
    }
    while(bytes>=64) {
        if constexpr(Streaming) _mm512_stream_si512(reinterpret_cast<__m512i*>(out),_mm512_loadu_si512(in));
        else _mm512_store_si512(reinterpret_cast<__m512i*>(out),_mm512_loadu_si512(in));
        out+=64; in+=64; bytes-=64;
    }
    std::memcpy(out,in,bytes);
    if constexpr(Streaming) _mm_sfence(); // Publish non-temporal stores before CUDA.
}
#endif
inline void host_copy_bytes(void* out,const void* in,size_t bytes) {
#if defined(__x86_64__) && defined(__GNUC__) && FHERMA_DIRECT_COPY
    if(bytes>=4096 && __builtin_cpu_supports("movdir64b")) {
        direct_copy64(out,in,bytes);return;
    }
#endif
#if defined(__x86_64__) && defined(__GNUC__) && FHERMA_STREAM_COPY
    if(bytes>=4096 && __builtin_cpu_supports("avx512f")) {
        vector_copy_avx512<true>(out,in,bytes); return;
    }
#endif
    std::memcpy(out,in,bytes);
}

inline void host_copy_cached(void* out,const void* in,size_t bytes) {
#if defined(__x86_64__) && defined(__GNUC__)
    if(bytes>=4096 && __builtin_cpu_supports("avx512f")) {
        vector_copy_avx512<false>(out,in,bytes);return;
    }
#endif
    std::memcpy(out,in,bytes);
}

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("clflushopt"))) inline void discard_cached_reads_x86(const void* source,size_t bytes) {
    uintptr_t first=(reinterpret_cast<uintptr_t>(source)+63)&~uintptr_t(63);
    uintptr_t end=(reinterpret_cast<uintptr_t>(source)+bytes)&~uintptr_t(63);
    for(uintptr_t address=first;address<end;address+=64)
        _mm_clflushopt(reinterpret_cast<void*>(address));
    _mm_sfence(); // Complete cache maintenance before publishing CPU completion.
}
#endif
inline void host_discard_cached_reads(const void* source,size_t bytes) {
#if defined(__x86_64__) && defined(__GNUC__)
    if(bytes>=64 && __builtin_cpu_supports("clflushopt")) discard_cached_reads_x86(source,bytes);
#else
    (void)source;(void)bytes;
#endif
}
