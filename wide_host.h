#pragma once
// Dependency-free host arithmetic, used only for untimed NTT setup.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace host_wide {
using Int = std::vector<uint32_t>;
inline int cmp(const Int& a, const Int& b) {
    for (size_t i=a.size(); i--;) if (a[i]!=b[i]) return a[i]<b[i] ? -1 : 1;
    return 0;
}
inline Int sub(const Int& a, const Int& b) {
    Int c(a.size()); uint64_t borrow=0;
    for (size_t i=0;i<a.size();++i) {
        uint64_t v=uint64_t(b[i])+borrow;
        c[i]=uint32_t(uint64_t(a[i])-v); borrow=uint64_t(a[i])<v;
    }
    return c;
}
inline Int small(size_t limbs, uint32_t n) { Int r(limbs); r[0]=n; return r; }
inline void add_mod_inplace(Int& a, const Int& b, const Int& q) {
    uint64_t carry=0;
    for (size_t i=0;i<a.size();++i) {
        uint64_t v=uint64_t(a[i])+b[i]+carry; a[i]=uint32_t(v); carry=v>>32;
    }
    if (carry || cmp(a,q)>=0) {
        uint64_t borrow=0;
        for (size_t i=0;i<a.size();++i) {
            uint64_t v=uint64_t(q[i])+borrow;
            uint32_t old=a[i]; a[i]=uint32_t(uint64_t(old)-v); borrow=uint64_t(old)<v;
        }
    }
}
inline unsigned bits(const Int& n) {
    for (size_t i=n.size();i--;) if (n[i]) return unsigned(i*32+32-__builtin_clz(n[i]));
    return 0;
}
inline Int mul(Int a, const Int& b, const Int& q) {
    Int c(a.size()); unsigned width=bits(b);
    for (unsigned i=0;i<width;++i) {
        if ((b[i/32]>>(i%32))&1) add_mod_inplace(c,a,q);
        if (i+1<width) add_mod_inplace(a,a,q);
    }
    return c;
}
inline Int pow(Int a, const Int& e, const Int& q) {
    Int c=small(q.size(),1); unsigned width=bits(e);
    for (unsigned i=0;i<width;++i) {
        if ((e[i/32]>>(i%32))&1) c=mul(c,a,q);
        if (i+1<width) a=mul(a,a,q);
    }
    return c;
}
inline Int divide_small(const Int& a, uint32_t d, uint32_t* remainder=nullptr) {
    Int r(a.size()); uint64_t carry=0;
    for (size_t i=a.size();i--;) {
        uint64_t v=(carry<<32)|a[i]; r[i]=uint32_t(v/d); carry=v%d;
    }
    if (remainder) *remainder=uint32_t(carry);
    return r;
}
struct Roots { Int psi, invpsi, invn; };
inline Roots roots(const Int& q, uint32_t n) {
    if (!n || (n&(n-1)) || n>(1u<<29) || q.empty()) throw std::runtime_error("invalid N or q");
    const auto one=small(q.size(),1);
    if (cmp(q,small(q.size(),2))<=0) throw std::runtime_error("q must be an odd prime");
    const auto qm1=sub(q,one), qm2=sub(q,small(q.size(),2));
    uint32_t remainder=0;
    const auto exponent=divide_small(qm1,2*n,&remainder);
    if (remainder) throw std::runtime_error("2N must divide q-1");
    for (uint32_t g=2; g<10000; ++g) {
        auto base=small(q.size(),g);
        if (cmp(base,q)>=0) break;
        auto psi=pow(base,exponent,q);
        if (cmp(pow(psi,small(q.size(),n),q),qm1)==0)
            return {psi,pow(psi,qm2,q),pow(small(q.size(),n),qm2,q)};
    }
    throw std::runtime_error("failed to find primitive 2N-th root");
}
}
