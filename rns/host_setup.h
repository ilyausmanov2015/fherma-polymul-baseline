#pragma once
// Parameter-only setup for exact integer convolution via 57 small primes.
#include "wide_host.h"
#include <limits>

namespace rns {
constexpr unsigned PrimeCount=57,WideWords=56,AccumWords=29,AbiWords=28,Tile=1024;
static_assert(AccumWords*32>=905,"compact CRT accumulation must fit the exact bound");
struct Twiddle { uint32_t value,shoup; };
struct SmallMod { uint32_t p,base,base_shoup,padding=0; uint64_t reciprocal; };
using Words=std::vector<uint32_t>;
inline uint32_t pow_mod(uint32_t a,uint32_t e,uint32_t p) {
    uint32_t result=1;
    while(e) { if(e&1) result=uint64_t(result)*a%p; e>>=1; if(e) a=uint64_t(a)*a%p; }
    return result;
}
inline bool prime(uint32_t p) {
    if(p<2) return false;
    if((p&1)==0) return p==2;
    for(uint32_t d=3;uint64_t(d)*d<=p;d+=2) if(p%d==0) return false;
    return true;
}
inline void mul_small(Words& a,uint32_t scalar) {
    uint64_t carry=0;
    for(auto& word:a) { uint64_t x=uint64_t(word)*scalar+carry; word=uint32_t(x); carry=x>>32; }
    if(carry) throw std::runtime_error("RNS setup integer overflow");
}
inline Words multiply(const Words& a,const Words& b) {
    Words result(a.size()+b.size());
    for(size_t i=0;i<a.size();++i) {
        uint64_t carry=0;
        for(size_t j=0;j<b.size();++j) {
            uint64_t x=uint64_t(a[i])*b[j]+result[i+j]+carry;
            result[i+j]=uint32_t(x); carry=x>>32;
        }
        result[i+b.size()]=uint32_t(carry);
    }
    return result;
}
inline uint32_t mod_small(const Words& a,uint32_t p) {
    uint64_t remainder=0;
    for(size_t i=a.size();i--;) remainder=((remainder<<32)|a[i])%p;
    return uint32_t(remainder);
}
inline Words mod_wide(const Words& a,const Words& q) {
    Words result(q.size()),one=host_wide::small(q.size(),1);
    for(unsigned bit=host_wide::bits(a);bit--;) {
        host_wide::add_mod_inplace(result,result,q);
        if((a[bit/32]>>(bit%32))&1) host_wide::add_mod_inplace(result,one,q);
    }
    return result;
}
inline Twiddle twiddle(uint32_t value,uint32_t p) {
    return {value,uint32_t((uint64_t(value)<<32)/p)};
}
struct Setup {
    std::vector<SmallMod> mods;
    std::vector<Twiddle> forward,inverse,scale,small_forward,small_inverse,input_powers;
    Words product,half_ceil,product_mod_q,bases,bases_mod_q;
};
inline Setup setup(unsigned n,const Words& q) {
    Setup s;
    s.product=host_wide::small(WideWords,1);
    uint32_t candidate=(uint32_t(1)<<31)-65535;
    while(s.mods.size()<PrimeCount) {
        if(candidate<=std::numeric_limits<uint32_t>::max()/3)
            throw std::runtime_error("RNS primes outside the two-subtraction input range");
        if(prime(candidate)) {
            uint32_t base=(uint64_t(1)<<32)%candidate;
            s.mods.push_back({candidate,base,twiddle(base,candidate).shoup,0,
                              std::numeric_limits<uint64_t>::max()/candidate});
            mul_small(s.product,candidate);
        }
        candidate-=65536;
    }
    auto qm1=host_wide::sub(q,host_wide::small(q.size(),1));
    auto bound=multiply(qm1,qm1); mul_small(bound,4*n);
    if(host_wide::cmp(s.product,bound)<=0) throw std::runtime_error("CRT range too small for exact nearest-integer correction");
    auto sum_bound=s.product; mul_small(sum_bound,PrimeCount);
    s.half_ceil=host_wide::divide_small(s.product,2);
    uint64_t carry=1;
    for(auto& word:s.half_ceil) { uint64_t x=uint64_t(word)+carry;word=uint32_t(x);carry=x>>32; }
    s.product_mod_q=mod_wide(s.product,q);
    s.bases.reserve(PrimeCount*WideWords);
    s.bases_mod_q.reserve(PrimeCount*AccumWords);
    s.input_powers.resize(PrimeCount*AbiWords);
    s.forward.resize(size_t(PrimeCount)*n);s.inverse.resize(size_t(PrimeCount)*n);
    s.scale.resize(size_t(PrimeCount)*n);
    if(n>=Tile) {s.small_forward.resize(PrimeCount*Tile);s.small_inverse.resize(PrimeCount*Tile);}
    for(unsigned prime_i=0;prime_i<PrimeCount;++prime_i) {
        uint32_t p=s.mods[prime_i].p,remainder=0;
        uint32_t limb_power=1;
        for(unsigned limb=0;limb<AbiWords;++limb) {
            s.input_powers[prime_i*AbiWords+limb]=twiddle(limb_power,p);
            limb_power=uint64_t(limb_power)*s.mods[prime_i].base%p;
        }
        auto basis=host_wide::divide_small(s.product,p,&remainder);
        if(remainder) throw std::runtime_error("CRT basis division not exact");
        s.bases.insert(s.bases.end(),basis.begin(),basis.end());
        auto reduced_basis=mod_wide(basis,q);reduced_basis.resize(AccumWords,0);
        s.bases_mod_q.insert(s.bases_mod_q.end(),reduced_basis.begin(),reduced_basis.end());
        uint32_t basis_inverse=pow_mod(mod_small(basis,p),p-2,p);
        uint32_t psi=0;
        for(uint32_t g=2;g<10000;++g) {
            uint32_t root=pow_mod(g,(p-1)/(2*n),p);
            if(pow_mod(root,n,p)==p-1) {psi=root;break;}
        }
        if(!psi) throw std::runtime_error("RNS root not found");
        uint32_t inverse_psi=pow_mod(psi,p-2,p);
        uint32_t inverse_n=pow_mod(n,p-2,p);
        uint32_t weighted_scale=uint64_t(inverse_n)*basis_inverse%p;
        uint32_t f=1,b=1,scaled=weighted_scale;
        for(unsigned i=0;i<n;++i) {
            size_t index=size_t(prime_i)*n+i;
            s.forward[index]=twiddle(f,p);s.inverse[index]=twiddle(b,p);
            s.scale[index]=twiddle(scaled,p);
            f=uint64_t(f)*psi%p;b=uint64_t(b)*inverse_psi%p;
            scaled=uint64_t(scaled)*inverse_psi%p;
        }
        if(n>=Tile) for(unsigned i=0;i<Tile;++i) {
            s.small_forward[prime_i*Tile+i]=s.forward[size_t(prime_i)*n+i*(n/Tile)];
            s.small_inverse[prime_i*Tile+i]=s.inverse[size_t(prime_i)*n+i*(n/Tile)];
        }
    }
    return s;
}
} // namespace rns
