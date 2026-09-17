#pragma once
#include "rns/host_setup.h"
namespace quartic {
constexpr unsigned ModCount=16,Components=4,PrimeCount=ModCount*Components;
constexpr unsigned AbiWords=28,PartBits=217,PartWords=7,WideWords=16,Tile=1024;
using rns::Words;using rns::SmallMod;using rns::Twiddle;
using rns::pow_mod;using rns::twiddle;
struct Roots { Twiddle powers[4],i; };
inline uint32_t square_root(uint32_t a,uint32_t p) {
    if(!a) return 0;
    if(pow_mod(a,(p-1)/2,p)!=1) throw std::runtime_error("quartic square root does not exist");
    uint32_t odd=p-1;unsigned order=0;
    while(!(odd&1)) {odd>>=1;++order;}
    uint32_t nonresidue=2;
    while(pow_mod(nonresidue,(p-1)/2,p)!=p-1) ++nonresidue;
    uint32_t c=pow_mod(nonresidue,odd,p),x=pow_mod(a,(odd+1)/2,p),t=pow_mod(a,odd,p);
    while(t!=1) {
        unsigned i=0;uint32_t value=t;
        do {value=uint64_t(value)*value%p;++i;} while(i<order && value!=1);
        if(i>=order) throw std::runtime_error("quartic Tonelli-Shanks invariant");
        uint32_t b=pow_mod(c,uint32_t(1)<<(order-i-1),p);
        x=uint64_t(x)*b%p;c=uint64_t(b)*b%p;t=uint64_t(t)*c%p;order=i;
    }
    if(uint64_t(x)*x%p!=a) throw std::runtime_error("quartic square root verification");
    return x;
}
struct Setup {
    std::vector<SmallMod> mods;
    std::vector<Roots> roots;
    std::vector<Twiddle> forward,inverse,scale,small_forward,small_inverse,input_powers;
    Words product,bases,bound;
};
inline Setup setup(unsigned n,uint32_t delta) {
    if(n<2 || n>32768 || (n&(n-1)) || delta==0 || delta>=0x10000000u)
        throw std::runtime_error("quartic parameter range");
    Setup s;s.product=host_wide::small(WideWords,1);
    uint32_t candidate=(uint32_t(1)<<31)-65535;
    while(s.mods.size()<ModCount) {
        if(candidate<=std::numeric_limits<uint32_t>::max()/3)
            throw std::runtime_error("quartic prime search exhausted");
        if(rns::prime(candidate) && pow_mod(delta,(candidate-1)/4,candidate)==1) {
            uint32_t base=(uint64_t(1)<<32)%candidate;
            s.mods.push_back({candidate,base,twiddle(base,candidate).shoup,0,
                              std::numeric_limits<uint64_t>::max()/candidate});
            rns::mul_small(s.product,candidate);
        }
        candidate-=65536;
    }
    Words part(PartWords,0xffffffffu);part.back()=(uint32_t(1)<<25)-1;
    s.bound=rns::multiply(part,part);s.bound.resize(WideWords);
    rns::mul_small(s.bound,n);rns::mul_small(s.bound,1+3*delta);
    auto four_bound=s.bound;rns::mul_small(four_bound,4);
    if(host_wide::cmp(s.product,four_bound)<=0) throw std::runtime_error("quartic CRT range insufficient");
    auto sum_bound=s.product;rns::mul_small(sum_bound,ModCount);
    s.roots.resize(ModCount);s.bases.reserve(ModCount*WideWords);
    s.input_powers.resize(ModCount*PartWords);
    s.forward.resize(size_t(ModCount)*n);s.inverse.resize(size_t(ModCount)*n);
    s.scale.resize(size_t(PrimeCount)*n);
    if(n>=Tile) {s.small_forward.resize(ModCount*Tile);s.small_inverse.resize(ModCount*Tile);}
    for(unsigned pi=0;pi<ModCount;++pi) {
        uint32_t p=s.mods[pi].p,r=square_root(square_root(delta,p),p);
        if(pow_mod(r,4,p)!=delta) throw std::runtime_error("quartic fourth root verification");
        uint32_t psi=0;
        for(uint32_t g=2;g<10000;++g) {
            uint32_t root=pow_mod(g,(p-1)/(2*n),p);
            if(pow_mod(root,n,p)==p-1) {psi=root;break;}
        }
        if(!psi) throw std::runtime_error("quartic NTT root not found");
        uint32_t imaginary=pow_mod(psi,n/2,p);
        if(uint64_t(imaginary)*imaginary%p!=p-1) throw std::runtime_error("quartic imaginary root verification");
        s.roots[pi].i=twiddle(imaginary,p);
        for(unsigned j=0;j<4;++j) s.roots[pi].powers[j]=twiddle(pow_mod(r,j,p),p);
        uint32_t remainder=0;auto basis=host_wide::divide_small(s.product,p,&remainder);
        if(remainder) throw std::runtime_error("quartic CRT basis division");
        s.bases.insert(s.bases.end(),basis.begin(),basis.end());
        uint32_t basis_inverse=pow_mod(rns::mod_small(basis,p),p-2,p);
        uint32_t inverse_psi=pow_mod(psi,p-2,p),inverse_r=pow_mod(r,p-2,p);
        uint32_t common=uint64_t(basis_inverse)*pow_mod(4*n,p-2,p)%p;
        uint32_t factors[4];
        for(unsigned j=0;j<4;++j) factors[j]=uint64_t(common)*pow_mod(inverse_r,j,p)%p;
        uint32_t forward=1,inverse=1;
        for(unsigned index=0;index<n;++index) {
            s.forward[pi*n+index]=twiddle(forward,p);s.inverse[pi*n+index]=twiddle(inverse,p);
            for(unsigned j=0;j<4;++j) s.scale[(j*ModCount+pi)*n+index]=twiddle(uint64_t(factors[j])*inverse%p,p);
            forward=uint64_t(forward)*psi%p;inverse=uint64_t(inverse)*inverse_psi%p;
        }
        uint32_t power=1;
        for(unsigned limb=0;limb<PartWords;++limb) {
            s.input_powers[pi*PartWords+limb]=twiddle(power,p);power=uint64_t(power)*s.mods[pi].base%p;
        }
        if(n>=Tile) for(unsigned i=0;i<Tile;++i) {
            s.small_forward[pi*Tile+i]=s.forward[pi*n+i*(n/Tile)];
            s.small_inverse[pi*Tile+i]=s.inverse[pi*n+i*(n/Tile)];
        }
    }
    return s;
}
} // namespace quartic
