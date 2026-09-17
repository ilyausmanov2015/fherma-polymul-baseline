#include "rns/host_setup.h"
#include <gmpxx.h>
#include <cstdio>
mpz_class integer(const rns::Words& words) {
    mpz_class x;mpz_import(x.get_mpz_t(),words.size(),-1,4,0,0,words.data());return x;
}
int main() {
    rns::Words q(28,0xffffffffu);q[0]=uint32_t(0)-11993087;q[27]=15;
    auto s=rns::setup(32768,q);
    auto Q=integer(q),P=integer(s.product);
    if(!(P>4*32768*(Q-1)*(Q-1)) || integer(s.product_mod_q)!=P%Q || integer(s.half_ceil)!=(P+1)/2) return 1;
    for(unsigned i=0;i<rns::PrimeCount;++i) {
        auto p=s.mods[i].p;
        if(!mpz_probab_prime_p(mpz_class(p).get_mpz_t(),30)) return 2;
        rns::Words basis(s.bases.begin()+i*56,s.bases.begin()+(i+1)*56);
        if(integer(basis)*p!=P) return 3;
        rns::Words reduced(s.bases_mod_q.begin()+i*rns::AccumWords,s.bases_mod_q.begin()+(i+1)*rns::AccumWords);
        if(integer(reduced)!=integer(basis)%Q) return 6;
        mpz_class B=mpz_class(1)<<64,recip=B/p;
        if(recip.get_ui()!=s.mods[i].reciprocal) return 4;
        for(unsigned j:{0u,1u,127u,16384u,32767u}) {
            auto f=s.forward[size_t(i)*32768+j],inv=s.inverse[size_t(i)*32768+j];
            if(uint64_t(f.value)*inv.value%p!=1 || (uint64_t(f.value)<<32)/p!=f.shoup) return 5;
        }
    }
    std::printf("RNS setup: 57 primes, %zu-bit product, roots and CRT constants verified\n",mpz_sizeinbase(P.get_mpz_t(),2));
}
