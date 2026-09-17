#include "quartic/host_setup.h"
#include <gmpxx.h>
#include <random>
#include <iostream>
mpz_class integer(const uint32_t* words,size_t size) {
    mpz_class value;mpz_import(value.get_mpz_t(),size,-1,4,0,0,words);return value;
}
uint32_t add(uint32_t a,uint32_t b,uint32_t p) {uint32_t x=a+b;return x>=p?x-p:x;}
uint32_t sub(uint32_t a,uint32_t b,uint32_t p) {return a<b?a+p-b:a-b;}
int main() {
    std::mt19937 random(44179);unsigned checked=0;
    for(uint32_t delta:{1u,11993087u,0x0fffffffu}) {
        constexpr unsigned n=32768;auto setup=quartic::setup(n,delta);
        auto P=integer(setup.product.data(),setup.product.size());
        auto bound=integer(setup.bound.data(),setup.bound.size());
        mpz_class B=mpz_class(1)<<217,q=(mpz_class(1)<<868)-delta;
        mpz_class expected_bound=mpz_class(n)*(B-1)*(B-1)*(1+3*delta);
        if(bound!=expected_bound || P<=4*bound || P*16>=(mpz_class(1)<<512)) return 1;
        for(unsigned trial=0;trial<128;++trial) {
            mpz_class values[4],sums[4];__uint128_t fractions[4]{};
            unsigned index=(trial*7919)%n;
            for(unsigned j=0;j<4;++j) {
                uint32_t words[16];for(auto& word:words) word=random();
                values[j]=integer(words,16)%bound;
                if(trial==0) values[j]=0;
                if(trial==1) values[j]=bound-1;
                if(trial==2) values[j]=1;
                if((trial+j)&1) values[j]=-values[j];
            }
            for(unsigned pi=0;pi<quartic::ModCount;++pi) {
                auto mod=setup.mods[pi];uint32_t p=mod.p,raw[4]{};
                if(mpz_probab_prime_p(mpz_class(p).get_mpz_t(),30)==0) return 2;
                auto basis=integer(setup.bases.data()+pi*16,16);
                if(basis*p!=P) return 3;
                for(unsigned k=0;k<4;++k) {
                    uint32_t root=uint64_t(setup.roots[pi].powers[1].value)*quartic::pow_mod(setup.roots[pi].i.value,k,p)%p;
                    uint32_t power=1;
                    for(unsigned j=0;j<4;++j) {
                        uint32_t term=uint64_t(mpz_fdiv_ui(values[j].get_mpz_t(),p))*power%p;
                        raw[k]=add(raw[k],term,p);power=uint64_t(power)*root%p;
                    }
                    raw[k]=uint64_t(raw[k])*n%p;
                    raw[k]=uint64_t(raw[k])*setup.forward[pi*n+index].value%p;
                }
                uint32_t u=add(raw[0],raw[2],p),v=sub(raw[0],raw[2],p);
                uint32_t s=add(raw[1],raw[3],p),t=uint64_t(sub(raw[1],raw[3],p))*setup.roots[pi].i.value%p;
                uint32_t mixed[4]={add(u,s,p),sub(v,t,p),sub(u,s,p),add(v,t,p)};
                for(unsigned j=0;j<4;++j) {
                    uint32_t weighted=uint64_t(mixed[j])*setup.scale[(j*quartic::ModCount+pi)*n+index].value%p;
                    sums[j]+=basis*weighted;fractions[j]+=__uint128_t(weighted)*mod.reciprocal;
                }
            }
            mpz_class assembled=0,reference=0;
            for(unsigned j=0;j<4;++j) {
                uint32_t nearest=uint32_t(fractions[j]>>64)+uint32_t(uint64_t(fractions[j])>>63);
                mpz_class recovered=sums[j]-nearest*P;
                if(recovered!=values[j]) {std::cerr<<"CRT mismatch "<<delta<<" "<<trial<<" "<<j<<"\n";return 4;}
                mpz_class magnitude=abs(recovered),shifted=magnitude<<(217*j);
                mpz_class folded=(shifted&((mpz_class(1)<<868)-1))+(shifted>>868)*delta;
                if(folded>=q) folded-=q;
                if(folded<0 || folded>=q) return 5;
                if(recovered<0 && folded!=0) folded=q-folded;
                assembled+=folded;reference+=values[j]<<(217*j);++checked;
            }
            mpz_mod(assembled.get_mpz_t(),assembled.get_mpz_t(),q.get_mpz_t());
            mpz_mod(reference.get_mpz_t(),reference.get_mpz_t(),q.get_mpz_t());
            if(assembled!=reference) return 6;
        }
        std::cout<<"delta="<<delta<<" P_bits="<<mpz_sizeinbase(P.get_mpz_t(),2)<<" bound_bits="<<mpz_sizeinbase(bound.get_mpz_t(),2)<<" PASS\n";
    }
    std::cout<<checked<<" signed quartic CRT and recomposition cases passed\n";
}
