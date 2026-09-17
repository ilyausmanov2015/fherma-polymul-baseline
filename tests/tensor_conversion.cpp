#include "rns/tensor_setup.h"
#include <gmpxx.h>
#include <random>
#include <iostream>
int main() {
    std::vector<uint32_t> q(rns::AbiWords,0xffffffffu);q[0]=uint32_t(0)-11993087;q.back()=15;
    auto setup=rns::setup(2,q);auto weights=rns_tensor::make_weights(setup.mods);
    std::mt19937 random(619047);
    unsigned checked=0;
    for(unsigned trial=0;trial<1024;++trial) {
        std::array<uint8_t,rns_tensor::RealK> bytes{};
        for(auto& byte:bytes) byte=trial<4 ? uint8_t(trial==0?0:trial==1?255:trial==2?128:127) : uint8_t(random());
        mpz_class integer;mpz_import(integer.get_mpz_t(),bytes.size(),-1,1,0,0,bytes.data());
        int32_t centered_sum=0;for(auto byte:bytes) centered_sum+=int(byte)-128;
        for(unsigned pi=0;pi<rns::PrimeCount;++pi) {
            uint64_t exact_sum=0;
            for(unsigned byte=0;byte<4;++byte) {
                unsigned col=4*pi+byte;int32_t dot=0;
                for(unsigned row=0;row<rns_tensor::RealK;++row)
                    dot+=(int(bytes[row])-128)*int(weights.matrix[col*rns_tensor::K+row]);
                int32_t recovered=dot+128*centered_sum+weights.correction[col];
                if(recovered<0 || recovered>int32_t(rns_tensor::RealK*255*255)) return 1;
                exact_sum+=uint64_t(recovered)<<(byte*8);
            }
            auto modulus=setup.mods[pi];
            uint64_t quotient=uint64_t((__uint128_t(exact_sum)*modulus.reciprocal)>>64);
            uint32_t residual=uint32_t(exact_sum-quotient*modulus.p);
            if(residual>=modulus.p) residual-=modulus.p;
            if(residual!=mpz_fdiv_ui(integer.get_mpz_t(),modulus.p)) return 2;
            ++checked;
        }
    }
    std::cout<<checked<<" exact byte-dot residue conversions passed\n";
}
