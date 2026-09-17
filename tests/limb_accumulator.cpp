#include "quartic/limb_accumulator.h"
#include <gmpxx.h>
#include <array>
#include <random>
#include <cassert>
#include <cstdio>

int main() {
    std::mt19937 random(184749);
    const mpz_class mask=(mpz_class(1)<<512)-1;
    for(unsigned trial=0;trial<1000;++trial) {
        quartic::LimbAccumulator<16> accumulator;
        mpz_class exact=0;
        for(unsigned term=0;term<16;++term) {
            std::array<uint32_t,16> basis;
            for(auto& word:basis) word=trial<3 ? (trial==0 ? 0 : UINT32_MAX) : random();
            uint32_t factor=trial<3 ? (trial==2 ? 1 : UINT32_MAX) : random();
            mpz_class value=0;
            for(unsigned word=16;word--;) { value<<=32;value+=basis[word]; }
            exact+=value*factor;
            accumulator.add(basis.data(),factor);
        }
        std::array<uint32_t,16> output;
        uint64_t overflow=accumulator.store(output);
        mpz_class actual=0;
        for(unsigned word=16;word--;) {actual<<=32;actual+=output[word];}
        assert(actual==(exact&mask));
        mpz_class high=exact>>512;
        assert(overflow==high.get_ui());
    }
    std::puts("1000 deferred-carry dot products and overflow values match GMP");
}
