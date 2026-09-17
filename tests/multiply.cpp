// Validate actual device Karatsuba and modular-fold helpers against GMP.
#include "build-emulation/solve_emulated.cpp"
#include <cstdio>
int main() {
    gmp_randclass random(gmp_randinit_mt);random.seed(982177);
    unsigned cases=0;
    for(uint32_t c:{1u,11993087u,0x0fffffffu}) {
        mpz_class q=(mpz_class(1)<<868)-c,B=mpz_class(1)<<448,H=mpz_class(1)<<420;
        Big modulus(q);
        std::vector<mpz_class> edges={0,1,2,mpz_class(B-1),B,mpz_class(B+1),mpz_class(H-1),H,
            mpz_class(q/2),mpz_class(q-1),mpz_class(q-2),mpz_class((H-1)*B),mpz_class((H-1)*B+1)};
        auto test=[&](const mpz_class& a,const mpz_class& b) {
            mpz_class exact=a*b;
            auto wide=karatsuba_product(Big(a),Big(b));
            mpz_class got=wide.lo.value()+(wide.hi.value()<<896);
            if(got!=exact || multiply(Big(a),Big(b),modulus).value()!=exact%q) {
                std::fprintf(stderr,"wide multiplication mismatch c=%u case=%u\n",c,cases);std::exit(1);
            }
            ++cases;
        };
        for(const auto& a:edges) for(const auto& b:edges) test(a,b);
        for(unsigned i=0;i<20000;++i) test(random.get_z_range(q),random.get_z_range(q));
    }
    std::printf("Karatsuba full-product and modular-fold oracle checks: %u passed\n",cases);
}
