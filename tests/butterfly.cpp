// Test the actual device helper bodies through the GMP adapter against
// independent GMP expressions, with values close to every reduction boundary.
#include "build-emulation/solve_emulated.cpp"
#include <cstdio>
int main() {
    gmp_randclass random(gmp_randinit_mt); random.seed(73119);
    unsigned cases=0;
    for(uint32_t c:{1u,11993087u,0x0fffffffu}) {
        mpz_class B=mpz_class(1)<<868,q=B-c;
        Big modulus(q);
        std::vector<mpz_class> edges={0,1,2,c-1,c,c+1,mpz_class(q/2),mpz_class(q/2+1),mpz_class(q-c),mpz_class(q-2),mpz_class(q-1)};
        auto test=[&](const mpz_class& a,const mpz_class& b) {
            auto add=butterfly_add(Big(a),Big(b),modulus).value();
            auto sub=butterfly_sub(Big(a),Big(b),modulus).value();
            mpz_class want_add=(a+b)%q,want_sub=(a-b+q)%q;
            ++cases;
            if(add!=want_add || sub!=want_sub) { std::fprintf(stderr,"butterfly mismatch c=%u case=%u\n",c,cases); std::exit(1); }
        };
        for(const auto& a:edges) for(const auto& b:edges) test(a,b);
        for(unsigned i=0;i<20000;++i) {
            mpz_class a=random.get_z_range(q),b=random.get_z_range(q);
            test(a,b);
            // Force sums immediately below, at, and above q when canonical.
            for(int offset=-1;offset<=1;++offset) {
                b=q-a+offset;
                if(b>=0 && b<q) test(a,b);
            }
        }
    }
    std::printf("Butterfly boundary/oracle checks: %u passed\n",cases);
}
