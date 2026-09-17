#pragma once
#include "rns/host_setup.h"
#include <array>
namespace rns_tensor {
constexpr unsigned RealK=rns::AbiWords*4,K=128,Columns=240,SumColumn=rns::PrimeCount*4;
struct Weights {
    std::array<int8_t,K*Columns> matrix{};
    std::array<int32_t,Columns> correction{};
};
inline Weights make_weights(const std::vector<rns::SmallMod>& mods) {
    Weights result;
    for(unsigned pi=0;pi<rns::PrimeCount;++pi) {
        uint32_t power=1,p=mods[pi].p;
        for(unsigned row=0;row<RealK;++row) {
            for(unsigned byte=0;byte<4;++byte) {
                unsigned col=4*pi+byte;
                int weight=int((power>>(byte*8))&255u)-128;
                result.matrix[col*K+row]=int8_t(weight);
                result.correction[col]+=128*weight;
            }
            power=uint64_t(power)*256%p;
        }
        for(unsigned byte=0;byte<4;++byte) result.correction[4*pi+byte]+=RealK*128*128;
    }
    for(unsigned row=0;row<RealK;++row) result.matrix[SumColumn*K+row]=1;
    return result;
}
} // namespace rns_tensor
