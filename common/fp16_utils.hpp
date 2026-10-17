#pragma once
#include <cstdint>
#include <cstring>

namespace baseline {

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

inline uint16_t fp32_to_fp16(float val) {
    uint32_t f;
    std::memcpy(&f, &val, 4);

    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        int shift = 14 - exp;
        uint16_t h = (uint16_t)(sign | (mant >> shift));
        return h;
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

}  // namespace baseline
