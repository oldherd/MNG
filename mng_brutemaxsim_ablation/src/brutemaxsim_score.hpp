#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mng_brutemaxsim_ablation {

inline void validate_maxsim_inputs(const float* source_tokens,
                                   uint32_t source_len,
                                   const float* target_tokens,
                                   uint32_t target_len,
                                   uint32_t dim) {
    if (source_tokens == nullptr || target_tokens == nullptr) {
        throw std::invalid_argument("brute MaxSim received a null token pointer");
    }
    if (source_len == 0 || target_len == 0 || dim == 0) {
        throw std::invalid_argument("brute MaxSim requires non-empty token matrices");
    }
}

template <typename DotFn, typename EmitFn>
inline void for_each_source_maxsim(const float* source_tokens,
                                   uint32_t source_len,
                                   const float* target_tokens,
                                   uint32_t target_len,
                                   uint32_t dim,
                                   DotFn&& dot,
                                   EmitFn&& emit) {
    validate_maxsim_inputs(
        source_tokens, source_len, target_tokens, target_len, dim);

    for (uint32_t i = 0; i < source_len; ++i) {
        const float* source = source_tokens + static_cast<size_t>(i) * dim;
        float best = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < target_len; ++j) {
            const float* target = target_tokens + static_cast<size_t>(j) * dim;
            const float score = dot(source, target, dim);
            if (score > best) best = score;
        }
        emit(i, best);
    }
}

template <typename DotFn>
inline float directed_mean_maxsim(const float* source_tokens,
                                  uint32_t source_len,
                                  const float* target_tokens,
                                  uint32_t target_len,
                                  uint32_t dim,
                                  DotFn&& dot) {
    float total = 0.0f;
    for_each_source_maxsim(
        source_tokens,
        source_len,
        target_tokens,
        target_len,
        dim,
        std::forward<DotFn>(dot),
        [&](uint32_t, float best) {
            total += best;
        });
    return total / static_cast<float>(source_len);
}

}  // namespace mng_brutemaxsim_ablation
