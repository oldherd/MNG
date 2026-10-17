#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mng_exact_rerank {

struct RerankCountResolution {
    double requested_amp = 0.0;
    uint32_t requested_candidates = 0;
    uint32_t effective_candidates = 0;
    bool enabled = false;
};

inline RerankCountResolution resolve_rerank_candidates(
    double requested_amp,
    uint32_t topk,
    uint32_t ef_search,
    uint32_t candidates_found) {
    if (!std::isfinite(requested_amp) || requested_amp < 0.0) {
        throw std::invalid_argument("rerank_amp must be finite and non-negative");
    }
    if (requested_amp == 0.0) {
        return RerankCountResolution{requested_amp, 0, 0, false};
    }
    if (topk == 0) {
        throw std::invalid_argument("enabled exact rerank requires topk > 0");
    }
    if (requested_amp < 1.0) {
        throw std::invalid_argument("enabled exact rerank requires rerank_amp >= 1");
    }

    const long double requested =
        std::ceil(static_cast<long double>(requested_amp) * topk);
    if (requested > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("rerank_amp * topk exceeds uint32_t range");
    }

    const uint32_t requested_candidates = static_cast<uint32_t>(requested);
    const uint32_t effective_candidates =
        std::min({requested_candidates, ef_search, candidates_found});
    return RerankCountResolution{
        requested_amp, requested_candidates, effective_candidates, true};
}

}  // namespace mng_exact_rerank
