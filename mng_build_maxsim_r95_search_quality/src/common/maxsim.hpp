#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
#include "fp16_utils.hpp"

namespace baseline {

// Dot product: fp32 query token × fp16 doc token
inline float dot_fp32_fp16(const float* q, const uint16_t* d, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += q[i] * fp16_to_fp32(d[i]);
    }
    return sum;
}

// Dot product: fp32 × fp32
inline float dot_fp32(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// MaxSim(Q, D) = sum_{q in Q} max_{d in D} dot(q, d)
// Q: fp32 (n_q_tokens × dim), D: fp16 (n_d_tokens × dim)
inline float maxsim_fp16(const float* Q, int n_q, const uint16_t* D, int n_d, int dim) {
    float total = 0.0f;
    for (int qi = 0; qi < n_q; ++qi) {
        const float* q = Q + qi * dim;
        float max_val = -std::numeric_limits<float>::infinity();
        for (int di = 0; di < n_d; ++di) {
            const uint16_t* d = D + di * dim;
            float score = dot_fp32_fp16(q, d, dim);
            if (score > max_val) max_val = score;
        }
        total += max_val;
    }
    return total;
}

// MaxSim(Q, D) with both fp32
inline float maxsim_fp32(const float* Q, int n_q, const float* D, int n_d, int dim) {
    float total = 0.0f;
    for (int qi = 0; qi < n_q; ++qi) {
        const float* q = Q + qi * dim;
        float max_val = -std::numeric_limits<float>::infinity();
        for (int di = 0; di < n_d; ++di) {
            const float* d = D + di * dim;
            float score = dot_fp32(q, d, dim);
            if (score > max_val) max_val = score;
        }
        total += max_val;
    }
    return total;
}

// max_{d in D} dot(q, d) — single query token vs all doc tokens (fp16 doc)
inline float max_dot_single_fp16(const float* q, const uint16_t* D, int n_d, int dim) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int di = 0; di < n_d; ++di) {
        float score = dot_fp32_fp16(q, D + di * dim, dim);
        if (score > max_val) max_val = score;
    }
    return max_val;
}

// max_{d in D} dot(q, d) — single query token vs all doc tokens (fp32 doc)
inline float max_dot_single_fp32(const float* q, const float* D, int n_d, int dim) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int di = 0; di < n_d; ++di) {
        float score = dot_fp32(q, D + di * dim, dim);
        if (score > max_val) max_val = score;
    }
    return max_val;
}

}  // namespace baseline
