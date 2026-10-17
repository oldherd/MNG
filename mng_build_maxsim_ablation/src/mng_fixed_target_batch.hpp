#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(MNG_BACKEND_AVX512) || defined(MNG_BACKEND_AVX2)
#include <immintrin.h>
#endif

namespace mng {

#if defined(MNG_BACKEND_AVX512)
constexpr uint32_t kFixedTargetBatchWidth = 8;
#elif defined(MNG_BACKEND_AVX2)
constexpr uint32_t kFixedTargetBatchWidth = 4;
#else
// The scalar implementation exists for equivalence tests. Production enables
// the batch path only for a SIMD backend.
constexpr uint32_t kFixedTargetBatchWidth = 4;
#endif

template <uint32_t Width>
inline void dot_rows_to_fixed_target_128_exact_impl(const float* const* rows,
                                                    const float* target,
                                                    float* out) {
#if defined(MNG_BACKEND_AVX512)
    __m512 accumulators[Width];
    for (uint32_t lane = 0; lane < Width; ++lane) {
        accumulators[lane] = _mm512_setzero_ps();
    }
    for (uint32_t d = 0; d < 128; d += 16) {
        const __m512 target_values = _mm512_loadu_ps(target + d);
        for (uint32_t lane = 0; lane < Width; ++lane) {
            const __m512 source_values = _mm512_loadu_ps(rows[lane] + d);
            accumulators[lane] =
                _mm512_fmadd_ps(source_values, target_values, accumulators[lane]);
        }
    }
    for (uint32_t lane = 0; lane < Width; ++lane) {
        out[lane] = _mm512_reduce_add_ps(accumulators[lane]);
    }
#elif defined(MNG_BACKEND_AVX2)
    __m256 accumulators[Width];
    for (uint32_t lane = 0; lane < Width; ++lane) {
        accumulators[lane] = _mm256_setzero_ps();
    }
    for (uint32_t d = 0; d < 128; d += 8) {
        const __m256 target_values = _mm256_loadu_ps(target + d);
        for (uint32_t lane = 0; lane < Width; ++lane) {
            const __m256 source_values = _mm256_loadu_ps(rows[lane] + d);
            accumulators[lane] =
                _mm256_fmadd_ps(source_values, target_values, accumulators[lane]);
        }
    }
    for (uint32_t lane = 0; lane < Width; ++lane) {
        const __m128 lo = _mm256_castps256_ps128(accumulators[lane]);
        const __m128 hi = _mm256_extractf128_ps(accumulators[lane], 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        out[lane] = _mm_cvtss_f32(sum);
    }
#else
    for (uint32_t lane = 0; lane < Width; ++lane) {
        float score = 0.0f;
        for (uint32_t d = 0; d < 128; ++d) {
            score += rows[lane][d] * target[d];
        }
        out[lane] = score;
    }
#endif
}

template <uint32_t Width>
inline void dot_rows_to_fixed_target_exact_impl(const float* const* rows,
                                                const float* target,
                                                uint32_t dim,
                                                float* out) {
    if (dim == 128) {
        dot_rows_to_fixed_target_128_exact_impl<Width>(rows, target, out);
        return;
    }
    for (uint32_t lane = 0; lane < Width; ++lane) {
        float score = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            score += rows[lane][d] * target[d];
        }
        out[lane] = score;
    }
}

inline void dot_rows_to_fixed_target_128_exact(const float* const* rows,
                                               uint32_t count,
                                               const float* target,
                                               float* out) {
    switch (count) {
        case 1:
            dot_rows_to_fixed_target_128_exact_impl<1>(rows, target, out);
            return;
        case 2:
            dot_rows_to_fixed_target_128_exact_impl<2>(rows, target, out);
            return;
        case 3:
            dot_rows_to_fixed_target_128_exact_impl<3>(rows, target, out);
            return;
        case 4:
            dot_rows_to_fixed_target_128_exact_impl<4>(rows, target, out);
            return;
#if defined(MNG_BACKEND_AVX512)
        case 5:
            dot_rows_to_fixed_target_128_exact_impl<5>(rows, target, out);
            return;
        case 6:
            dot_rows_to_fixed_target_128_exact_impl<6>(rows, target, out);
            return;
        case 7:
            dot_rows_to_fixed_target_128_exact_impl<7>(rows, target, out);
            return;
        case 8:
            dot_rows_to_fixed_target_128_exact_impl<8>(rows, target, out);
            return;
#endif
        default:
            return;
    }
}

inline void dot_rows_to_fixed_target_exact(const float* const* rows,
                                           uint32_t count,
                                           const float* target,
                                           uint32_t dim,
                                           float* out) {
    if (dim == 128) {
        dot_rows_to_fixed_target_128_exact(rows, count, target, out);
        return;
    }
    switch (count) {
        case 1:
            dot_rows_to_fixed_target_exact_impl<1>(rows, target, dim, out);
            return;
        case 2:
            dot_rows_to_fixed_target_exact_impl<2>(rows, target, dim, out);
            return;
        case 3:
            dot_rows_to_fixed_target_exact_impl<3>(rows, target, dim, out);
            return;
        case 4:
            dot_rows_to_fixed_target_exact_impl<4>(rows, target, dim, out);
            return;
#if defined(MNG_BACKEND_AVX512)
        case 5:
            dot_rows_to_fixed_target_exact_impl<5>(rows, target, dim, out);
            return;
        case 6:
            dot_rows_to_fixed_target_exact_impl<6>(rows, target, dim, out);
            return;
        case 7:
            dot_rows_to_fixed_target_exact_impl<7>(rows, target, dim, out);
            return;
        case 8:
            dot_rows_to_fixed_target_exact_impl<8>(rows, target, dim, out);
            return;
#endif
        default:
            return;
    }
}

struct FixedTargetBatchScratchView {
    float* row_scores = nullptr;
    uint16_t* row_order = nullptr;
    uint16_t* top_beta_ids = nullptr;
    uint32_t row_stride = 0;
};

template <bool FixedDim128>
inline float directed_doc_to_fixed_target_batch_exact_impl(
    uint32_t src,
    const float* token_data,
    const uint64_t* token_offset,
    const uint32_t* token_count,
    const float* centroid_data,
    const uint64_t* centroid_offset,
    const uint16_t* centroid_count,
    const uint64_t* group_offsets,
    const uint16_t* token_order,
    uint32_t dim,
    const float* target_tokens,
    uint32_t target_len,
    uint32_t beta,
    FixedTargetBatchScratchView scratch) {
    const uint32_t row_dim = FixedDim128 ? 128u : dim;
    const uint64_t src_tbase = token_offset[src];
    const uint32_t src_len = token_count[src];
    const uint64_t src_cbase = centroid_offset[src];
    const uint32_t src_k = centroid_count[src];
    float total = 0.0f;

    const float* source_rows[kFixedTargetBatchWidth];
    float dot_results[kFixedTargetBatchWidth];
    float best_scores[kFixedTargetBatchWidth];

    for (uint32_t c0 = 0; c0 < src_k; c0 += kFixedTargetBatchWidth) {
        const uint32_t centroid_lanes =
            std::min<uint32_t>(kFixedTargetBatchWidth, src_k - c0);
        for (uint32_t lane = 0; lane < centroid_lanes; ++lane) {
            source_rows[lane] =
                centroid_data + (size_t)(src_cbase + c0 + lane) * row_dim;
        }

        if constexpr (FixedDim128) {
            if (centroid_lanes == kFixedTargetBatchWidth) {
                for (uint32_t j = 0; j < target_len; ++j) {
                    dot_rows_to_fixed_target_128_exact_impl<kFixedTargetBatchWidth>(
                        source_rows, target_tokens + (size_t)j * 128, dot_results);
                    for (uint32_t lane = 0; lane < kFixedTargetBatchWidth; ++lane) {
                        scratch.row_scores[(size_t)lane * scratch.row_stride + j] =
                            dot_results[lane];
                    }
                }
            } else {
                for (uint32_t j = 0; j < target_len; ++j) {
                    dot_rows_to_fixed_target_128_exact(
                        source_rows, centroid_lanes,
                        target_tokens + (size_t)j * 128, dot_results);
                    for (uint32_t lane = 0; lane < centroid_lanes; ++lane) {
                        scratch.row_scores[(size_t)lane * scratch.row_stride + j] =
                            dot_results[lane];
                    }
                }
            }
        } else {
            for (uint32_t j = 0; j < target_len; ++j) {
                dot_rows_to_fixed_target_exact(
                    source_rows, centroid_lanes,
                    target_tokens + (size_t)j * row_dim, row_dim, dot_results);
                for (uint32_t lane = 0; lane < centroid_lanes; ++lane) {
                    scratch.row_scores[(size_t)lane * scratch.row_stride + j] =
                        dot_results[lane];
                }
            }
        }

        for (uint32_t lane = 0; lane < centroid_lanes; ++lane) {
            float* row_scores =
                scratch.row_scores + (size_t)lane * scratch.row_stride;
            uint16_t* row_order =
                scratch.row_order + (size_t)lane * scratch.row_stride;
            for (uint32_t j = 0; j < target_len; ++j) {
                row_order[j] = (uint16_t)j;
            }
            auto cmp_row = [&](uint16_t a, uint16_t b) {
                const float sa = row_scores[a];
                const float sb = row_scores[b];
                if (sa != sb) return sa > sb;
                return a < b;
            };
            if (target_len > beta) {
                std::nth_element(row_order, row_order + beta,
                                 row_order + target_len, cmp_row);
            }
            uint16_t* top_ids =
                scratch.top_beta_ids + (size_t)(c0 + lane) * beta;
            for (uint32_t j = 0; j < beta; ++j) {
                top_ids[j] = row_order[j];
            }
        }

        for (uint32_t lane = 0; lane < centroid_lanes; ++lane) {
            const uint64_t centroid_id = src_cbase + c0 + lane;
            const uint64_t group_begin = group_offsets[centroid_id];
            const uint64_t group_end = group_offsets[centroid_id + 1];
            const uint16_t* top_ids =
                scratch.top_beta_ids + (size_t)(c0 + lane) * beta;

            for (uint64_t p0 = group_begin; p0 < group_end;
                 p0 += kFixedTargetBatchWidth) {
                const uint32_t token_lanes = (uint32_t)std::min<uint64_t>(
                    kFixedTargetBatchWidth, group_end - p0);
                for (uint32_t token_lane = 0; token_lane < token_lanes;
                     ++token_lane) {
                    const uint16_t local_token = token_order[p0 + token_lane];
                    source_rows[token_lane] =
                        token_data + (size_t)(src_tbase + local_token) * row_dim;
                    best_scores[token_lane] =
                        -std::numeric_limits<float>::infinity();
                }
                if constexpr (FixedDim128) {
                    if (token_lanes == kFixedTargetBatchWidth) {
                        for (uint32_t j = 0; j < beta; ++j) {
                            const float* target =
                                target_tokens + (size_t)top_ids[j] * 128;
                            dot_rows_to_fixed_target_128_exact_impl<
                                kFixedTargetBatchWidth>(
                                source_rows, target, dot_results);
                            for (uint32_t token_lane = 0;
                                 token_lane < kFixedTargetBatchWidth;
                                 ++token_lane) {
                                if (dot_results[token_lane] >
                                    best_scores[token_lane]) {
                                    best_scores[token_lane] =
                                        dot_results[token_lane];
                                }
                            }
                        }
                    } else {
                        for (uint32_t j = 0; j < beta; ++j) {
                            const float* target =
                                target_tokens + (size_t)top_ids[j] * 128;
                            dot_rows_to_fixed_target_128_exact(
                                source_rows, token_lanes, target, dot_results);
                            for (uint32_t token_lane = 0;
                                 token_lane < token_lanes;
                                 ++token_lane) {
                                if (dot_results[token_lane] >
                                    best_scores[token_lane]) {
                                    best_scores[token_lane] =
                                        dot_results[token_lane];
                                }
                            }
                        }
                    }
                } else {
                    for (uint32_t j = 0; j < beta; ++j) {
                        const float* target =
                            target_tokens + (size_t)top_ids[j] * row_dim;
                        dot_rows_to_fixed_target_exact(
                            source_rows, token_lanes, target, row_dim,
                            dot_results);
                        for (uint32_t token_lane = 0;
                             token_lane < token_lanes;
                             ++token_lane) {
                            if (dot_results[token_lane] >
                                best_scores[token_lane]) {
                                best_scores[token_lane] =
                                    dot_results[token_lane];
                            }
                        }
                    }
                }
                for (uint32_t token_lane = 0; token_lane < token_lanes;
                     ++token_lane) {
                    total += best_scores[token_lane];
                }
            }
        }
    }
    return total / (float)src_len;
}

inline float directed_doc_to_fixed_target_batch_128_exact(
    uint32_t src,
    const float* token_data,
    const uint64_t* token_offset,
    const uint32_t* token_count,
    const float* centroid_data,
    const uint64_t* centroid_offset,
    const uint16_t* centroid_count,
    const uint64_t* group_offsets,
    const uint16_t* token_order,
    const float* target_tokens,
    uint32_t target_len,
    uint32_t beta,
    FixedTargetBatchScratchView scratch) {
    return directed_doc_to_fixed_target_batch_exact_impl<true>(
        src, token_data, token_offset, token_count, centroid_data,
        centroid_offset, centroid_count, group_offsets, token_order, 128,
        target_tokens, target_len, beta, scratch);
}

inline float directed_doc_to_fixed_target_batch_exact(
    uint32_t src,
    const float* token_data,
    const uint64_t* token_offset,
    const uint32_t* token_count,
    const float* centroid_data,
    const uint64_t* centroid_offset,
    const uint16_t* centroid_count,
    const uint64_t* group_offsets,
    const uint16_t* token_order,
    uint32_t dim,
    const float* target_tokens,
    uint32_t target_len,
    uint32_t beta,
    FixedTargetBatchScratchView scratch) {
    if (dim == 128) {
        return directed_doc_to_fixed_target_batch_128_exact(
            src, token_data, token_offset, token_count, centroid_data,
            centroid_offset, centroid_count, group_offsets, token_order,
            target_tokens, target_len, beta, scratch);
    }
    return directed_doc_to_fixed_target_batch_exact_impl<false>(
        src, token_data, token_offset, token_count, centroid_data,
        centroid_offset, centroid_count, group_offsets, token_order, dim,
        target_tokens, target_len, beta, scratch);
}

}  // namespace mng
