#pragma once

#include "mng_fixed_target_batch.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace mng {

struct FixedSourceDocumentView {
    uint64_t token_base = 0;
    uint32_t token_count = 0;
    uint64_t centroid_base = 0;
    uint32_t centroid_count = 0;
};

struct FixedSourceBatchScratchView {
    float* row_scores = nullptr;
    uint16_t* row_order = nullptr;
    uint16_t* top_beta_ids = nullptr;
};

inline void dot_fixed_source_rows_128_exact(const float* const* rows,
                                            uint32_t count,
                                            const float* fixed_row,
                                            float* out) {
    if (count == kFixedTargetBatchWidth) {
        dot_rows_to_fixed_target_128_exact_impl<kFixedTargetBatchWidth>(
            rows, fixed_row, out);
        return;
    }
    dot_rows_to_fixed_target_128_exact(rows, count, fixed_row, out);
}

inline void dot_fixed_source_rows_exact(const float* const* rows,
                                        uint32_t count,
                                        const float* fixed_row,
                                        uint32_t dim,
                                        float* out) {
    if (dim == 128) {
        dot_fixed_source_rows_128_exact(rows, count, fixed_row, out);
        return;
    }
    dot_rows_to_fixed_target_exact(rows, count, fixed_row, dim, out);
}

template <bool FixedDim128>
inline float directed_fixed_source_to_target_row_batch_exact_impl(
    FixedSourceDocumentView source,
    uint32_t dst,
    const float* token_data,
    const uint64_t* token_offset,
    const uint32_t* token_count,
    const float* centroid_data,
    const uint64_t* group_offsets,
    const uint16_t* token_order,
    uint32_t dim,
    uint32_t beta,
    FixedSourceBatchScratchView scratch) {
    const uint32_t row_dim = FixedDim128 ? 128u : dim;
    const uint64_t dst_token_base = token_offset[dst];
    const uint32_t dst_token_count = token_count[dst];
    float total = 0.0f;

    const float* target_rows[kFixedTargetBatchWidth];
    float dot_results[kFixedTargetBatchWidth];

    for (uint32_t c = 0; c < source.centroid_count; ++c) {
        const float* centroid =
            centroid_data + (size_t)(source.centroid_base + c) * row_dim;
        uint32_t j0 = 0;
        if constexpr (FixedDim128) {
            for (; j0 + kFixedTargetBatchWidth <= dst_token_count;
                 j0 += kFixedTargetBatchWidth) {
                for (uint32_t lane = 0; lane < kFixedTargetBatchWidth; ++lane) {
                    target_rows[lane] =
                        token_data +
                        (size_t)(dst_token_base + j0 + lane) * 128;
                }
                dot_rows_to_fixed_target_128_exact_impl<
                    kFixedTargetBatchWidth>(
                    target_rows, centroid, dot_results);
                for (uint32_t lane = 0; lane < kFixedTargetBatchWidth; ++lane) {
                    scratch.row_scores[j0 + lane] = dot_results[lane];
                    scratch.row_order[j0 + lane] =
                        static_cast<uint16_t>(j0 + lane);
                }
            }
            if (j0 < dst_token_count) {
                const uint32_t lanes = dst_token_count - j0;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    target_rows[lane] =
                        token_data +
                        (size_t)(dst_token_base + j0 + lane) * 128;
                }
                dot_rows_to_fixed_target_128_exact(
                    target_rows, lanes, centroid, dot_results);
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    scratch.row_scores[j0 + lane] = dot_results[lane];
                    scratch.row_order[j0 + lane] =
                        static_cast<uint16_t>(j0 + lane);
                }
            }
        } else {
            for (; j0 < dst_token_count; j0 += kFixedTargetBatchWidth) {
                const uint32_t lanes = std::min<uint32_t>(
                    kFixedTargetBatchWidth, dst_token_count - j0);
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    target_rows[lane] =
                        token_data +
                        (size_t)(dst_token_base + j0 + lane) * row_dim;
                }
                dot_fixed_source_rows_exact(
                    target_rows, lanes, centroid, row_dim, dot_results);
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    scratch.row_scores[j0 + lane] = dot_results[lane];
                    scratch.row_order[j0 + lane] =
                        static_cast<uint16_t>(j0 + lane);
                }
            }
        }

        auto cmp_row = [&](uint16_t a, uint16_t b) {
            const float score_a = scratch.row_scores[a];
            const float score_b = scratch.row_scores[b];
            if (score_a != score_b) return score_a > score_b;
            return a < b;
        };
        if (dst_token_count > beta) {
            std::nth_element(
                scratch.row_order,
                scratch.row_order + beta,
                scratch.row_order + dst_token_count,
                cmp_row);
        }
        uint16_t* top_ids =
            scratch.top_beta_ids + (size_t)c * beta;
        for (uint32_t j = 0; j < beta; ++j) {
            top_ids[j] = scratch.row_order[j];
        }

        const uint64_t group_begin =
            group_offsets[source.centroid_base + c];
        const uint64_t group_end =
            group_offsets[source.centroid_base + c + 1];
        for (uint64_t p = group_begin; p < group_end; ++p) {
            const uint16_t local_token = token_order[p];
            const float* source_token =
                token_data +
                (size_t)(source.token_base + local_token) * row_dim;
            float best = -std::numeric_limits<float>::infinity();
            j0 = 0;
            if constexpr (FixedDim128) {
                for (; j0 + kFixedTargetBatchWidth <= beta;
                     j0 += kFixedTargetBatchWidth) {
                    for (uint32_t lane = 0;
                         lane < kFixedTargetBatchWidth;
                         ++lane) {
                        target_rows[lane] =
                            token_data +
                            (size_t)(dst_token_base + top_ids[j0 + lane]) *
                                128;
                    }
                    dot_rows_to_fixed_target_128_exact_impl<
                        kFixedTargetBatchWidth>(
                        target_rows, source_token, dot_results);
                    for (uint32_t lane = 0;
                         lane < kFixedTargetBatchWidth;
                         ++lane) {
                        if (dot_results[lane] > best) {
                            best = dot_results[lane];
                        }
                    }
                }
                if (j0 < beta) {
                    const uint32_t lanes = beta - j0;
                    for (uint32_t lane = 0; lane < lanes; ++lane) {
                        target_rows[lane] =
                            token_data +
                            (size_t)(dst_token_base + top_ids[j0 + lane]) *
                                128;
                    }
                    dot_rows_to_fixed_target_128_exact(
                        target_rows, lanes, source_token, dot_results);
                    for (uint32_t lane = 0; lane < lanes; ++lane) {
                        if (dot_results[lane] > best) {
                            best = dot_results[lane];
                        }
                    }
                }
            } else {
                for (; j0 < beta; j0 += kFixedTargetBatchWidth) {
                    const uint32_t lanes = std::min<uint32_t>(
                        kFixedTargetBatchWidth, beta - j0);
                    for (uint32_t lane = 0; lane < lanes; ++lane) {
                        target_rows[lane] =
                            token_data +
                            (size_t)(dst_token_base + top_ids[j0 + lane]) *
                                row_dim;
                    }
                    dot_fixed_source_rows_exact(
                        target_rows, lanes, source_token, row_dim, dot_results);
                    for (uint32_t lane = 0; lane < lanes; ++lane) {
                        if (dot_results[lane] > best) {
                            best = dot_results[lane];
                        }
                    }
                }
            }
            total += best;
        }
    }
    return total / (float)source.token_count;
}

inline float directed_fixed_source_to_target_row_batch_128_exact(
    FixedSourceDocumentView source,
    uint32_t dst,
    const float* token_data,
    const uint64_t* token_offset,
    const uint32_t* token_count,
    const float* centroid_data,
    const uint64_t* group_offsets,
    const uint16_t* token_order,
    uint32_t beta,
    FixedSourceBatchScratchView scratch) {
    return directed_fixed_source_to_target_row_batch_exact_impl<true>(
        source, dst, token_data, token_offset, token_count, centroid_data,
        group_offsets, token_order, 128, beta, scratch);
}

inline float directed_fixed_source_to_target_row_batch_exact(
    FixedSourceDocumentView source,
    uint32_t dst,
    const float* token_data,
    const uint64_t* token_offset,
    const uint32_t* token_count,
    const float* centroid_data,
    const uint64_t* group_offsets,
    const uint16_t* token_order,
    uint32_t dim,
    uint32_t beta,
    FixedSourceBatchScratchView scratch) {
    if (dim == 128) {
        return directed_fixed_source_to_target_row_batch_128_exact(
            source, dst, token_data, token_offset, token_count, centroid_data,
            group_offsets, token_order, beta, scratch);
    }
    return directed_fixed_source_to_target_row_batch_exact_impl<false>(
        source, dst, token_data, token_offset, token_count, centroid_data,
        group_offsets, token_order, dim, beta, scratch);
}

}  // namespace mng
