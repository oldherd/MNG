#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#elif defined(MNG_EXACT_RERANK_USE_MKL)
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

#ifndef MNG_EXACT_RERANK_VALIDATE
#define MNG_EXACT_RERANK_VALIDATE 0
#endif

#ifndef MNG_EXACT_RERANK_PRETRANSPOSE_Q32
#define MNG_EXACT_RERANK_PRETRANSPOSE_Q32 1
#endif

#ifndef MNG_EXACT_RERANK_DIAGNOSTICS
#define MNG_EXACT_RERANK_DIAGNOSTICS 1
#endif

#ifndef MNG_EXACT_RERANK_TRUST_INTERNAL_CANDIDATES
#define MNG_EXACT_RERANK_TRUST_INTERNAL_CANDIDATES 0
#endif

namespace mng_exact_rerank {

template <typename T, size_t Alignment = 64>
class AlignedBuffer {
    static_assert(std::is_trivially_copyable<T>::value,
                  "AlignedBuffer is only for trivially copyable hot data");

public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept {
        swap(other);
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    ~AlignedBuffer() {
        reset();
    }

    void reserve_uninitialized(size_t count) {
        if (count <= capacity_) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::overflow_error("aligned buffer byte size overflow");
        }

        void* next = nullptr;
        if (posix_memalign(&next, Alignment, count * sizeof(T)) != 0 || next == nullptr) {
            throw std::bad_alloc();
        }
        std::free(data_);
        data_ = static_cast<T*>(next);
        capacity_ = count;
    }

    void reset() noexcept {
        std::free(data_);
        data_ = nullptr;
        capacity_ = 0;
    }

    T* data() noexcept {
        return data_;
    }

    const T* data() const noexcept {
        return data_;
    }

    size_t capacity() const noexcept {
        return capacity_;
    }

private:
    void swap(AlignedBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(capacity_, other.capacity_);
    }

    T* data_ = nullptr;
    size_t capacity_ = 0;
};

struct TokenStoreView {
    const float* tokens = nullptr;
    const uint64_t* token_offsets = nullptr;
    const uint32_t* token_counts = nullptr;
    uint32_t num_docs = 0;
    uint32_t dim = 0;
};

struct RerankStats {
    uint32_t candidates_scored = 0;
    uint32_t results_written = 0;
    uint32_t gemm_calls = 0;
    uint64_t packed_tokens = 0;
    uint64_t query_transpose_elements = 0;
};

struct ExactRerankScratch {
    uint32_t max_candidates = 0;
    uint32_t max_query_tokens = 0;
    uint32_t dim = 0;
    uint32_t block_token_capacity = 0;

    AlignedBuffer<float> packed_tokens;
    AlignedBuffer<float> score_matrix;
    AlignedBuffer<float> exact_scores;
    AlignedBuffer<float> query_maxima;
    AlignedBuffer<float> query_transposed;
    AlignedBuffer<uint32_t> candidate_order;
    AlignedBuffer<uint32_t> pack_order;
    AlignedBuffer<uint32_t> segment_offsets;
    AlignedBuffer<uint32_t> segment_slots;

    void prepare(uint32_t candidates,
                 uint32_t query_tokens,
                 uint32_t vector_dim,
                 uint32_t requested_block_tokens,
                 uint32_t max_document_tokens) {
        if (candidates == 0 || query_tokens == 0 || vector_dim == 0) {
            throw std::invalid_argument("rerank scratch dimensions must be non-zero");
        }
        max_candidates = candidates;
        max_query_tokens = query_tokens;
        dim = vector_dim;
        block_token_capacity =
            std::max<uint32_t>(requested_block_tokens, max_document_tokens);
        if (block_token_capacity == 0) {
            throw std::invalid_argument("rerank block token capacity must be non-zero");
        }

        packed_tokens.reserve_uninitialized(
            checked_product(block_token_capacity, dim));
        score_matrix.reserve_uninitialized(
            checked_product(max_query_tokens, block_token_capacity));
        exact_scores.reserve_uninitialized(max_candidates);
        query_maxima.reserve_uninitialized(max_query_tokens);
        query_transposed.reserve_uninitialized(
            checked_product(dim, max_query_tokens));
        candidate_order.reserve_uninitialized(max_candidates);
        pack_order.reserve_uninitialized(max_candidates);
        segment_offsets.reserve_uninitialized(static_cast<size_t>(max_candidates) + 1);
        segment_slots.reserve_uninitialized(max_candidates);
    }

private:
    static size_t checked_product(size_t a, size_t b) {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
            throw std::overflow_error("rerank scratch element count overflow");
        }
        return a * b;
    }
};

inline void validate_common_inputs(const float* query_tokens,
                                   uint32_t query_len,
                                   const uint32_t* candidate_ids,
                                   uint32_t candidate_count,
                                   const uint32_t* out_ids,
                                   uint32_t topk,
                                   const TokenStoreView& docs,
                                   const ExactRerankScratch& scratch) {
    if (query_tokens == nullptr || candidate_ids == nullptr || docs.tokens == nullptr ||
        docs.token_offsets == nullptr || docs.token_counts == nullptr) {
        throw std::invalid_argument("exact rerank received a null input pointer");
    }
    if (topk != 0 && out_ids == nullptr) {
        throw std::invalid_argument("exact rerank received a null output pointer");
    }
    if (query_len == 0 || docs.dim == 0) {
        throw std::invalid_argument("exact rerank requires non-empty token matrices");
    }
    if (candidate_count > scratch.max_candidates ||
        query_len > scratch.max_query_tokens || docs.dim != scratch.dim) {
        throw std::invalid_argument("exact rerank scratch capacity mismatch");
    }
}

inline float exact_mean_maxsim_reference(const float* query_tokens,
                                         uint32_t query_len,
                                         const float* document_tokens,
                                         uint32_t document_len,
                                         uint32_t dim) {
    if (query_tokens == nullptr || document_tokens == nullptr ||
        query_len == 0 || document_len == 0 || dim == 0) {
        throw std::invalid_argument("reference MaxSim requires non-empty matrices");
    }

    float total = 0.0f;
    for (uint32_t qi = 0; qi < query_len; ++qi) {
        const float* query = query_tokens + static_cast<size_t>(qi) * dim;
        float best = -std::numeric_limits<float>::infinity();
        for (uint32_t di = 0; di < document_len; ++di) {
            const float* document =
                document_tokens + static_cast<size_t>(di) * dim;
            float dot = 0.0f;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : dot)
#endif
            for (uint32_t d = 0; d < dim; ++d) {
                dot += query[d] * document[d];
            }
            if (dot > best) best = dot;
        }
        total += best;
    }
    return total / static_cast<float>(query_len);
}

inline void transpose_query_once(const float* query_tokens,
                                 uint32_t query_len,
                                 uint32_t dim,
                                 float* query_transposed) {
    if (query_tokens == nullptr || query_transposed == nullptr ||
        query_len == 0 || dim == 0) {
        throw std::invalid_argument("query transpose requires non-empty matrices");
    }

    constexpr uint32_t tile = 16;
    for (uint32_t dim_begin = 0; dim_begin < dim; dim_begin += tile) {
        const uint32_t dim_end = std::min<uint32_t>(dim, dim_begin + tile);
        for (uint32_t query_begin = 0; query_begin < query_len;
             query_begin += tile) {
            const uint32_t query_end =
                std::min<uint32_t>(query_len, query_begin + tile);
            for (uint32_t d = dim_begin; d < dim_end; ++d) {
                float* destination =
                    query_transposed + static_cast<size_t>(d) * query_len;
                for (uint32_t qi = query_begin; qi < query_end; ++qi) {
                    destination[qi] =
                        query_tokens[static_cast<size_t>(qi) * dim + d];
                }
            }
        }
    }
}

inline float reduce_token_major_segment_generic(const float* score_matrix,
                                                uint32_t query_len,
                                                uint32_t begin_token,
                                                uint32_t end_token,
                                                float* query_maxima) {
    std::fill(query_maxima,
              query_maxima + query_len,
              -std::numeric_limits<float>::infinity());
    for (uint32_t token = begin_token; token < end_token; ++token) {
        const float* row =
            score_matrix + static_cast<size_t>(token) * query_len;
#if defined(_OPENMP)
#pragma omp simd
#endif
        for (uint32_t qi = 0; qi < query_len; ++qi) {
            if (row[qi] > query_maxima[qi]) query_maxima[qi] = row[qi];
        }
    }

    float total = 0.0f;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : total)
#endif
    for (uint32_t qi = 0; qi < query_len; ++qi) {
        total += query_maxima[qi];
    }
    return total / static_cast<float>(query_len);
}

#if defined(__AVX512F__)
inline float reduce_token_major_segment_q32_avx512(const float* score_matrix,
                                                   uint32_t begin_token,
                                                   uint32_t end_token) noexcept {
    __m512 best_lo = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    __m512 best_hi = best_lo;
    for (uint32_t token = begin_token; token < end_token; ++token) {
        const float* row = score_matrix + static_cast<size_t>(token) * 32;
        best_lo = _mm512_max_ps(best_lo, _mm512_loadu_ps(row));
        best_hi = _mm512_max_ps(best_hi, _mm512_loadu_ps(row + 16));
    }
    return (_mm512_reduce_add_ps(best_lo) + _mm512_reduce_add_ps(best_hi)) /
           32.0f;
}
#endif

inline uint32_t select_topk(const uint32_t* candidate_ids,
                            uint32_t candidate_count,
                            uint32_t topk,
                            ExactRerankScratch& scratch,
                            uint32_t* out_ids) {
    const uint32_t keep = std::min<uint32_t>(candidate_count, topk);
    uint32_t* order = scratch.candidate_order.data();
    const float* scores = scratch.exact_scores.data();
    for (uint32_t i = 0; i < candidate_count; ++i) order[i] = i;

    auto better = [&](uint32_t lhs, uint32_t rhs) {
        if (scores[lhs] != scores[rhs]) return scores[lhs] > scores[rhs];
        return candidate_ids[lhs] < candidate_ids[rhs];
    };
    if (keep < candidate_count) {
        std::nth_element(order, order + keep, order + candidate_count, better);
    }
    std::sort(order, order + keep, better);
    for (uint32_t i = 0; i < keep; ++i) {
        out_ids[i] = candidate_ids[order[i]];
    }
    return keep;
}

inline RerankStats rerank_reference(const float* query_tokens,
                                    uint32_t query_len,
                                    const uint32_t* candidate_ids,
                                    uint32_t candidate_count,
                                    uint32_t topk,
                                    const TokenStoreView& docs,
                                    ExactRerankScratch& scratch,
                                    uint32_t* out_ids) {
    validate_common_inputs(
        query_tokens, query_len, candidate_ids, candidate_count, out_ids, topk,
        docs, scratch);

    for (uint32_t slot = 0; slot < candidate_count; ++slot) {
        const uint32_t doc = candidate_ids[slot];
        if (doc >= docs.num_docs) {
            throw std::out_of_range("exact rerank candidate document id is out of range");
        }
        const uint32_t document_len = docs.token_counts[doc];
        if (document_len == 0) {
            throw std::invalid_argument("exact rerank candidate has no tokens");
        }
        const float* document_tokens =
            docs.tokens + static_cast<size_t>(docs.token_offsets[doc]) * docs.dim;
        scratch.exact_scores.data()[slot] = exact_mean_maxsim_reference(
            query_tokens, query_len, document_tokens, document_len, docs.dim);
    }

    RerankStats stats;
    stats.candidates_scored = candidate_count;
    stats.results_written =
        select_topk(candidate_ids, candidate_count, topk, scratch, out_ids);
    return stats;
}

template <bool SortCandidatesByTokenOffset>
inline RerankStats rerank_blocked_sgemm(const float* query_tokens,
                                        uint32_t query_len,
                                        const uint32_t* candidate_ids,
                                        uint32_t candidate_count,
                                        uint32_t topk,
                                        const TokenStoreView& docs,
                                        ExactRerankScratch& scratch,
                                        uint32_t* out_ids) {
    validate_common_inputs(
        query_tokens, query_len, candidate_ids, candidate_count, out_ids, topk,
        docs, scratch);
    if (scratch.block_token_capacity >
            static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        query_len > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        docs.dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CBLAS dimensions exceed int range");
    }

    uint32_t* pack_order = scratch.pack_order.data();
    if constexpr (SortCandidatesByTokenOffset) {
        for (uint32_t slot = 0; slot < candidate_count; ++slot) {
            const uint32_t doc = candidate_ids[slot];
            if (doc >= docs.num_docs) {
                throw std::out_of_range(
                    "exact rerank candidate document id is out of range");
            }
            const uint32_t document_len = docs.token_counts[doc];
            if (document_len == 0) {
                throw std::invalid_argument(
                    "exact rerank candidate has no tokens");
            }
            if (document_len > scratch.block_token_capacity) {
                throw std::invalid_argument(
                    "rerank block is smaller than a candidate document");
            }
            pack_order[slot] = slot;
        }
        std::sort(pack_order, pack_order + candidate_count,
                  [&](uint32_t lhs, uint32_t rhs) {
                      const uint32_t lhs_doc = candidate_ids[lhs];
                      const uint32_t rhs_doc = candidate_ids[rhs];
                      const uint64_t lhs_offset = docs.token_offsets[lhs_doc];
                      const uint64_t rhs_offset = docs.token_offsets[rhs_doc];
                      if (lhs_offset != rhs_offset) return lhs_offset < rhs_offset;
                      return lhs_doc < rhs_doc;
                  });
    }

    RerankStats stats;
    stats.candidates_scored = candidate_count;
    uint32_t next = 0;
    while (next < candidate_count) {
        uint32_t block_docs = 0;
        uint32_t block_tokens = 0;
        while (next < candidate_count) {
            uint32_t slot = next;
            if constexpr (SortCandidatesByTokenOffset) {
                slot = pack_order[next];
            }
            const uint32_t doc = candidate_ids[slot];
            if constexpr (!SortCandidatesByTokenOffset) {
                if (doc >= docs.num_docs) {
                    throw std::out_of_range(
                        "exact rerank candidate document id is out of range");
                }
            }
            const uint32_t document_len = docs.token_counts[doc];
            if constexpr (!SortCandidatesByTokenOffset) {
                if (document_len == 0) {
                    throw std::invalid_argument(
                        "exact rerank candidate has no tokens");
                }
                if (document_len > scratch.block_token_capacity) {
                    throw std::invalid_argument(
                        "rerank block is smaller than a candidate document");
                }
            }
            if (block_docs != 0 &&
                document_len > scratch.block_token_capacity - block_tokens) {
                break;
            }

            scratch.segment_offsets.data()[block_docs] = block_tokens;
            scratch.segment_slots.data()[block_docs] = slot;
            const float* source =
                docs.tokens + static_cast<size_t>(docs.token_offsets[doc]) * docs.dim;
            float* destination =
                scratch.packed_tokens.data() + static_cast<size_t>(block_tokens) * docs.dim;
            std::memcpy(destination, source,
                        static_cast<size_t>(document_len) * docs.dim * sizeof(float));
            block_tokens += document_len;
            ++block_docs;
            ++next;
        }
        scratch.segment_offsets.data()[block_docs] = block_tokens;

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<int>(query_len),
                    static_cast<int>(block_tokens),
                    static_cast<int>(docs.dim),
                    1.0f,
                    query_tokens,
                    static_cast<int>(docs.dim),
                    scratch.packed_tokens.data(),
                    static_cast<int>(docs.dim),
                    0.0f,
                    scratch.score_matrix.data(),
                    static_cast<int>(block_tokens));

        for (uint32_t block_doc = 0; block_doc < block_docs; ++block_doc) {
            const uint32_t begin = scratch.segment_offsets.data()[block_doc];
            const uint32_t end = scratch.segment_offsets.data()[block_doc + 1];
            float total = 0.0f;
            for (uint32_t qi = 0; qi < query_len; ++qi) {
                const float* row =
                    scratch.score_matrix.data() + static_cast<size_t>(qi) * block_tokens;
                float best = -std::numeric_limits<float>::infinity();
                for (uint32_t token = begin; token < end; ++token) {
                    if (row[token] > best) best = row[token];
                }
                total += best;
            }
            const uint32_t slot = scratch.segment_slots.data()[block_doc];
            const float score = total / static_cast<float>(query_len);
#if MNG_EXACT_RERANK_VALIDATE
            if (!std::isfinite(score)) {
                throw std::runtime_error("exact rerank produced a non-finite score");
            }
#endif
            scratch.exact_scores.data()[slot] = score;
        }
        ++stats.gemm_calls;
        stats.packed_tokens += block_tokens;
    }

    stats.results_written =
        select_topk(candidate_ids, candidate_count, topk, scratch, out_ids);
    return stats;
}

template <bool SortCandidatesByTokenOffset>
inline RerankStats rerank_blocked_sgemm_token_major_impl(
    const float* query_tokens,
    uint32_t query_len,
    const uint32_t* candidate_ids,
    uint32_t candidate_count,
    uint32_t topk,
    const TokenStoreView& docs,
    ExactRerankScratch& scratch,
    uint32_t* out_ids,
    const float* gemm_query,
    CBLAS_TRANSPOSE gemm_query_operation,
    int gemm_query_leading_dim) {
    validate_common_inputs(
        query_tokens, query_len, candidate_ids, candidate_count, out_ids, topk,
        docs, scratch);
    if (scratch.block_token_capacity >
            static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        query_len > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        docs.dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CBLAS dimensions exceed int range");
    }
    if (gemm_query == nullptr || gemm_query_leading_dim <= 0) {
        throw std::invalid_argument("SGEMM query matrix is invalid");
    }

    uint32_t* pack_order = scratch.pack_order.data();
    if constexpr (SortCandidatesByTokenOffset) {
        for (uint32_t slot = 0; slot < candidate_count; ++slot) {
#if !MNG_EXACT_RERANK_TRUST_INTERNAL_CANDIDATES
            const uint32_t doc = candidate_ids[slot];
            if (doc >= docs.num_docs) {
                throw std::out_of_range(
                    "exact rerank candidate document id is out of range");
            }
            const uint32_t document_len = docs.token_counts[doc];
            if (document_len == 0) {
                throw std::invalid_argument(
                    "exact rerank candidate has no tokens");
            }
            if (document_len > scratch.block_token_capacity) {
                throw std::invalid_argument(
                    "rerank block is smaller than a candidate document");
            }
#endif
            pack_order[slot] = slot;
        }
        std::sort(pack_order, pack_order + candidate_count,
                  [&](uint32_t lhs, uint32_t rhs) {
                      const uint32_t lhs_doc = candidate_ids[lhs];
                      const uint32_t rhs_doc = candidate_ids[rhs];
                      const uint64_t lhs_offset = docs.token_offsets[lhs_doc];
                      const uint64_t rhs_offset = docs.token_offsets[rhs_doc];
                      if (lhs_offset != rhs_offset) return lhs_offset < rhs_offset;
                      return lhs_doc < rhs_doc;
                  });
    }

    RerankStats stats;
#if MNG_EXACT_RERANK_DIAGNOSTICS
    stats.candidates_scored = candidate_count;
#endif
    uint32_t next = 0;
    while (next < candidate_count) {
        uint32_t block_docs = 0;
        uint32_t block_tokens = 0;
        while (next < candidate_count) {
            uint32_t slot = next;
            if constexpr (SortCandidatesByTokenOffset) {
                slot = pack_order[next];
            }
            const uint32_t doc = candidate_ids[slot];
#if !MNG_EXACT_RERANK_TRUST_INTERNAL_CANDIDATES
            if (doc >= docs.num_docs) {
                throw std::out_of_range(
                    "exact rerank candidate document id is out of range");
            }
#endif
            const uint32_t document_len = docs.token_counts[doc];
#if !MNG_EXACT_RERANK_TRUST_INTERNAL_CANDIDATES
            if (document_len == 0) {
                throw std::invalid_argument(
                    "exact rerank candidate has no tokens");
            }
            if (document_len > scratch.block_token_capacity) {
                throw std::invalid_argument(
                    "rerank block is smaller than a candidate document");
            }
#endif
            if (block_docs != 0 &&
                document_len > scratch.block_token_capacity - block_tokens) {
                break;
            }

            scratch.segment_offsets.data()[block_docs] = block_tokens;
            scratch.segment_slots.data()[block_docs] = slot;
            const float* source =
                docs.tokens + static_cast<size_t>(docs.token_offsets[doc]) * docs.dim;
            float* destination =
                scratch.packed_tokens.data() + static_cast<size_t>(block_tokens) * docs.dim;
            std::memcpy(destination, source,
                        static_cast<size_t>(document_len) * docs.dim * sizeof(float));
            block_tokens += document_len;
            ++block_docs;
            ++next;
        }
        scratch.segment_offsets.data()[block_docs] = block_tokens;

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    gemm_query_operation,
                    static_cast<int>(block_tokens),
                    static_cast<int>(query_len),
                    static_cast<int>(docs.dim),
                    1.0f,
                    scratch.packed_tokens.data(),
                    static_cast<int>(docs.dim),
                    gemm_query,
                    gemm_query_leading_dim,
                    0.0f,
                    scratch.score_matrix.data(),
                    static_cast<int>(query_len));

#if defined(__AVX512F__)
        if (query_len == 32) {
            for (uint32_t block_doc = 0; block_doc < block_docs; ++block_doc) {
                const uint32_t begin = scratch.segment_offsets.data()[block_doc];
                const uint32_t end = scratch.segment_offsets.data()[block_doc + 1];
                const uint32_t slot = scratch.segment_slots.data()[block_doc];
                const float score = reduce_token_major_segment_q32_avx512(
                    scratch.score_matrix.data(), begin, end);
#if MNG_EXACT_RERANK_VALIDATE
                if (!std::isfinite(score)) {
                    throw std::runtime_error("exact rerank produced a non-finite score");
                }
#endif
                scratch.exact_scores.data()[slot] = score;
            }
        } else
#endif
        {
            float* query_maxima = scratch.query_maxima.data();
            for (uint32_t block_doc = 0; block_doc < block_docs; ++block_doc) {
                const uint32_t begin = scratch.segment_offsets.data()[block_doc];
                const uint32_t end = scratch.segment_offsets.data()[block_doc + 1];
                const uint32_t slot = scratch.segment_slots.data()[block_doc];
                const float score = reduce_token_major_segment_generic(
                    scratch.score_matrix.data(), query_len, begin, end,
                    query_maxima);
#if MNG_EXACT_RERANK_VALIDATE
                if (!std::isfinite(score)) {
                    throw std::runtime_error("exact rerank produced a non-finite score");
                }
#endif
                scratch.exact_scores.data()[slot] = score;
            }
        }
#if MNG_EXACT_RERANK_DIAGNOSTICS
        ++stats.gemm_calls;
        stats.packed_tokens += block_tokens;
#endif
    }

    stats.results_written =
        select_topk(candidate_ids, candidate_count, topk, scratch, out_ids);
    return stats;
}

template <bool SortCandidatesByTokenOffset>
inline RerankStats rerank_blocked_sgemm_token_major(
    const float* query_tokens,
    uint32_t query_len,
    const uint32_t* candidate_ids,
    uint32_t candidate_count,
    uint32_t topk,
    const TokenStoreView& docs,
    ExactRerankScratch& scratch,
    uint32_t* out_ids) {
    return rerank_blocked_sgemm_token_major_impl<SortCandidatesByTokenOffset>(
        query_tokens, query_len, candidate_ids, candidate_count, topk, docs,
        scratch, out_ids, query_tokens, CblasTrans,
        static_cast<int>(docs.dim));
}

template <bool SortCandidatesByTokenOffset>
inline RerankStats rerank_blocked_sgemm_token_major_pretransposed(
    const float* query_tokens,
    uint32_t query_len,
    const uint32_t* candidate_ids,
    uint32_t candidate_count,
    uint32_t topk,
    const TokenStoreView& docs,
    ExactRerankScratch& scratch,
    uint32_t* out_ids) {
    float* query_transposed = scratch.query_transposed.data();
    transpose_query_once(query_tokens, query_len, docs.dim, query_transposed);
    RerankStats stats =
        rerank_blocked_sgemm_token_major_impl<SortCandidatesByTokenOffset>(
            query_tokens, query_len, candidate_ids, candidate_count, topk, docs,
            scratch, out_ids, query_transposed, CblasNoTrans,
            static_cast<int>(query_len));
#if MNG_EXACT_RERANK_DIAGNOSTICS
    stats.query_transpose_elements =
        static_cast<uint64_t>(query_len) * docs.dim;
#endif
    return stats;
}

template <bool SortCandidatesByTokenOffset>
inline RerankStats rerank_blocked_sgemm_token_major_auto(
    const float* query_tokens,
    uint32_t query_len,
    const uint32_t* candidate_ids,
    uint32_t candidate_count,
    uint32_t topk,
    const TokenStoreView& docs,
    ExactRerankScratch& scratch,
    uint32_t* out_ids) {
#if MNG_EXACT_RERANK_PRETRANSPOSE_Q32
    if (query_len == 32) {
        return rerank_blocked_sgemm_token_major_pretransposed<
            SortCandidatesByTokenOffset>(
            query_tokens, query_len, candidate_ids, candidate_count, topk, docs,
            scratch, out_ids);
    }
#endif
    return rerank_blocked_sgemm_token_major<SortCandidatesByTokenOffset>(
        query_tokens, query_len, candidate_ids, candidate_count, topk, docs,
        scratch, out_ids);
}

}  // namespace mng_exact_rerank
