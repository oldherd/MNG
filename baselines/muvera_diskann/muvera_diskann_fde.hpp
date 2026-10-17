#pragma once

#include "muvera_diskann_index.hpp"
#include "utils/mips_point.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace muvera_diskann {

inline int simhash_bucket(const float* vec, int dim,
                          const std::vector<std::vector<float>>& planes) {
    int bucket = 0;
    for (int h = 0; h < static_cast<int>(planes.size()); ++h) {
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot += vec[d] * planes[h][d];
        if (dot > 0.0f) bucket |= (1 << h);
    }
    return bucket;
}

inline void project_vector(const float* in, int dim, const float* projection,
                           int d_proj, float* out) {
    for (int j = 0; j < d_proj; ++j) {
        float value = 0.0f;
        for (int i = 0; i < dim; ++i) value += in[i] * projection[i * d_proj + j];
        out[j] = value;
    }
}

inline void compute_query_fde(const float* tokens, int n_tokens, const Index& idx,
                              float* fde_out) {
    std::fill(fde_out, fde_out + idx.d_fde, 0.0f);
    std::vector<float> bucket_sum(idx.dim);

    for (int r = 0; r < idx.R; ++r) {
        const auto& state = idx.rep_states[r];
        std::vector<std::vector<int>> bucket_members(idx.B);
        for (int t = 0; t < n_tokens; ++t) {
            bucket_members[simhash_bucket(tokens + t * idx.dim, idx.dim,
                                          state.simhash_planes)]
                .push_back(t);
        }
        for (int bucket = 0; bucket < idx.B; ++bucket) {
            if (bucket_members[bucket].empty()) continue;
            std::fill(bucket_sum.begin(), bucket_sum.end(), 0.0f);
            for (int token : bucket_members[bucket]) {
                for (int d = 0; d < idx.dim; ++d) {
                    bucket_sum[d] += tokens[token * idx.dim + d];
                }
            }
            const int offset = r * idx.B * idx.d_proj + bucket * idx.d_proj;
            project_vector(bucket_sum.data(), idx.dim, state.projection.data(),
                           idx.d_proj, fde_out + offset);
        }
    }
}

inline std::vector<float> ball_carve(const float* query, int n_tokens, int dim,
                                     float tau) {
    std::vector<bool> used(n_tokens, false);
    std::vector<float> result;
    result.reserve(static_cast<size_t>(n_tokens) * dim);
    std::vector<float> norms(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        float norm = 0.0f;
        for (int d = 0; d < dim; ++d) norm += query[i * dim + d] * query[i * dim + d];
        norms[i] = std::sqrt(norm);
    }

    for (int i = 0; i < n_tokens; ++i) {
        if (used[i]) continue;
        used[i] = true;
        std::vector<float> sum(query + i * dim, query + (i + 1) * dim);
        for (int j = i + 1; j < n_tokens; ++j) {
            if (used[j]) continue;
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += query[i * dim + d] * query[j * dim + d];
            const float denom = norms[i] * norms[j];
            if (denom > 0.0f && dot / denom >= tau) {
                used[j] = true;
                for (int d = 0; d < dim; ++d) sum[d] += query[j * dim + d];
            }
        }
        result.insert(result.end(), sum.begin(), sum.end());
    }
    return result;
}

inline void build_pq_lut(const Index& idx, const float* query_fde,
                         std::vector<float>& lut) {
    lut.resize(static_cast<size_t>(idx.pq_m) * idx.pq_ksub);
    for (int m = 0; m < idx.pq_m; ++m) {
        const float* query = query_fde + static_cast<size_t>(m) * idx.pq_group_size;
        for (int code = 0; code < idx.pq_ksub; ++code) {
            const float* centroid = idx.pq_centroids.data() +
                (static_cast<size_t>(m) * idx.pq_ksub + code) * idx.pq_group_size;
            float score = 0.0f;
            for (int d = 0; d < idx.pq_group_size; ++d) score += query[d] * centroid[d];
            lut[static_cast<size_t>(m) * idx.pq_ksub + code] = score;
        }
    }
}

struct PQPoint {
    using distanceType = float;

    const uint8_t* code = nullptr;
    const float* lut = nullptr;
    int pq_m = 0;
    int pq_ksub = 0;
    int64_t id_value = -1;

    float distance(const PQPoint& query) const {
        if (code == nullptr || query.lut == nullptr || pq_m != query.pq_m) {
            throw std::runtime_error("muvera_diskann: invalid asymmetric PQ point");
        }
        float score = 0.0f;
        for (int m = 0; m < pq_m; ++m) {
            score += query.lut[static_cast<size_t>(m) * pq_ksub + code[m]];
        }
        return -score;
    }

    bool same_as(const PQPoint& other) const { return code != nullptr && code == other.code; }
    void prefetch() const {
        for (int offset = 0; offset < pq_m; offset += 64) {
            __builtin_prefetch(code + offset);
        }
    }
    long id() const { return id_value; }
};

struct PQPointRange {
    using Point = PQPoint;

    const uint8_t* codes = nullptr;
    size_t n = 0;
    int pq_m = 0;
    int pq_ksub = 0;

    size_t size() const { return n; }
    PQPoint operator[](size_t i) const {
        return PQPoint{codes + i * static_cast<size_t>(pq_m), nullptr,
                       pq_m, pq_ksub, static_cast<int64_t>(i)};
    }
};

struct RawFDEPointRange {
    using Point = parlayANN::Mips_Point<float>;

    float* values = nullptr;
    size_t n = 0;
    int dim = 0;
    typename Point::parameters params;

    RawFDEPointRange(float* values, size_t n, int dim)
        : values(values), n(n), dim(dim), params(dim) {}

    size_t size() const { return n; }
    long dimension() const { return dim; }
    Point operator[](size_t i) const {
        return Point(reinterpret_cast<uint8_t*>(values + i * static_cast<size_t>(dim)),
                     static_cast<long>(i), params);
    }
};

}  // namespace muvera_diskann
