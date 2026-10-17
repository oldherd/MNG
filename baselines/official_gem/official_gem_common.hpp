#pragma once

#include "mvbundle.hpp"
#include "common/fp16_utils.hpp"
#include "common/maxsim.hpp"
#include "official_gem/official_gem_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "hnswlib/vectorset.h"

namespace official_gem {

inline void l2_normalize(float* x, int dim) {
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) ss += (double)x[d] * (double)x[d];
    double inv = 1.0 / std::sqrt(std::max(ss, 1e-30));
    for (int d = 0; d < dim; ++d) x[d] = (float)((double)x[d] * inv);
}

inline bool is_nonzero_token(const float* x, int dim) {
    for (int d = 0; d < dim; ++d) {
        if (x[d] != 0.0f) return true;
    }
    return false;
}

inline void set_official_cluster_count(int n_cquant) {
    if (n_cquant <= 0) throw std::runtime_error("n_cquant must be positive");
    NUM_CLUSTER_CALC = n_cquant;
}

struct OfficialBaseMemory {
    std::vector<float> base_data;
    std::vector<int> base_codes;
    std::vector<vectorset> base;

    std::size_t data_bytes() const {
        return base_data.size() * sizeof(float);
    }

    std::size_t code_bytes() const {
        return base_codes.size() * sizeof(int);
    }
};

inline OfficialBaseMemory make_official_base_memory(const mv::MVBundle& doc_bundle,
                                                    const OfficialGEMIndex& index) {
    static_assert(sizeof(int32_t) == sizeof(int), "official hnswlib expects int token codes");
    if (doc_bundle.dim() != index.dim || doc_bundle.n_docs() != index.n_docs ||
        doc_bundle.n_tokens() != index.n_tokens) {
        throw std::runtime_error("official GEM base materialization bundle/index mismatch");
    }
    if (doc_bundle.dtype() != mv::DType::Float16) {
        throw std::runtime_error("official GEM base materialization expects fp16 doc bundle");
    }
    if (index.token_codes.size() != index.n_tokens || index.offsets.size() != index.n_docs + 1u ||
        index.doclens.size() != index.n_docs) {
        throw std::runtime_error("official GEM base materialization index metadata mismatch");
    }

    OfficialBaseMemory memory;
    memory.base_data.resize(index.n_tokens * (std::size_t)index.dim);
    memory.base_codes.resize(index.n_tokens);
    memory.base.reserve(index.n_docs);

    const uint16_t* fp16 = doc_bundle.embeddings_fp16();
    for (std::size_t token = 0; token < index.n_tokens; ++token) {
        const uint16_t* src = fp16 + token * (std::size_t)index.dim;
        float* dst = memory.base_data.data() + token * (std::size_t)index.dim;
        for (int d = 0; d < index.dim; ++d) {
            dst[d] = baseline::fp16_to_fp32(src[d]);
        }
        memory.base_codes[token] = (int)index.token_codes[token];
    }

    for (std::size_t doc = 0; doc < index.n_docs; ++doc) {
        const std::size_t begin = (std::size_t)index.offsets[doc];
        const std::size_t end = (std::size_t)index.offsets[doc + 1u];
        const int len = index.doclens[doc];
        if (begin > end || end > index.n_tokens || len != (int)(end - begin)) {
            throw std::runtime_error("official GEM doc offset/doclen mismatch");
        }
        memory.base.emplace_back(memory.base_data.data() + begin * (std::size_t)index.dim,
                                 memory.base_codes.data() + begin,
                                 index.dim,
                                 len);
    }
    return memory;
}

inline std::vector<std::vector<hnswlib::labeltype>>
make_cluster_sets(const OfficialGEMIndex& index) {
    std::vector<std::vector<hnswlib::labeltype>> out(index.cluster_to_docs.size());
    for (std::size_t c = 0; c < index.cluster_to_docs.size(); ++c) {
        out[c].reserve(index.cluster_to_docs[c].size());
        for (int32_t doc : index.cluster_to_docs[c]) {
            out[c].push_back((hnswlib::labeltype)doc);
        }
    }
    return out;
}

inline std::vector<int> make_cluster_order(int n_cindex) {
    std::vector<int> order((std::size_t)n_cindex);
    std::iota(order.begin(), order.end(), 0);
    return order;
}

inline std::vector<int32_t> top_unique_clusters(const std::vector<float>& scores,
                                                int rows,
                                                int cols,
                                                int topk) {
    std::unordered_set<int32_t> seen;
    std::vector<int32_t> out;
    if (rows <= 0 || cols <= 0 || topk <= 0) return out;
    topk = std::min(topk, cols);
    for (int row = 0; row < rows; ++row) {
        std::vector<std::pair<float, int32_t>> row_scores;
        row_scores.reserve((std::size_t)cols);
        const float* base = scores.data() + (std::size_t)row * (std::size_t)cols;
        for (int col = 0; col < cols; ++col) row_scores.push_back({base[col], col});
        std::partial_sort(row_scores.begin(),
                          row_scores.begin() + topk,
                          row_scores.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
        for (int i = 0; i < topk; ++i) {
            if (seen.insert(row_scores[(std::size_t)i].second).second) {
                out.push_back(row_scores[(std::size_t)i].second);
            }
        }
    }
    return out;
}

inline std::vector<int32_t> rerank_raw_maxsim(const mv::MVBundle& doc_bundle,
                                              const float* Q,
                                              int n_q_tokens,
                                              const std::vector<int32_t>& candidates,
                                              int k) {
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(candidates.size());
    for (int32_t pid : candidates) {
        if (pid < 0 || (std::size_t)pid >= doc_bundle.n_docs()) continue;
        auto dview = doc_bundle.get((std::size_t)pid);
        float score = baseline::maxsim_fp16(
            Q,
            n_q_tokens,
            dview.as_fp16(),
            (int)dview.tokens,
            doc_bundle.dim());
        scored.push_back({score, pid});
    }
    auto desc = [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    };
    if ((int)scored.size() > k) {
        std::nth_element(scored.begin(), scored.begin() + k, scored.end(), desc);
        scored.resize((std::size_t)k);
    }
    std::sort(scored.begin(), scored.end(), desc);

    std::vector<int32_t> out;
    out.reserve(scored.size());
    for (const auto& item : scored) out.push_back(item.second);
    return out;
}

}  // namespace official_gem
