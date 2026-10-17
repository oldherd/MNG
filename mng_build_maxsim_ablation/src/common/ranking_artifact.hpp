#pragma once

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/maxsim.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace baseline {

inline void write_rankings_with_exact_maxsim(
        const std::string& results_path,
        const std::vector<std::vector<int32_t>>& rankings,
        size_t width,
        const std::string& doc_bundle_path,
        const std::string& query_bundle_path) {
    if (results_path.empty()) return;

    auto docs = mv::MVBundle::load(doc_bundle_path);
    auto queries = mv::MVBundle::load(query_bundle_path);
    if (queries.dtype() != mv::DType::Float32) {
        throw std::runtime_error("ranking score export requires fp32 query embeddings");
    }
    if (docs.dim() != queries.dim()) {
        throw std::runtime_error("ranking score export query/document dimension mismatch");
    }
    if (rankings.size() != queries.n_docs()) {
        throw std::runtime_error("ranking score export query count mismatch");
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::vector<float>> scores(rankings.size());
    for (size_t qi = 0; qi < rankings.size(); ++qi) {
        const auto query = queries.get(qi);
        const size_t count = std::min(width, rankings[qi].size());
        scores[qi].assign(count, nan);
        for (size_t rank = 0; rank < count; ++rank) {
            const int32_t doc_id = rankings[qi][rank];
            if (doc_id < 0 || static_cast<size_t>(doc_id) >= docs.n_docs()) continue;
            const auto doc = docs.get(static_cast<size_t>(doc_id));
            if (docs.dtype() == mv::DType::Float16) {
                scores[qi][rank] = baseline::maxsim_fp16(
                    query.as_fp32(), static_cast<int>(query.tokens),
                    doc.as_fp16(), static_cast<int>(doc.tokens), docs.dim());
            } else {
                scores[qi][rank] = baseline::maxsim_fp32(
                    query.as_fp32(), static_cast<int>(query.tokens),
                    doc.as_fp32(), static_cast<int>(doc.tokens), docs.dim());
            }
        }
    }

    write_rankings_i32(results_path, rankings, width);
    write_ranking_scores_f32(results_path, scores, width,
                             "exact_maxsim_recomputed_after_search");
}

}  // namespace baseline
