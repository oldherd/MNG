#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace baseline {

// Recall@k: fraction of GT top-k that appear in the approximate top-k
inline float recall_at_k(const int32_t* approx, const int32_t* gt, size_t k) {
    std::unordered_set<int32_t> gt_set(gt, gt + k);
    size_t hits = 0;
    for (size_t i = 0; i < k; ++i) {
        if (gt_set.count(approx[i])) ++hits;
    }
    return (float)hits / (float)k;
}

// MRR@k: reciprocal rank of the first relevant result within top-k
// qrels[i] = set of relevant doc IDs for query i
inline float mrr_at_k(const int32_t* results, size_t k,
                      const std::unordered_set<int32_t>& qrels) {
    for (size_t i = 0; i < k; ++i) {
        if (qrels.count(results[i])) {
            return 1.0f / (float)(i + 1);
        }
    }
    return 0.0f;
}

// Success@k: 1 if any relevant doc appears in top-k, else 0
inline float success_at_k(const int32_t* results, size_t k,
                          const std::unordered_set<int32_t>& qrels) {
    for (size_t i = 0; i < k; ++i) {
        if (qrels.count(results[i])) return 1.0f;
    }
    return 0.0f;
}

// Batch evaluation helper: compute recall@k for all queries against GT topk
struct EvalResults {
    float recall_at_10 = 0, recall_at_100 = 0, recall_at_1000 = 0;
    float avg_latency_ms = 0;
    double build_time_sec = 0;
    size_t index_size_bytes = 0;
};

}  // namespace baseline
