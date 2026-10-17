// PLAID/ColBERTv2 vector-search searcher.
//
// Search pipeline:
//   1. top-nprobe centroid cells per query token -> candidate pids
//   2. pruned centroid interaction to ndocs
//   3. full centroid interaction to ndocs / 4
//   4. residual decompression + MaxSim final ranking

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/index_io.hpp"
#include "common/ranking_artifact.hpp"
#include "common/search_runtime.hpp"
#include "common/timer.hpp"
#include "plaid_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) sum += a[d] * b[d];
    return sum;
}

void compute_centroid_scores(const plaid::PLAIDIndex& index,
                             const float* Q,
                             int n_q_tokens,
                             std::vector<float>& scores) {
    scores.assign((std::size_t)index.n_centroids * (std::size_t)n_q_tokens, 0.0f);
    for (int c = 0; c < index.n_centroids; ++c) {
        const float* centroid = index.centroids.data() + (std::size_t)c * (std::size_t)index.dim;
        float* row = scores.data() + (std::size_t)c * (std::size_t)n_q_tokens;
        for (int qt = 0; qt < n_q_tokens; ++qt) {
            row[qt] = dot_product(centroid, Q + (std::size_t)qt * (std::size_t)index.dim, index.dim);
        }
    }
}

std::vector<int32_t> top_probe_cells(const std::vector<float>& centroid_scores,
                                     int n_centroids,
                                     int n_q_tokens,
                                     int nprobe) {
    using Pair = std::pair<float, int32_t>;
    std::vector<int32_t> cells;
    cells.reserve((std::size_t)nprobe * (std::size_t)n_q_tokens);

    for (int qt = 0; qt < n_q_tokens; ++qt) {
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;
        for (int c = 0; c < n_centroids; ++c) {
            float score = centroid_scores[(std::size_t)c * (std::size_t)n_q_tokens + qt];
            if ((int)heap.size() < nprobe) {
                heap.push({score, c});
            } else if (score > heap.top().first) {
                heap.pop();
                heap.push({score, c});
            }
        }
        while (!heap.empty()) {
            cells.push_back(heap.top().second);
            heap.pop();
        }
    }

    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
}

std::vector<int32_t> gather_candidate_pids(const plaid::PLAIDIndex& index,
                                           const std::vector<int32_t>& cells) {
    std::vector<int32_t> pids;
    std::size_t reserve = 0;
    for (int32_t cell : cells) reserve += index.ivf_pid_lists[(std::size_t)cell].size();
    pids.reserve(reserve);

    for (int32_t cell : cells) {
        const auto& list = index.ivf_pid_lists[(std::size_t)cell];
        pids.insert(pids.end(), list.begin(), list.end());
    }
    std::sort(pids.begin(), pids.end());
    pids.erase(std::unique(pids.begin(), pids.end()), pids.end());
    return pids;
}

std::vector<uint8_t> centroid_prune_mask(const std::vector<float>& centroid_scores,
                                         int n_centroids,
                                         int n_q_tokens,
                                         float threshold) {
    std::vector<uint8_t> mask((std::size_t)n_centroids, 0);
    for (int c = 0; c < n_centroids; ++c) {
        const float* row = centroid_scores.data() + (std::size_t)c * (std::size_t)n_q_tokens;
        float best = row[0];
        for (int qt = 1; qt < n_q_tokens; ++qt) best = std::max(best, row[qt]);
        mask[(std::size_t)c] = (best >= threshold) ? 1 : 0;
    }
    return mask;
}

float centroid_interaction_score(const plaid::PLAIDIndex& index,
                                 int32_t pid,
                                 const std::vector<float>& centroid_scores,
                                 int n_q_tokens,
                                 const std::vector<uint8_t>* mask,
                                 std::vector<int32_t>& unique_codes,
                                 std::vector<float>& per_query_max) {
    const std::size_t start = (std::size_t)index.offsets[(std::size_t)pid];
    const std::size_t end = (std::size_t)index.offsets[(std::size_t)pid + 1u];
    unique_codes.clear();
    unique_codes.reserve(end - start);
    for (std::size_t t = start; t < end; ++t) unique_codes.push_back(index.codes[t]);
    std::sort(unique_codes.begin(), unique_codes.end());
    unique_codes.erase(std::unique(unique_codes.begin(), unique_codes.end()), unique_codes.end());

    std::fill(per_query_max.begin(), per_query_max.end(), -9999.0f);
    for (int32_t code : unique_codes) {
        if (mask && !(*mask)[(std::size_t)code]) continue;
        const float* row = centroid_scores.data() + (std::size_t)code * (std::size_t)n_q_tokens;
        for (int qt = 0; qt < n_q_tokens; ++qt) {
            per_query_max[(std::size_t)qt] = std::max(per_query_max[(std::size_t)qt], row[qt]);
        }
    }

    float score = 0.0f;
    for (float x : per_query_max) score += x;
    return score;
}

std::vector<int32_t> select_top_by_centroid_score(const plaid::PLAIDIndex& index,
                                                   const std::vector<int32_t>& pids,
                                                   const std::vector<float>& centroid_scores,
                                                   int n_q_tokens,
                                                   const std::vector<uint8_t>* mask,
                                                   int limit) {
    if (pids.empty() || limit <= 0) return {};

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(pids.size());
    std::vector<int32_t> unique_codes;
    std::vector<float> per_query_max((std::size_t)n_q_tokens);
    for (int32_t pid : pids) {
        float score = centroid_interaction_score(index, pid, centroid_scores, n_q_tokens,
                                                 mask, unique_codes, per_query_max);
        scored.push_back({score, pid});
    }

    auto desc = [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    };
    if ((int)scored.size() > limit) {
        std::nth_element(scored.begin(), scored.begin() + limit, scored.end(), desc);
        scored.resize((std::size_t)limit);
    }
    std::sort(scored.begin(), scored.end(), desc);

    std::vector<int32_t> result;
    result.reserve(scored.size());
    for (const auto& item : scored) result.push_back(item.second);
    return result;
}

uint8_t unpack_bucket(const uint8_t* row, int dim_index, int nbits) {
    const int bit_pos = dim_index * nbits;
    const int byte_index = bit_pos / 8;
    const int offset_in_byte = bit_pos % 8;
    const int shift = 8 - offset_in_byte - nbits;
    const uint8_t mask = (uint8_t)((1u << nbits) - 1u);
    return (uint8_t)((row[byte_index] >> shift) & mask);
}

float decompressed_maxsim_score(const plaid::PLAIDIndex& index,
                                int32_t pid,
                                const float* Q,
                                int n_q_tokens,
                                std::vector<float>& token,
                                std::vector<float>& per_query_max) {
    const std::size_t start = (std::size_t)index.offsets[(std::size_t)pid];
    const std::size_t end = (std::size_t)index.offsets[(std::size_t)pid + 1u];
    std::fill(per_query_max.begin(), per_query_max.end(), -9999.0f);

    for (std::size_t t = start; t < end; ++t) {
        int32_t code = index.codes[t];
        const float* centroid = index.centroids.data() + (std::size_t)code * (std::size_t)index.dim;
        const uint8_t* residual = index.residuals.data() + t * index.packed_dim;

        double ss = 0.0;
        for (int d = 0; d < index.dim; ++d) {
            uint8_t bucket = unpack_bucket(residual, d, index.nbits);
            float value = centroid[d] + index.bucket_weights[(std::size_t)bucket];
            token[(std::size_t)d] = value;
            ss += (double)value * (double)value;
        }
        float inv_norm = (float)(1.0 / std::sqrt(std::max(ss, 1e-30)));

        for (int qt = 0; qt < n_q_tokens; ++qt) {
            const float* q = Q + (std::size_t)qt * (std::size_t)index.dim;
            float dot = 0.0f;
            for (int d = 0; d < index.dim; ++d) {
                dot += q[d] * token[(std::size_t)d] * inv_norm;
            }
            per_query_max[(std::size_t)qt] = std::max(per_query_max[(std::size_t)qt], dot);
        }
    }

    float score = 0.0f;
    for (float x : per_query_max) score += x;
    return score;
}

std::vector<int32_t> final_rank(const plaid::PLAIDIndex& index,
                                const std::vector<int32_t>& pids,
                                const float* Q,
                                int n_q_tokens,
                                int k) {
    if (pids.empty() || k <= 0) return {};

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(pids.size());
    std::vector<float> token((std::size_t)index.dim);
    std::vector<float> per_query_max((std::size_t)n_q_tokens);

    for (int32_t pid : pids) {
        float score = decompressed_maxsim_score(index, pid, Q, n_q_tokens, token, per_query_max);
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

    std::vector<int32_t> result;
    result.reserve(scored.size());
    for (const auto& item : scored) result.push_back(item.second);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    baseline::enforce_single_thread_searcher("plaid");

    auto args = baseline::parse_args(argc, argv);
    if (args.index_dir.empty() || args.query_bundle.empty()) {
        std::cerr << "Usage: searcher_plaid --index_dir <dir> --query <bundle> "
                  << "[--doc <bundle>] [--gt <dir>] [--k N] [--output file] "
                  << "[--param nprobe=4] [--param centroid_score_threshold=0.4] "
                  << "[--param ndocs=4096] [--param profile=k1000]\n";
        return 1;
    }

    int nprobe = baseline::get_extra_int(args, "nprobe", 4);
    float threshold = baseline::get_extra_float(args, "centroid_score_threshold", 0.4f);
    int ndocs = baseline::get_extra_int(args, "ndocs", 4096);
    std::string profile = baseline::get_extra_string(
        args, "profile", "paper_k1000");
    nprobe = std::max(1, nprobe);
    ndocs = std::max(1, ndocs);
    const int ndocs_stage3 = std::max(1, ndocs / 4);

    baseline::Timer load_timer;
    load_timer.start();

    plaid::PLAIDIndex index = plaid::PLAIDIndex::load(args.index_dir);
    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    if (query_bundle.dtype() != mv::DType::Float32) {
        throw std::runtime_error("PLAID searcher expects fp32 query .mvbundle input");
    }
    if (query_bundle.dim() != index.dim) {
        throw std::runtime_error("dimension mismatch between PLAID index and query bundle");
    }

    const std::size_t n_queries = query_bundle.n_docs();
    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
    }

    load_timer.stop();
    double load_sec = load_timer.elapsed_sec();

    const int k = args.k;
    std::cerr << "[plaid-search] loaded in " << load_sec
              << "s; profile=" << profile
              << " nprobe=" << nprobe
              << " threshold=" << threshold
              << " ndocs=" << ndocs
              << " ndocs_stage3=" << ndocs_stage3
              << " k=" << k << "\n";

    std::vector<std::vector<int32_t>> all_topk(n_queries);
    std::vector<double> stage1_counts(n_queries, 0.0);
    std::vector<double> stage2_counts(n_queries, 0.0);
    std::vector<double> stage3_counts(n_queries, 0.0);

    baseline::Timer search_timer;
    search_timer.start();

    std::vector<float> centroid_scores;
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        if (qi % 100 == 0) {
            std::cerr << "\r[plaid-search] query " << qi << "/" << n_queries << std::flush;
        }

        auto qview = query_bundle.get(qi);
        const float* Q = qview.as_fp32();
        int n_q_tokens = (int)qview.tokens;

        compute_centroid_scores(index, Q, n_q_tokens, centroid_scores);
        std::vector<int32_t> cells = top_probe_cells(centroid_scores, index.n_centroids,
                                                     n_q_tokens, nprobe);
        std::vector<int32_t> candidates = gather_candidate_pids(index, cells);
        stage1_counts[qi] = (double)candidates.size();

        std::vector<uint8_t> mask = centroid_prune_mask(centroid_scores, index.n_centroids,
                                                        n_q_tokens, threshold);
        std::vector<int32_t> stage2 = select_top_by_centroid_score(index, candidates,
                                                                   centroid_scores,
                                                                   n_q_tokens,
                                                                   &mask, ndocs);
        stage2_counts[qi] = (double)stage2.size();

        std::vector<int32_t> stage3 = select_top_by_centroid_score(index, stage2,
                                                                   centroid_scores,
                                                                   n_q_tokens,
                                                                   nullptr,
                                                                   ndocs_stage3);
        stage3_counts[qi] = (double)stage3.size();

        all_topk[qi] = final_rank(index, stage3, Q, n_q_tokens, k);
    }

    search_timer.stop();
    double search_sec = search_timer.elapsed_sec();
    double per_query_ms = (search_sec * 1000.0) / (double)n_queries;
    std::cerr << "\n[plaid-search] done in " << search_sec << "s ("
              << per_query_ms << " ms/q)\n";

    baseline::write_rankings_with_exact_maxsim(
        args.output, all_topk, static_cast<size_t>(k),
        args.doc_bundle, args.query_bundle);

    float avg_recall_10 = 0, avg_recall_100 = 0, avg_recall_1000 = 0;
    float avg_mrr_10 = 0, avg_success_5 = 0, avg_success_100 = 0;
    if (gt) {
        for (std::size_t qi = 0; qi < n_queries; ++qi) {
            const int32_t* gt_topk = gt->topk_indices_row(qi);
            int have = (int)all_topk[qi].size();
            if (have >= 10) avg_recall_10 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 10);
            if (have >= 100 && gt->top_k() >= 100) avg_recall_100 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 100);
            if (have >= 1000 && gt->top_k() >= 1000) avg_recall_1000 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 1000);
            std::unordered_set<int32_t> qrels_set;
            qrels_set.insert(gt_topk[0]);
            if (have >= 10) avg_mrr_10 += baseline::mrr_at_k(all_topk[qi].data(), 10, qrels_set);
            if (have >= 5) avg_success_5 += baseline::success_at_k(all_topk[qi].data(), 5, qrels_set);
            if (have >= 100) avg_success_100 += baseline::success_at_k(all_topk[qi].data(), 100, qrels_set);
        }
        avg_recall_10 /= n_queries;
        avg_recall_100 /= n_queries;
        avg_recall_1000 /= n_queries;
        avg_mrr_10 /= n_queries;
        avg_success_5 /= n_queries;
        avg_success_100 /= n_queries;
    }

    double avg_stage1 = 0.0, avg_stage2 = 0.0, avg_stage3 = 0.0;
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        avg_stage1 += stage1_counts[qi];
        avg_stage2 += stage2_counts[qi];
        avg_stage3 += stage3_counts[qi];
    }
    avg_stage1 /= (double)n_queries;
    avg_stage2 /= (double)n_queries;
    avg_stage3 /= (double)n_queries;

    std::cout << "[plaid] Recall@10: " << avg_recall_10 << "\n";
    std::cout << "[plaid] Recall@100: " << avg_recall_100 << "\n";
    std::cout << "[plaid] Recall@1000: " << avg_recall_1000 << "\n";
    std::cout << "[plaid] MRR@10: " << avg_mrr_10 << "\n";
    std::cout << "[plaid] Success@5: " << avg_success_5 << "\n";
    std::cout << "[plaid] Success@100: " << avg_success_100 << "\n";
    std::cout << "[plaid] Avg stage1 candidates: " << avg_stage1 << "\n";
    std::cout << "[plaid] Avg stage2 docs: " << avg_stage2 << "\n";
    std::cout << "[plaid] Avg stage3 docs: " << avg_stage3 << "\n";

    if (!args.output.empty()) {
        std::string dataset = !args.doc_bundle.empty() ? args.doc_bundle : index.input_doc_bundle;
        std::vector<std::pair<std::string, std::string>> params = {
            {"layout", plaid::kLayout},
            {"compression", "colbertv2_residual"},
            {"metric", "ip"},
            {"profile", profile},
            {"n_centroids", std::to_string(index.n_centroids)},
            {"nbits", std::to_string(index.nbits)},
            {"nprobe", std::to_string(nprobe)},
            {"centroid_score_threshold", std::to_string(threshold)},
            {"ndocs", std::to_string(ndocs)},
            {"ndocs_stage3", std::to_string(ndocs_stage3)},
            {"avg_stage1_candidates", std::to_string(avg_stage1)},
            {"avg_stage2_docs", std::to_string(avg_stage2)},
            {"avg_stage3_docs", std::to_string(avg_stage3)},
            {"final_scoring", "residual_decompressed_maxsim"},
            {"exact_refine", "false"},
            {"shared_doc_store", "none"},
        };
        baseline::write_results_json(args.output, "plaid", dataset, k, params,
            avg_recall_10, avg_recall_100, avg_recall_1000,
            avg_mrr_10, avg_success_5, avg_success_100,
            index.build_time_sec, load_sec, search_sec, per_query_ms,
            plaid::logical_index_bytes(args.index_dir));
    }

    return 0;
}
