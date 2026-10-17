// WARP vector-search searcher.
//
// Implements the WARP engine path in scalar C++:
// WARPSELECT, compacted centroid-list decompression, and missing-score merge.

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/index_io.hpp"
#include "common/ranking_artifact.hpp"
#include "common/search_runtime.hpp"
#include "common/timer.hpp"
#include "warp_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) sum += a[d] * b[d];
    return sum;
}

bool is_nonzero_token(const float* q, int dim) {
    for (int d = 0; d < dim; ++d) {
        if (q[d] != 0.0f) return true;
    }
    return false;
}

void compute_centroid_scores_query_major(const warp::WARPIndex& index,
                                         const float* Q,
                                         int n_q_tokens,
                                         std::vector<float>& scores) {
    scores.assign((std::size_t)n_q_tokens * (std::size_t)index.n_centroids, 0.0f);
    for (int qt = 0; qt < n_q_tokens; ++qt) {
        const float* q = Q + (std::size_t)qt * (std::size_t)index.dim;
        float* row = scores.data() + (std::size_t)qt * (std::size_t)index.n_centroids;
        for (int c = 0; c < index.n_centroids; ++c) {
            const float* centroid = index.centroids.data() + (std::size_t)c * (std::size_t)index.dim;
            row[c] = dot_product(q, centroid, index.dim);
        }
    }
}

struct SelectedCells {
    std::vector<int32_t> active_tokens;
    std::vector<int32_t> cells;   // active_tokens.size() * nprobe
    std::vector<float> scores;    // same shape
    std::vector<float> mse;       // active_tokens.size()
};

SelectedCells warp_select_centroids(const warp::WARPIndex& index,
                                    const float* Q,
                                    int n_q_tokens,
                                    const std::vector<float>& centroid_scores,
                                    int nprobe,
                                    int t_prime,
                                    int bound) {
    bound = std::min(bound, index.n_centroids);
    nprobe = std::min(nprobe, bound);
    if (bound <= 0 || nprobe <= 0) throw std::runtime_error("invalid WARP nprobe/bound");

    std::vector<int32_t> centroid_idx((std::size_t)index.n_centroids);
    std::iota(centroid_idx.begin(), centroid_idx.end(), 0);

    SelectedCells out;
    for (int qt = 0; qt < n_q_tokens; ++qt) {
        const float* q = Q + (std::size_t)qt * (std::size_t)index.dim;
        if (!is_nonzero_token(q, index.dim)) continue;

        const float* row = centroid_scores.data() + (std::size_t)qt * (std::size_t)index.n_centroids;
        auto centroid_better = [row](int32_t a, int32_t b) {
            if (row[a] != row[b]) return row[a] > row[b];
            return a < b;
        };

        int effective_bound = bound;
        int64_t cumsum = 0;
        int idx = 0;
        while (true) {
            std::partial_sort(centroid_idx.begin(), centroid_idx.begin() + effective_bound,
                              centroid_idx.end(), centroid_better);
            cumsum = 0;
            idx = 0;
            while (cumsum < t_prime && idx < effective_bound) {
                cumsum += index.sizes[(std::size_t)centroid_idx[(std::size_t)idx]];
                ++idx;
            }
            if (cumsum >= t_prime) break;
            if (effective_bound == index.n_centroids) {
                throw std::runtime_error("WARP index contains fewer entries than t_prime");
            }
            effective_bound = std::min(index.n_centroids, effective_bound * 2);
        }

        out.active_tokens.push_back(qt);
        for (int j = 0; j < nprobe; ++j) {
            int32_t c = centroid_idx[(std::size_t)j];
            out.cells.push_back(c);
            out.scores.push_back(row[c]);
        }

        int32_t mse_centroid = centroid_idx[(std::size_t)std::max(0, idx - 1)];
        out.mse.push_back(row[mse_centroid]);
    }
    return out;
}

uint8_t unpack_bucket(const uint8_t* row, int dim_index, int nbits) {
    int bit_pos = dim_index * nbits;
    int byte_index = bit_pos / 8;
    int offset_in_byte = bit_pos % 8;
    int shift = 8 - offset_in_byte - nbits;
    uint8_t mask = (uint8_t)((1u << nbits) - 1u);
    return (uint8_t)((row[byte_index] >> shift) & mask);
}

float residual_dot(const warp::WARPIndex& index,
                   const uint8_t* residual,
                   const float* q) {
    float score = 0.0f;
    for (int d = 0; d < index.dim; ++d) {
        uint8_t bucket = unpack_bucket(residual, d, index.nbits);
        score += q[d] * index.bucket_weights[(std::size_t)bucket];
    }
    return score;
}

void add_centroid_list_scores(const warp::WARPIndex& index,
                              int32_t centroid_id,
                              float centroid_score,
                              const float* q,
                              std::unordered_map<int32_t, float>& token_scores) {
    const std::size_t begin = (std::size_t)index.offsets[(std::size_t)centroid_id];
    const std::size_t end = (std::size_t)index.offsets[(std::size_t)centroid_id + 1u];
    int32_t prev_pid = -1;
    float prev_score = -std::numeric_limits<float>::infinity();

    auto commit = [&]() {
        if (prev_pid < 0) return;
        auto it = token_scores.find(prev_pid);
        if (it == token_scores.end()) token_scores.emplace(prev_pid, prev_score);
        else if (prev_score > it->second) it->second = prev_score;
    };

    for (std::size_t pos = begin; pos < end; ++pos) {
        int32_t pid = index.pids[pos];
        const uint8_t* residual = index.residuals.data() + pos * index.packed_dim;
        float score = centroid_score + residual_dot(index, residual, q);

        if (pid != prev_pid) {
            commit();
            prev_pid = pid;
            prev_score = score;
        } else if (score > prev_score) {
            prev_score = score;
        }
    }
    commit();
}

std::vector<int32_t> rank_warp(const warp::WARPIndex& index,
                               const float* Q,
                               const SelectedCells& selected,
                               int nprobe,
                               int k,
                               double& candidate_entries,
                               double& candidate_docs) {
    const std::size_t ntokens = selected.active_tokens.size();
    if (ntokens == 0 || k <= 0) return {};

    float total_mse = 0.0f;
    for (float x : selected.mse) total_mse += x;

    std::unordered_map<int32_t, float> doc_scores;
    doc_scores.reserve(4096);
    std::vector<std::unordered_map<int32_t, float>> token_maps(ntokens);

    for (std::size_t ti = 0; ti < ntokens; ++ti) {
        auto& token_scores = token_maps[ti];
        token_scores.reserve(2048);
        int qt = selected.active_tokens[ti];
        const float* q = Q + (std::size_t)qt * (std::size_t)index.dim;
        for (int j = 0; j < nprobe; ++j) {
            std::size_t off = ti * (std::size_t)nprobe + (std::size_t)j;
            int32_t cell = selected.cells[off];
            candidate_entries += (double)index.sizes[(std::size_t)cell];
            add_centroid_list_scores(index, cell, selected.scores[off], q, token_scores);
        }
        candidate_docs += (double)token_scores.size();
        for (const auto& item : token_scores) {
            auto inserted = doc_scores.emplace(item.first, total_mse);
            inserted.first->second += item.second - selected.mse[ti];
        }
    }

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(doc_scores.size());
    for (const auto& item : doc_scores) scored.push_back({item.second, item.first});
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

int default_t_prime(const warp::WARPIndex& index, int k) {
    if (index.n_centroids <= (1 << 16)) {
        int value = (int)(std::sqrt(8.0 * (double)index.n_tokens) / 1000.0) * 1000;
        return std::max(1, value);
    }
    return (k > 100) ? 100000 : 50000;
}

}  // namespace

int main(int argc, char** argv) {
    baseline::enforce_single_thread_searcher("warp");

    auto args = baseline::parse_args(argc, argv);
    if (args.index_dir.empty() || args.query_bundle.empty()) {
        std::cerr << "Usage: searcher_warp --index_dir <dir> --query <bundle> "
                  << "[--doc <bundle>] [--gt <dir>] [--k N] [--output file] "
                  << "[--param nprobe=32] [--param t_prime=0] [--param bound=196]\n";
        return 1;
    }

    int nprobe = baseline::get_extra_int(args, "nprobe", 32);
    int t_prime_arg = baseline::get_extra_int(args, "t_prime", 0);
    int bound = baseline::get_extra_int(args, "bound", 196);
    std::string profile = baseline::get_extra_string(
        args, "profile", "paper_lotte_nbits4_nprobe32");

    baseline::Timer load_timer;
    load_timer.start();

    warp::WARPIndex index = warp::WARPIndex::load(args.index_dir);
    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    if (query_bundle.dtype() != mv::DType::Float32) {
        throw std::runtime_error("WARP searcher expects fp32 query .mvbundle input");
    }
    if (query_bundle.dim() != index.dim) {
        throw std::runtime_error("dimension mismatch between WARP index and query bundle");
    }

    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
    }

    load_timer.stop();
    double load_sec = load_timer.elapsed_sec();

    const int k = args.k;
    const int t_prime = (t_prime_arg > 0) ? t_prime_arg : default_t_prime(index, k);
    const std::size_t n_queries = query_bundle.n_docs();
    std::cerr << "[warp-search] loaded in " << load_sec
              << "s; profile=" << profile
              << " nprobe=" << nprobe
              << " t_prime=" << t_prime
              << " bound=" << bound
              << " k=" << k
              << " nbits=" << index.nbits << "\n";

    std::vector<std::vector<int32_t>> all_topk(n_queries);
    std::vector<double> entry_counts(n_queries, 0.0);
    std::vector<double> doc_counts(n_queries, 0.0);
    std::vector<float> centroid_scores;

    baseline::Timer search_timer;
    search_timer.start();

    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        if (qi % 100 == 0) {
            std::cerr << "\r[warp-search] query " << qi << "/" << n_queries << std::flush;
        }
        auto qview = query_bundle.get(qi);
        const float* Q = qview.as_fp32();
        int n_q_tokens = (int)qview.tokens;

        compute_centroid_scores_query_major(index, Q, n_q_tokens, centroid_scores);
        SelectedCells selected = warp_select_centroids(index, Q, n_q_tokens,
                                                       centroid_scores, nprobe,
                                                       t_prime, bound);
        all_topk[qi] = rank_warp(index, Q, selected, std::min(nprobe, bound), k,
                                 entry_counts[qi], doc_counts[qi]);
    }

    search_timer.stop();
    double search_sec = search_timer.elapsed_sec();
    double per_query_ms = (search_sec * 1000.0) / (double)n_queries;
    std::cerr << "\n[warp-search] done in " << search_sec << "s ("
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

    double avg_entries = 0.0, avg_token_docs = 0.0;
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        avg_entries += entry_counts[qi];
        avg_token_docs += doc_counts[qi];
    }
    avg_entries /= (double)n_queries;
    avg_token_docs /= (double)n_queries;

    std::cout << "[warp] Recall@10: " << avg_recall_10 << "\n";
    std::cout << "[warp] Recall@100: " << avg_recall_100 << "\n";
    std::cout << "[warp] Recall@1000: " << avg_recall_1000 << "\n";
    std::cout << "[warp] MRR@10: " << avg_mrr_10 << "\n";
    std::cout << "[warp] Success@5: " << avg_success_5 << "\n";
    std::cout << "[warp] Success@100: " << avg_success_100 << "\n";
    std::cout << "[warp] Avg decompressed entries: " << avg_entries << "\n";
    std::cout << "[warp] Avg token-doc candidates: " << avg_token_docs << "\n";

    if (!args.output.empty()) {
        std::string dataset = !args.doc_bundle.empty() ? args.doc_bundle : index.input_doc_bundle;
        std::vector<std::pair<std::string, std::string>> params = {
            {"layout", warp::kLayout},
            {"compression", "compacted_bbit_residual"},
            {"metric", "ip"},
            {"profile", profile},
            {"n_centroids", std::to_string(index.n_centroids)},
            {"nbits", std::to_string(index.nbits)},
            {"nprobe", std::to_string(nprobe)},
            {"t_prime", std::to_string(t_prime)},
            {"bound", std::to_string(bound)},
            {"avg_decompressed_entries", std::to_string(avg_entries)},
            {"avg_token_doc_candidates", std::to_string(avg_token_docs)},
            {"missing_score_merge", "mse_estimate"},
            {"shared_doc_store", "none"},
        };
        baseline::write_results_json(args.output, "warp", dataset, k, params,
            avg_recall_10, avg_recall_100, avg_recall_1000,
            avg_mrr_10, avg_success_5, avg_success_100,
            index.build_time_sec, load_sec, search_sec, per_query_ms,
            warp::logical_index_bytes(args.index_dir));
    }

    return 0;
}
