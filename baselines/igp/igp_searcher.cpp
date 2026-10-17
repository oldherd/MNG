// IGP vector-search searcher.
//
// Single-threaded implementation of the paper/upstream path:
// proximity-graph centroid probing, lower-bound candidate filtering, and
// residual-SQ decoded MaxSim refinement.

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/index_io.hpp"
#include "common/ranking_artifact.hpp"
#include "common/search_runtime.hpp"
#include "common/timer.hpp"
#include "igp_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

bool is_nonzero_token(const float* q, int dim) {
    for (int d = 0; d < dim; ++d) {
        if (q[d] != 0.0f) return true;
    }
    return false;
}

float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) sum += a[d] * b[d];
    return sum;
}

uint8_t unpack_bucket(const uint8_t* row, int dim_index, int nbits) {
    int bit_pos = dim_index * nbits;
    int byte_index = bit_pos / 8;
    int offset_in_byte = bit_pos % 8;
    int shift = 8 - offset_in_byte - nbits;
    uint8_t mask = (uint8_t)((1u << nbits) - 1u);
    return (uint8_t)((row[byte_index] >> shift) & mask);
}

float residual_dot(const igp::IGPIndex& index,
                   const uint8_t* residual,
                   const float* q) {
    float score = 0.0f;
    for (int d = 0; d < index.dim; ++d) {
        uint8_t bucket = unpack_bucket(residual, d, index.nbits);
        score += q[d] * index.bucket_weights[(std::size_t)bucket];
    }
    return score;
}

struct ProbeResult {
    std::vector<int32_t> active_tokens;
    std::vector<int32_t> centroids; // active_tokens.size() * phi_pb
    std::vector<float> scores;      // same shape
};

ProbeResult probe_query_centroids(const igp::IGPIndex& index,
                                  baseline::HnswlibIndex& pg,
                                  const float* Q,
                                  int n_q_tokens,
                                  int phi_pb) {
    ProbeResult out;
    if (phi_pb <= 0) return out;

    std::vector<int64_t> labels((std::size_t)phi_pb);
    std::vector<float> scores((std::size_t)phi_pb);
    for (int qt = 0; qt < n_q_tokens; ++qt) {
        const float* q = Q + (std::size_t)qt * (std::size_t)index.dim;
        if (!is_nonzero_token(q, index.dim)) continue;

        pg.search(q, (std::size_t)phi_pb, labels.data(), scores.data());
        out.active_tokens.push_back(qt);
        for (int j = 0; j < phi_pb; ++j) {
            out.centroids.push_back(labels[(std::size_t)j] < 0 ? -1 : (int32_t)labels[(std::size_t)j]);
            out.scores.push_back(scores[(std::size_t)j]);
        }
    }
    return out;
}

struct FilterStats {
    double seen_items = 0.0;
    double vq_score_linear_scan = 0.0;
    double refined_items = 0.0;
};

std::vector<int32_t> select_lower_bound_candidates(const igp::IGPIndex& index,
                                                   const ProbeResult& probe,
                                                   int phi_pb,
                                                   int phi_ref,
                                                   FilterStats& stats) {
    if (probe.active_tokens.empty()) return {};

    std::vector<int32_t> cand_ids;
    cand_ids.reserve(4096);
    std::vector<int32_t> pid_to_cand(index.n_docs, -1);

    const std::size_t n_active = probe.active_tokens.size();
    for (std::size_t ti = 0; ti < n_active; ++ti) {
        for (int j = 0; j < phi_pb; ++j) {
            int32_t centroid_id = probe.centroids[ti * (std::size_t)phi_pb + (std::size_t)j];
            if (centroid_id < 0) continue;
            const auto& pids = index.centroid_to_pids[(std::size_t)centroid_id];
            for (int32_t pid : pids) {
                if (pid_to_cand[(std::size_t)pid] < 0) {
                    pid_to_cand[(std::size_t)pid] = (int32_t)cand_ids.size();
                    cand_ids.push_back(pid);
                }
            }
        }
    }
    stats.seen_items = (double)cand_ids.size();
    if (cand_ids.empty()) return {};

    std::vector<float> approx(cand_ids.size(), 0.0f);
    std::vector<float> vec_max(cand_ids.size(), 0.0f);

    for (std::size_t ti = 0; ti < n_active; ++ti) {
        std::fill(vec_max.begin(), vec_max.end(), 0.0f);
        for (int j = 0; j < phi_pb; ++j) {
            std::size_t off = ti * (std::size_t)phi_pb + (std::size_t)j;
            int32_t centroid_id = probe.centroids[off];
            if (centroid_id < 0) continue;
            float score_comp = probe.scores[off] + 1.0f;
            const auto& pids = index.centroid_to_pids[(std::size_t)centroid_id];
            for (int32_t pid : pids) {
                int32_t cand = pid_to_cand[(std::size_t)pid];
                if (cand >= 0 && score_comp > vec_max[(std::size_t)cand]) {
                    vec_max[(std::size_t)cand] = score_comp;
                }
                stats.vq_score_linear_scan += 1.0;
            }
        }
        for (std::size_t i = 0; i < approx.size(); ++i) {
            approx[i] += vec_max[i];
        }
    }

    int effective_phi_ref = std::min<int>(std::max(phi_ref, 1), (int)cand_ids.size());
    std::vector<int32_t> order(cand_ids.size());
    std::iota(order.begin(), order.end(), 0);
    auto desc = [&](int32_t a, int32_t b) {
        float sa = approx[(std::size_t)a];
        float sb = approx[(std::size_t)b];
        if (sa != sb) return sa > sb;
        return cand_ids[(std::size_t)a] < cand_ids[(std::size_t)b];
    };
    if ((int)order.size() > effective_phi_ref) {
        std::nth_element(order.begin(), order.begin() + effective_phi_ref, order.end(), desc);
        order.resize((std::size_t)effective_phi_ref);
    }
    std::sort(order.begin(), order.end(), desc);

    std::vector<int32_t> selected;
    selected.reserve(order.size());
    for (int32_t cand : order) selected.push_back(cand_ids[(std::size_t)cand]);
    stats.refined_items = (double)selected.size();
    return selected;
}

float decoded_doc_maxsim(const igp::IGPIndex& index,
                         const float* Q,
                         int n_q_tokens,
                         int32_t pid) {
    const std::size_t begin = (std::size_t)index.offsets[(std::size_t)pid];
    const std::size_t end = (std::size_t)index.offsets[(std::size_t)pid + 1u];
    if (begin >= end) return -std::numeric_limits<float>::infinity();

    float total = 0.0f;
    for (int qt = 0; qt < n_q_tokens; ++qt) {
        const float* q = Q + (std::size_t)qt * (std::size_t)index.dim;
        if (!is_nonzero_token(q, index.dim)) continue;

        float best = -std::numeric_limits<float>::infinity();
        for (std::size_t token = begin; token < end; ++token) {
            int32_t centroid_id = index.assignments[token];
            const float* centroid = index.centroids.data() +
                (std::size_t)centroid_id * (std::size_t)index.dim;
            const uint8_t* residual = index.residuals.data() + token * index.packed_dim;
            float score = dot_product(q, centroid, index.dim) + residual_dot(index, residual, q);
            if (score > best) best = score;
        }
        total += best;
    }
    return total;
}

std::vector<int32_t> refine_candidates(const igp::IGPIndex& index,
                                       const float* Q,
                                       int n_q_tokens,
                                       const std::vector<int32_t>& candidates,
                                       int k) {
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(candidates.size());
    for (int32_t pid : candidates) {
        scored.push_back({decoded_doc_maxsim(index, Q, n_q_tokens, pid), pid});
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
    baseline::enforce_single_thread_searcher("igp");

    auto args = baseline::parse_args(argc, argv);
    if (args.index_dir.empty() || args.query_bundle.empty()) {
        std::cerr << "Usage: searcher_igp --index_dir <dir> --query <bundle> "
                  << "[--gt <dir>] [--k N] [--output file] "
                  << "[--param phi_pb=8] [--param phi_ref=0] [--param ef_search=0]\n";
        return 1;
    }

    int phi_pb = baseline::get_extra_int(args, "phi_pb", baseline::get_extra_int(args, "nprobe", 8));
    int phi_ref = baseline::get_extra_int(args, "phi_ref", baseline::get_extra_int(args, "probe_topk", 0));
    int ef_search = baseline::get_extra_int(args, "ef_search", 0);
    int pg_addition_search = baseline::get_extra_int(args, "pg_addition_search", 8);
    std::string profile = baseline::get_extra_string(
        args, "profile", "paper_igp_phi_pb8");

    baseline::Timer load_timer;
    load_timer.start();

    igp::IGPIndex index = igp::IGPIndex::load(args.index_dir);
    auto pg = igp::load_pg(args.index_dir, index.dim);
    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    if (query_bundle.dtype() != mv::DType::Float32) {
        throw std::runtime_error("IGP searcher expects fp32 query .mvbundle input");
    }
    if (query_bundle.dim() != index.dim) {
        throw std::runtime_error("dimension mismatch between IGP index and query bundle");
    }

    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
    }

    load_timer.stop();
    double load_sec = load_timer.elapsed_sec();

    const int k = args.k;
    if (phi_ref <= 0) phi_ref = std::max(k, 20 * k);
    phi_ref = std::max(phi_ref, k);
    phi_pb = std::max(1, std::min(phi_pb, index.n_centroids));
    if (ef_search <= 0) ef_search = std::min(index.n_centroids, phi_pb + pg_addition_search);
    ef_search = std::max(phi_pb, std::min(ef_search, index.n_centroids));
    pg->set_ef_search(ef_search);

    const std::size_t n_queries = query_bundle.n_docs();
    std::cerr << "[igp-search] loaded in " << load_sec
              << "s; profile=" << profile
              << " phi_pb=" << phi_pb
              << " phi_ref=" << phi_ref
              << " ef_search=" << ef_search
              << " k=" << k
              << " nbits=" << index.nbits << "\n";

    std::vector<std::vector<int32_t>> all_topk(n_queries);
    std::vector<FilterStats> stats(n_queries);

    baseline::Timer search_timer;
    search_timer.start();

    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        if (qi % 100 == 0) {
            std::cerr << "\r[igp-search] query " << qi << "/" << n_queries << std::flush;
        }
        auto qview = query_bundle.get(qi);
        const float* Q = qview.as_fp32();
        int n_q_tokens = (int)qview.tokens;

        ProbeResult probe = probe_query_centroids(index, *pg, Q, n_q_tokens, phi_pb);
        std::vector<int32_t> candidates =
            select_lower_bound_candidates(index, probe, phi_pb, phi_ref, stats[qi]);
        all_topk[qi] = refine_candidates(index, Q, n_q_tokens, candidates, k);
    }

    search_timer.stop();
    double search_sec = search_timer.elapsed_sec();
    double per_query_ms = (search_sec * 1000.0) / (double)n_queries;
    std::cerr << "\n[igp-search] done in " << search_sec << "s ("
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

    double avg_seen_items = 0.0;
    double avg_vq_scans = 0.0;
    double avg_refined_items = 0.0;
    for (const auto& s : stats) {
        avg_seen_items += s.seen_items;
        avg_vq_scans += s.vq_score_linear_scan;
        avg_refined_items += s.refined_items;
    }
    avg_seen_items /= (double)n_queries;
    avg_vq_scans /= (double)n_queries;
    avg_refined_items /= (double)n_queries;

    std::cout << "[igp] Recall@10: " << avg_recall_10 << "\n";
    std::cout << "[igp] Recall@100: " << avg_recall_100 << "\n";
    std::cout << "[igp] Recall@1000: " << avg_recall_1000 << "\n";
    std::cout << "[igp] MRR@10: " << avg_mrr_10 << "\n";
    std::cout << "[igp] Success@5: " << avg_success_5 << "\n";
    std::cout << "[igp] Success@100: " << avg_success_100 << "\n";
    std::cout << "[igp] Avg seen items: " << avg_seen_items << "\n";
    std::cout << "[igp] Avg VQ-score linear scans: " << avg_vq_scans << "\n";
    std::cout << "[igp] Avg refined items: " << avg_refined_items << "\n";

    if (!args.output.empty()) {
        std::string dataset = !args.doc_bundle.empty() ? args.doc_bundle : index.input_doc_bundle;
        std::vector<std::pair<std::string, std::string>> params = {
            {"layout", igp::kLayout},
            {"compression", "residual_scalar_quantization"},
            {"metric", "ip"},
            {"profile", profile},
            {"n_centroids", std::to_string(index.n_centroids)},
            {"nbits", std::to_string(index.nbits)},
            {"phi_pb", std::to_string(phi_pb)},
            {"nprobe", std::to_string(phi_pb)},
            {"phi_ref", std::to_string(phi_ref)},
            {"probe_topk", std::to_string(phi_ref)},
            {"ef_search", std::to_string(ef_search)},
            {"pg_addition_search", std::to_string(pg_addition_search)},
            {"pg_backend", "hnswlib_inner_product"},
            {"pg_probe_mode", "topk_with_ef_phi_pb_plus_addition"},
            {"lower_bound_shift", "centroid_score_plus_1"},
            {"avg_seen_items", std::to_string(avg_seen_items)},
            {"avg_vq_score_linear_scans", std::to_string(avg_vq_scans)},
            {"avg_refined_items", std::to_string(avg_refined_items)},
            {"final_scoring", "residual_decompressed_maxsim"},
            {"shared_doc_store", "none"},
        };
        baseline::write_results_json(args.output, "igp", dataset, k, params,
            avg_recall_10, avg_recall_100, avg_recall_1000,
            avg_mrr_10, avg_success_5, avg_success_100,
            index.build_time_sec, load_sec, search_sec, per_query_ms,
            igp::logical_index_bytes(args.index_dir));
    }

    return 0;
}
