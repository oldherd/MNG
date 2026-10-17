#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/maxsim.hpp"
#include "common/ranking_artifact.hpp"
#include "common/search_runtime.hpp"
#include "common/timer.hpp"
#include "muvera_diskann_fde.hpp"
#include "utils/beamSearch.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
    baseline::enforce_single_thread_searcher("muvera_diskann");
    if (parlay::num_workers() != 1) {
        std::cerr << "[muvera-diskann-searcher] ERROR: Parlay search must use exactly "
                  << "one worker; observed " << parlay::num_workers() << ".\n";
        return 2;
    }

    auto args = baseline::parse_args(argc, argv);
    if (args.index_dir.empty() || args.query_bundle.empty() || args.doc_bundle.empty()) {
        std::cerr << "Usage: searcher_muvera_diskann --index_dir <dir> --doc <bundle> "
                  << "--query <bundle> [--gt <dir>] [--k N] [--output file] "
                  << "[--param beam_width=1000] [--param ball_carve_tau=0.7] "
                  << "[--param max_queries=0]\n";
        return 1;
    }

    const int beam_width = baseline::get_extra_int(args, "beam_width", 1000);
    const float ball_carve_tau =
        baseline::get_extra_float(args, "ball_carve_tau", 0.7f);
    const int max_queries = baseline::get_extra_int(args, "max_queries", 0);
    if (args.k <= 0 || beam_width < args.k || ball_carve_tau < -1.0f ||
        ball_carve_tau > 1.0f || max_queries < 0) {
        std::cerr << "[muvera-diskann-searcher] ERROR: require k>0, beam_width>=k, "
                  << "tau in [-1,1], and max_queries>=0.\n";
        return 1;
    }

    baseline::Timer load_timer;
    load_timer.start();
    auto idx = muvera_diskann::Index::load(args.index_dir);
    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    if (doc_bundle.n_docs() < idx.n_docs || doc_bundle.dim() != idx.dim ||
        query_bundle.dim() != idx.dim) {
        std::cerr << "[muvera-diskann-searcher] ERROR: bundle/index shape mismatch.\n";
        return 1;
    }
    const size_t n_queries = max_queries > 0
        ? std::min(query_bundle.n_docs(), static_cast<size_t>(max_queries))
        : query_bundle.n_docs();

    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
        if (gt->n_queries() < n_queries) {
            std::cerr << "[muvera-diskann-searcher] ERROR: ground truth has too few queries.\n";
            return 1;
        }
    }
    load_timer.stop();

    const int actual_beam = std::min<int>(beam_width, static_cast<int>(idx.n_docs));
    muvera_diskann::PQPointRange pq_points{
        idx.pq_codes.data(), idx.n_docs, idx.pq_m, idx.pq_ksub};
    parlayANN::QueryParams query_params(
        args.k, actual_beam, static_cast<long>(idx.n_docs), idx.graph_R);
    const parlay::sequence<uint32_t> start_points = {0};

    std::cerr << "[muvera-diskann-searcher] runtime: Parlay workers="
              << parlay::num_workers() << ", OpenMP/env search threads=1\n";
    std::cerr << "[muvera-diskann-searcher] Params: k=" << args.k
              << " beam_width=" << actual_beam
              << " ball_carve_tau=" << ball_carve_tau
              << " graph_R=" << idx.graph_R << " graph_L=" << idx.graph_L
              << " pq=256x8_asymmetric\n";

    std::vector<std::vector<int32_t>> all_topk(n_queries);
    std::vector<float> query_fde(idx.d_fde);
    std::vector<float> pq_lut;
    baseline::Timer search_timer;
    search_timer.start();

    for (size_t qi = 0; qi < n_queries; ++qi) {
        auto query_view = query_bundle.get(qi);
        const float* query = query_view.as_fp32();
        const int n_query_tokens = static_cast<int>(query_view.tokens);
        if (n_query_tokens <= 0) continue;

        muvera_diskann::compute_query_fde(query, n_query_tokens, idx, query_fde.data());
        muvera_diskann::build_pq_lut(idx, query_fde.data(), pq_lut);
        const muvera_diskann::PQPoint query_point{
            nullptr, pq_lut.data(), idx.pq_m, idx.pq_ksub, -1};
        auto search_result = parlayANN::filtered_beam_search(
            *idx.graph, query_point, pq_points, query_point, pq_points,
            start_points, query_params, false);
        const auto& candidates = search_result.first.first;

        auto carved = muvera_diskann::ball_carve(
            query, n_query_tokens, idx.dim, ball_carve_tau);
        const int carved_tokens = static_cast<int>(carved.size() / idx.dim);
        using ScoredDoc = std::pair<float, int32_t>;
        std::priority_queue<ScoredDoc, std::vector<ScoredDoc>,
                            std::greater<ScoredDoc>> heap;
        for (const auto& candidate : candidates) {
            const int32_t doc_id = static_cast<int32_t>(candidate.first);
            auto doc_view = doc_bundle.get(static_cast<size_t>(doc_id));
            const float score = baseline::maxsim_fp16(
                carved.data(), carved_tokens, doc_view.as_fp16(),
                static_cast<int>(doc_view.tokens), idx.dim);
            if (static_cast<int>(heap.size()) < args.k) {
                heap.push({score, doc_id});
            } else if (score > heap.top().first) {
                heap.pop();
                heap.push({score, doc_id});
            }
        }
        all_topk[qi].resize(heap.size());
        for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i) {
            all_topk[qi][i] = heap.top().second;
            heap.pop();
        }
        if (qi % 100 == 0) {
            std::cerr << "\r[muvera-diskann-searcher] Query " << qi << "/"
                      << n_queries << std::flush;
        }
    }
    search_timer.stop();
    std::cerr << "\n";

    baseline::write_rankings_with_exact_maxsim(
        args.output, all_topk, static_cast<size_t>(args.k),
        args.doc_bundle, args.query_bundle);

    float recall_10 = 0.0f;
    float recall_100 = 0.0f;
    float recall_1000 = 0.0f;
    float mrr_10 = 0.0f;
    float success_5 = 0.0f;
    float success_100 = 0.0f;
    if (gt && n_queries > 0) {
        for (size_t qi = 0; qi < n_queries; ++qi) {
            const int32_t* truth = gt->topk_indices_row(qi);
            const int have = static_cast<int>(all_topk[qi].size());
            if (have >= 10 && gt->top_k() >= 10) {
                recall_10 += baseline::recall_at_k(all_topk[qi].data(), truth, 10);
            }
            if (have >= 100 && gt->top_k() >= 100) {
                recall_100 += baseline::recall_at_k(all_topk[qi].data(), truth, 100);
            }
            if (have >= 1000 && gt->top_k() >= 1000) {
                recall_1000 += baseline::recall_at_k(all_topk[qi].data(), truth, 1000);
            }
            std::unordered_set<int32_t> relevant = {truth[0]};
            if (have >= 10) mrr_10 += baseline::mrr_at_k(all_topk[qi].data(), 10, relevant);
            if (have >= 5) success_5 += baseline::success_at_k(all_topk[qi].data(), 5, relevant);
            if (have >= 100) {
                success_100 += baseline::success_at_k(all_topk[qi].data(), 100, relevant);
            }
        }
        recall_10 /= n_queries;
        recall_100 /= n_queries;
        recall_1000 /= n_queries;
        mrr_10 /= n_queries;
        success_5 /= n_queries;
        success_100 /= n_queries;
    }

    const double search_sec = search_timer.elapsed_sec();
    const double per_query_ms = n_queries > 0 ? search_sec * 1000.0 / n_queries : 0.0;
    std::cout << "[muvera_diskann] Recall@10: " << recall_10 << "\n";
    std::cout << "[muvera_diskann] Recall@100: " << recall_100 << "\n";
    std::cout << "[muvera_diskann] Recall@1000: " << recall_1000 << "\n";
    std::cout << "[muvera_diskann] Search time: " << search_sec << "s ("
              << per_query_ms << " ms/query)\n";

    if (!args.output.empty()) {
        baseline::write_results_json(
            args.output, "muvera_diskann", args.doc_bundle, args.k,
            {{"formal_baseline", "pending_validation"},
             {"ann_backend", "parlayann_vamana_diskann"},
             {"diskann_R", std::to_string(idx.graph_R)},
             {"diskann_L", std::to_string(idx.graph_L)},
             {"beam_width", std::to_string(actual_beam)},
             {"ball_carve_tau", std::to_string(ball_carve_tau)},
             {"pq", "PQ-256-8_asymmetric_graph_search"},
             {"parlay_workers", std::to_string(parlay::num_workers())},
             {"index_build_scope", idx.build_scope}},
            recall_10, recall_100, recall_1000, mrr_10, success_5, success_100,
            idx.graph_build_sec, load_timer.elapsed_sec(), search_sec, per_query_ms,
            baseline::directory_file_size(args.index_dir));
    }
    return 0;
}
