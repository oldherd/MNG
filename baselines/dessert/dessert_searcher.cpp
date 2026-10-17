// DESSERT searcher — IVF pre-filter + per-document TinyTable sketch scoring.

#include "mvbundle.hpp"
#include "common/fp16_utils.hpp"
#include "common/kmeans.hpp"
#include "common/timer.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/index_io.hpp"
#include "common/ranking_artifact.hpp"
#include "common/search_runtime.hpp"
#include "dessert_index.hpp"
#include "dessert_tinytable.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
    baseline::enforce_single_thread_searcher("dessert");

    auto args = baseline::parse_args(argc, argv);
    if (args.index_dir.empty() || args.doc_bundle.empty() || args.query_bundle.empty()) {
        std::cerr << "Usage: searcher_dessert --index_dir <dir> --doc <bundle> --query <bundle> "
                  << "[--gt <dir>] [--k N] [--output file] [--param filter_k=4096] [--param filter_probe=1]\n";
        return 1;
    }

    int filter_k     = baseline::get_extra_int(args, "filter_k", 4096);
    int filter_probe = baseline::get_extra_int(args, "filter_probe", 1);

    baseline::Timer load_timer;
    load_timer.start();

    auto idx = dessert::DESSERTIndex::load(args.index_dir);
    auto meta = baseline::read_meta(args.index_dir);
    double build_sec = std::stod(meta["build_time_sec"]);

    dessert::SparseRandomProjection srp;
    srp.dim = idx.dim;
    srp.hashes_per_table = idx.C;
    srp.num_tables = idx.L;
    srp.sample_size = idx.srp_sample_size;
    srp.random_bits = idx.srp_random_bits;
    srp.hash_indices = idx.srp_hash_indices;

    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    const size_t n_queries = query_bundle.n_docs();
    const int dim = idx.dim;
    const int L = idx.L;
    const int C = idx.C;
    const int n_filter = idx.n_filter;
    const uint32_t range = 1u << C;

    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
    }

    load_timer.stop();
    double load_sec = load_timer.elapsed_sec();
    std::cerr << "[dessert-search] Loaded in " << load_sec << "s\n";
    std::cerr << "[dessert-search] filter_k=" << filter_k << " filter_probe=" << filter_probe << "\n";

    auto sim_lookup = dessert::make_collision_lookup((uint32_t)L, (uint32_t)C);
    std::vector<int32_t> prefilter_counts(idx.n_docs, 0);
    std::vector<int32_t> touched_docs;

    const int k = args.k;
    std::vector<std::vector<int32_t>> all_topk(n_queries);
    baseline::Timer search_timer;
    search_timer.start();

    for (size_t qi = 0; qi < n_queries; ++qi) {
        if (qi % 100 == 0) {
            std::cerr << "\r[dessert-search] Query " << qi << "/" << n_queries << std::flush;
        }

        auto qview = query_bundle.get(qi);
        const float* Q = qview.as_fp32();
        int n_q_tokens = (int)qview.tokens;

        // Phase 1: DESSERT centroid pre-filter. Official DocSearch concatenates
        // query token nearest centroids and removes duplicates before counting
        // document bucket frequency.
        std::vector<int32_t> query_centroids;
        query_centroids.reserve((size_t)n_q_tokens * filter_probe);
        for (int qt = 0; qt < n_q_tokens; ++qt) {
            const float* q = Q + (size_t)qt * dim;
            auto top_c = baseline::KMeans::topk_centroids(
                q, idx.filter_centroids.data(), n_filter, dim, filter_probe);
            for (int c : top_c) query_centroids.push_back(c);
        }
        std::sort(query_centroids.begin(), query_centroids.end());
        query_centroids.erase(std::unique(query_centroids.begin(), query_centroids.end()),
                              query_centroids.end());

        touched_docs.clear();
        for (int32_t centroid_id : query_centroids) {
            for (int32_t doc_id : idx.ivf_lists[centroid_id]) {
                if (prefilter_counts[doc_id] == 0) touched_docs.push_back(doc_id);
                prefilter_counts[doc_id] += 1;
            }
        }

        using CountPair = std::pair<int32_t, int32_t>;
        std::priority_queue<CountPair, std::vector<CountPair>, std::greater<CountPair>> pre_heap;
        for (int32_t doc_id : touched_docs) {
            int32_t count = prefilter_counts[doc_id];
            prefilter_counts[doc_id] = 0;
            if ((int)pre_heap.size() < filter_k) {
                pre_heap.push({count, doc_id});
            } else if (count > pre_heap.top().first) {
                pre_heap.pop();
                pre_heap.push({count, doc_id});
            }
        }

        std::vector<int32_t> candidates(pre_heap.size());
        for (int i = (int)pre_heap.size() - 1; i >= 0; --i) {
            candidates[i] = pre_heap.top().second;
            pre_heap.pop();
        }

        // Phase 2: TinyTable scoring. This mirrors MaxFlashArray::getDocumentScores:
        // hash query vectors, score each candidate by collision-count estimated
        // MaxSim, and normalize by query length.
        auto query_hashes = dessert::hash_vectors_serial(srp, Q, n_q_tokens, dim);

        using ScorePair = std::pair<float, int32_t>;
        std::priority_queue<ScorePair, std::vector<ScorePair>, std::greater<ScorePair>> heap;
        std::vector<uint32_t> count_buffer;

        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            int32_t doc_id = candidates[ci];
            uint32_t doc_tokens = idx.doc_n_tokens[doc_id];
            const uint16_t* table_data =
                idx.doc_table_data.data() + idx.doc_table_offsets[doc_id];
            float score = dessert::score_tiny_table(
                table_data, doc_tokens, query_hashes, (uint32_t)n_q_tokens,
                (uint32_t)L, range, count_buffer, sim_lookup);

            ScorePair item{score, doc_id};
            if ((int)heap.size() < k) {
                heap.push(item);
            } else if (item.first > heap.top().first) {
                heap.pop();
                heap.push(item);
            }
        }

        all_topk[qi].resize(heap.size());
        for (int i = (int)heap.size() - 1; i >= 0; --i) {
            all_topk[qi][i] = heap.top().second;
            heap.pop();
        }
    }

    search_timer.stop();
    double search_sec = search_timer.elapsed_sec();
    double per_query_ms = (search_sec * 1000.0) / n_queries;
    std::cerr << "\n[dessert-search] Done in " << search_sec << "s\n";

    baseline::write_rankings_with_exact_maxsim(
        args.output, all_topk, static_cast<size_t>(k),
        args.doc_bundle, args.query_bundle);

    float avg_recall_10 = 0, avg_recall_100 = 0, avg_recall_1000 = 0;
    float avg_mrr_10 = 0, avg_success_5 = 0, avg_success_100 = 0;
    if (gt) {
        for (size_t qi = 0; qi < n_queries; ++qi) {
            const int32_t* gt_topk = gt->topk_indices_row(qi);
            int have = (int)all_topk[qi].size();
            if (have >= 10) {
                avg_recall_10 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 10);
            }
            if (have >= 100 && gt->top_k() >= 100) {
                avg_recall_100 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 100);
            }
            if (have >= 1000 && gt->top_k() >= 1000) {
                avg_recall_1000 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 1000);
            }
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

        std::cout << "[dessert] Recall@10: " << avg_recall_10 << "\n";
        std::cout << "[dessert] Recall@100: " << avg_recall_100 << "\n";
        std::cout << "[dessert] Recall@1000: " << avg_recall_1000 << "\n";
        std::cout << "[dessert] MRR@10: " << avg_mrr_10 << "\n";
        std::cout << "[dessert] Success@5: " << avg_success_5 << "\n";
        std::cout << "[dessert] Success@100: " << avg_success_100 << "\n";
    }

    std::cout << "[dessert] Build: " << build_sec << "s | Load: " << load_sec
              << "s | Search: " << search_sec << "s (" << per_query_ms << " ms/q)\n";

    if (!args.output.empty()) {
        std::vector<std::pair<std::string, std::string>> params = {
            {"L", std::to_string(L)},
            {"C", std::to_string(C)},
            {"n_filter", std::to_string(n_filter)},
            {"filter_k", std::to_string(filter_k)},
            {"filter_probe", std::to_string(filter_probe)},
            {"layout", "tinytables"}
        };
        size_t index_bytes = baseline::directory_file_size(args.index_dir);

        baseline::write_results_json(
            args.output, "dessert", args.doc_bundle, k, params,
            avg_recall_10, avg_recall_100, avg_recall_1000,
            avg_mrr_10, avg_success_5, avg_success_100,
            build_sec, load_sec, search_sec, per_query_ms, index_bytes);
    }

    return 0;
}
