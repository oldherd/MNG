// Brute-force MaxSim scan — correctness reference baseline.
// No indexer needed: directly scans all documents for each query.

#include "mvbundle.hpp"
#include "common/maxsim.hpp"
#include "common/timer.hpp"
#include "common/bench_utils.hpp"
#include "common/ranking_artifact.hpp"
#include "common/eval.hpp"
#include "common/search_runtime.hpp"
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <unordered_set>

int main(int argc, char** argv) {
    baseline::enforce_single_thread_searcher("brute");

    auto args = baseline::parse_args(argc, argv);
    if (args.doc_bundle.empty() || args.query_bundle.empty()) {
        std::cerr << "Usage: searcher_brute --doc <bundle> --query <bundle> "
                  << "[--gt <dir>] [--k N] [--output file]\n";
        return 1;
    }

    std::cerr << "[brute] Loading doc bundle: " << args.doc_bundle << "\n";
    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    std::cerr << "[brute] Docs: " << doc_bundle.n_docs()
              << ", tokens: " << doc_bundle.n_tokens()
              << ", dim: " << doc_bundle.dim() << "\n";

    std::cerr << "[brute] Loading query bundle: " << args.query_bundle << "\n";
    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    std::cerr << "[brute] Queries: " << query_bundle.n_docs() << "\n";

    const int dim = doc_bundle.dim();
    const size_t n_docs = doc_bundle.n_docs();
    const size_t n_queries = query_bundle.n_docs();
    const int k = args.k;

    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
    }

    double build_sec = 0.0;
    double load_sec = 0.0;

    // Search
    std::vector<std::vector<int32_t>> all_topk(n_queries);
    baseline::Timer search_timer;
    search_timer.start();

    for (size_t qi = 0; qi < n_queries; ++qi) {
        if (qi % 10 == 0) {
            std::cerr << "\r[brute] Query " << qi << "/" << n_queries << std::flush;
        }

        auto qview = query_bundle.get(qi);
        const float* Q = qview.as_fp32();
        int n_q_tokens = (int)qview.tokens;

        using Pair = std::pair<float, int32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

        for (size_t di = 0; di < n_docs; ++di) {
            auto dview = doc_bundle.get(di);
            const uint16_t* D = dview.as_fp16();
            int n_d_tokens = (int)dview.tokens;

            float score = baseline::maxsim_fp16(Q, n_q_tokens, D, n_d_tokens, dim);

            if ((int)heap.size() < k) {
                heap.push({score, (int32_t)di});
            } else if (score > heap.top().first) {
                heap.pop();
                heap.push({score, (int32_t)di});
            }
        }

        all_topk[qi].resize(heap.size());
        for (int i = (int)heap.size() - 1; i >= 0; --i) {
            all_topk[qi][i] = heap.top().second;
            heap.pop();
        }
    }

    search_timer.stop();
    std::cerr << "\n[brute] Search done in " << search_timer.elapsed_sec() << "s\n";

    double search_sec = search_timer.elapsed_sec();
    double per_query_ms = (search_sec * 1000.0) / n_queries;

    baseline::write_rankings_with_exact_maxsim(
        args.output, all_topk, static_cast<size_t>(k),
        args.doc_bundle, args.query_bundle);

    // Evaluate
    float avg_recall_10 = 0, avg_recall_100 = 0, avg_recall_1000 = 0;
    float avg_mrr_10 = 0, avg_success_5 = 0, avg_success_100 = 0;

    if (gt) {
        for (size_t qi = 0; qi < n_queries; ++qi) {
            const int32_t* gt_topk = gt->topk_indices_row(qi);
            int have = (int)all_topk[qi].size();

            if (have >= 10)
                avg_recall_10 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 10);
            if (have >= 100 && gt->top_k() >= 100)
                avg_recall_100 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 100);
            if (have >= 1000 && gt->top_k() >= 1000)
                avg_recall_1000 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 1000);

            // MRR@10 and Success@{5,100}: treat GT top-1 as the single relevant doc
            std::unordered_set<int32_t> qrels_set;
            qrels_set.insert(gt_topk[0]);
            if (have >= 10)
                avg_mrr_10 += baseline::mrr_at_k(all_topk[qi].data(), 10, qrels_set);
            if (have >= 5)
                avg_success_5 += baseline::success_at_k(all_topk[qi].data(), 5, qrels_set);
            if (have >= 100)
                avg_success_100 += baseline::success_at_k(all_topk[qi].data(), 100, qrels_set);
        }
        avg_recall_10 /= n_queries;
        avg_recall_100 /= n_queries;
        avg_recall_1000 /= n_queries;
        avg_mrr_10 /= n_queries;
        avg_success_5 /= n_queries;
        avg_success_100 /= n_queries;

        std::cout << "[brute] Recall@10: " << avg_recall_10 << "\n";
        std::cout << "[brute] Recall@100: " << avg_recall_100 << "\n";
        std::cout << "[brute] Recall@1000: " << avg_recall_1000 << "\n";
        std::cout << "[brute] MRR@10: " << avg_mrr_10 << "\n";
        std::cout << "[brute] Success@5: " << avg_success_5 << "\n";
        std::cout << "[brute] Success@100: " << avg_success_100 << "\n";
    }

    std::cout << "[brute] Search time: " << search_sec << "s (" << per_query_ms << " ms/query)\n";

    if (!args.output.empty()) {
        baseline::write_results_json(args.output, "brute", args.doc_bundle,
            k, args.extra,
            avg_recall_10, avg_recall_100, avg_recall_1000,
            avg_mrr_10, avg_success_5, avg_success_100,
            build_sec, load_sec, search_sec, per_query_ms, 0);
    }

    return 0;
}
