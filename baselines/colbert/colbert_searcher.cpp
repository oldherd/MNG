// ColBERT vector-search searcher: FAISS IVFPQ token retrieval + exact MaxSim refinement.

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/index_io.hpp"
#include "common/maxsim.hpp"
#include "common/ranking_artifact.hpp"
#include "common/search_runtime.hpp"
#include "common/timer.hpp"
#include "colbert_index.hpp"

#include <faiss/IndexIVF.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>

namespace {

int32_t token_to_doc(const int64_t* offsets, std::size_t n_docs, faiss::idx_t token_id) {
    const int64_t token = (int64_t)token_id;
    const int64_t* begin = offsets;
    const int64_t* end = offsets + n_docs + 1;
    const int64_t* it = std::upper_bound(begin, end, token);
    if (it == begin) return -1;
    std::size_t doc = (std::size_t)((it - begin) - 1);
    if (doc >= n_docs || token < offsets[doc] || token >= offsets[doc + 1]) return -1;
    return (int32_t)doc;
}

}  // namespace

int main(int argc, char** argv) {
    baseline::enforce_single_thread_searcher("colbert");

    auto args = baseline::parse_args(argc, argv);
    if (args.index_dir.empty() || args.doc_bundle.empty() || args.query_bundle.empty()) {
        std::cerr << "Usage: searcher_colbert --index_dir <dir> --doc <bundle> --query <bundle> "
                  << "[--gt <dir>] [--k N] [--output file] [--param nprobe=10] [--param k_prime=1000]\n";
        return 1;
    }

    int nprobe = baseline::get_extra_int(args, "nprobe", 10);
    int k_prime = baseline::get_extra_int(args, "k_prime", 1000);

    baseline::Timer load_timer;
    load_timer.start();

    auto meta = baseline::read_meta(args.index_dir);
    if (meta["layout"] != colbert::kLayout) {
        throw std::runtime_error("colbert index layout mismatch: expected " +
                                 std::string(colbert::kLayout) + ", got " + meta["layout"]);
    }
    double build_sec = std::stod(meta["build_time_sec"]);

    auto index = colbert::read_faiss_index(args.index_dir);
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
    if (!ivf) throw std::runtime_error("FAISS index is not IndexIVF");
    ivf->nprobe = nprobe;

    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    const std::size_t n_docs = doc_bundle.n_docs();
    const std::size_t n_queries = query_bundle.n_docs();
    const int dim = doc_bundle.dim();
    if (query_bundle.dim() != dim || index->d != dim) {
        throw std::runtime_error("dimension mismatch between index/doc/query");
    }
    const int64_t* offsets = doc_bundle.offsets();

    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
    }

    load_timer.stop();
    double load_sec = load_timer.elapsed_sec();

    const int k = args.k;
    std::cerr << "[colbert-search] loaded in " << load_sec << "s; nprobe=" << nprobe
              << " k_prime=" << k_prime << " metric=" << meta["metric"] << "\n";

    std::vector<std::vector<int32_t>> all_topk(n_queries);
    std::vector<double> candidate_counts(n_queries, 0.0);
    baseline::Timer search_timer;
    search_timer.start();

    std::vector<float> distances(k_prime);
    std::vector<faiss::idx_t> labels(k_prime);

    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        if (qi % 100 == 0) {
            std::cerr << "\r[colbert-search] query " << qi << "/" << n_queries << std::flush;
        }

        auto qview = query_bundle.get(qi);
        const float* Q = qview.as_fp32();
        int n_q_tokens = (int)qview.tokens;

        std::unordered_set<int32_t> candidate_docs;
        for (int qt = 0; qt < n_q_tokens; ++qt) {
            const float* q = Q + (std::size_t)qt * dim;
            index->search(1, q, k_prime, distances.data(), labels.data());
            for (int i = 0; i < k_prime; ++i) {
                if (labels[i] < 0) continue;
                int32_t doc_id = token_to_doc(offsets, n_docs, labels[i]);
                if (doc_id >= 0) candidate_docs.insert(doc_id);
            }
        }
        candidate_counts[qi] = (double)candidate_docs.size();

        using Pair = std::pair<float, int32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;
        for (int32_t doc_id : candidate_docs) {
            auto dview = doc_bundle.get((std::size_t)doc_id);
            float score = baseline::maxsim_fp16(Q, n_q_tokens, dview.as_fp16(), (int)dview.tokens, dim);
            if ((int)heap.size() < k) {
                heap.push({score, doc_id});
            } else if (score > heap.top().first) {
                heap.pop();
                heap.push({score, doc_id});
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
    double per_query_ms = (search_sec * 1000.0) / (double)n_queries;
    std::cerr << "\n[colbert-search] done in " << search_sec << "s (" << per_query_ms << " ms/q)\n";

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

    double avg_candidates = 0.0;
    for (double x : candidate_counts) avg_candidates += x;
    avg_candidates /= (double)n_queries;

    std::cout << "[colbert] Recall@10: " << avg_recall_10 << "\n";
    std::cout << "[colbert] Recall@100: " << avg_recall_100 << "\n";
    std::cout << "[colbert] Recall@1000: " << avg_recall_1000 << "\n";
    std::cout << "[colbert] MRR@10: " << avg_mrr_10 << "\n";
    std::cout << "[colbert] Success@5: " << avg_success_5 << "\n";
    std::cout << "[colbert] Success@100: " << avg_success_100 << "\n";
    std::cout << "[colbert] Avg candidates: " << avg_candidates << "\n";

    if (!args.output.empty()) {
        std::vector<std::pair<std::string, std::string>> params = {
            {"layout", meta["layout"]},
            {"ann_backend", meta["ann_backend"]},
            {"faiss_index", meta["faiss_index"]},
            {"metric", meta["metric"]},
            {"nlist", meta["nlist"]},
            {"pq_m", meta["pq_m"]},
            {"nbits", meta["nbits"]},
            {"nprobe", std::to_string(nprobe)},
            {"k_prime", std::to_string(k_prime)},
            {"avg_candidates", std::to_string(avg_candidates)},
            {"exact_refine", "true"},
            {"shared_doc_store", "mvbundle"},
        };
        baseline::write_results_json(args.output, "colbert", args.doc_bundle, k, params,
            avg_recall_10, avg_recall_100, avg_recall_1000,
            avg_mrr_10, avg_success_5, avg_success_100,
            build_sec, load_sec, search_sec, per_query_ms,
            colbert::logical_index_bytes(args.index_dir));
    }

    return 0;
}
