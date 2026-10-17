#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/ranking_artifact.hpp"
#include "common/search_runtime.hpp"
#include "common/timer.hpp"
#include "official_gem/official_gem_common.hpp"
#include "official_gem/official_gem_index.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct OfficialRuntime {
    official_gem::OfficialGEMIndex index;
    official_gem::OfficialBaseMemory base_memory;
    std::vector<std::vector<hnswlib::labeltype>> cluster_set;
    std::vector<int32_t> cluster_entries;
    std::unique_ptr<hnswlib::L2VSSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
};

OfficialRuntime load_runtime(const std::string& index_dir, const mv::MVBundle& doc_bundle) {
    OfficialRuntime rt;
    rt.index = official_gem::OfficialGEMIndex::load(index_dir);
    official_gem::set_official_cluster_count(rt.index.n_cquant);
    rt.base_memory = official_gem::make_official_base_memory(doc_bundle, rt.index);
    rt.cluster_set = official_gem::make_cluster_sets(rt.index);
    rt.cluster_entries.assign((std::size_t)rt.index.n_cindex, -1);
    rt.space = std::make_unique<hnswlib::L2VSSpace>((size_t)rt.index.dim);
    rt.alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        rt.space.get(),
        rt.index.n_docs + 1u,
        (size_t)rt.index.M,
        (size_t)rt.index.ef_construction);
    rt.alg->loadIndex(rt.index.hnsw_path(index_dir), rt.space.get());

    std::vector<uint8_t> loaded(rt.index.n_docs, 0);
    for (int cluster = 0; cluster < rt.index.n_cindex; ++cluster) {
        const auto& docs = rt.cluster_set[(std::size_t)cluster];
        if (docs.empty()) continue;
        rt.cluster_entries[(std::size_t)cluster] = (int32_t)docs[0];
        for (hnswlib::labeltype doc : docs) {
            if (doc < 0 || (std::size_t)doc >= rt.index.n_docs) continue;
            if (!loaded[(std::size_t)doc]) {
                rt.alg->loadDataAddress(&rt.base_memory.base[(std::size_t)doc], doc);
                loaded[(std::size_t)doc] = 1;
            }
        }
    }
    rt.alg->search_set.resize(rt.alg->max_elements_);
    return rt;
}

std::vector<float> active_query_tokens(const mv::MVBundle& query_bundle,
                                       std::size_t qi,
                                       int dim) {
    auto qview = query_bundle.get(qi);
    const float* Q = qview.as_fp32();
    std::vector<float> tokens;
    tokens.reserve((std::size_t)qview.tokens * (std::size_t)dim);
    for (std::size_t qt = 0; qt < qview.tokens; ++qt) {
        const float* src = Q + qt * (std::size_t)dim;
        if (!official_gem::is_nonzero_token(src, dim)) continue;
        std::size_t old = tokens.size();
        tokens.resize(old + (std::size_t)dim);
        std::copy(src, src + dim, tokens.data() + old);
        official_gem::l2_normalize(tokens.data() + old, dim);
    }
    return tokens;
}

std::vector<int32_t> official_candidates(OfficialRuntime& rt,
                                         const std::vector<float>& query_tokens,
                                         int nprobe,
                                         int ef_search,
                                         int rerankK) {
    const int active = (int)(query_tokens.size() / (std::size_t)rt.index.dim);
    if (active <= 0) return {};

    std::vector<float> graph_cluster_scores((std::size_t)active *
                                            (std::size_t)rt.index.n_cindex);
    hnswlib::fast_dot_product_blas(active,
                                   rt.index.dim,
                                   rt.index.n_cindex,
                                   const_cast<float*>(query_tokens.data()),
                                   rt.index.cindex_centroids.data(),
                                   graph_cluster_scores.data());
    std::vector<int32_t> clusters = official_gem::top_unique_clusters(
        graph_cluster_scores, active, rt.index.n_cindex, nprobe);

    std::vector<hnswlib::labeltype> entry_points;
    rt.alg->search_set.assign(rt.alg->search_set.size(), false);
    for (int32_t cluster : clusters) {
        if (cluster < 0 || cluster >= rt.index.n_cindex) continue;
        int32_t entry = rt.cluster_entries[(std::size_t)cluster];
        if (entry < 0) continue;
        entry_points.push_back((hnswlib::labeltype)entry);
        for (hnswlib::labeltype doc : rt.cluster_set[(std::size_t)cluster]) {
            auto it = rt.alg->label_lookup_.find(doc);
            if (it != rt.alg->label_lookup_.end()) {
                rt.alg->search_set[it->second] = true;
            }
        }
    }
    if (entry_points.empty()) return {};

    std::vector<float> query_cluster_scores((std::size_t)rt.index.n_cquant *
                                            (std::size_t)active);
    hnswlib::fast_dot_product_blas(rt.index.n_cquant,
                                   rt.index.dim,
                                   active,
                                   rt.index.cquant_centroids.data(),
                                   const_cast<float*>(query_tokens.data()),
                                   query_cluster_scores.data());
    vectorset query_cluster(query_cluster_scores.data(), nullptr, rt.index.n_cquant, active);

    rt.alg->setEf((size_t)ef_search);
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
        rt.alg->searchKnnClusterEntries(&query_cluster, (size_t)ef_search, entry_points);

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(result.size());
    while (!result.empty()) {
        scored.push_back({result.top().first, (int32_t)result.top().second});
        result.pop();
    }
    auto asc = [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    };
    const int take = std::min<int>(rerankK, (int)scored.size());
    if ((int)scored.size() > take) {
        std::partial_sort(scored.begin(), scored.begin() + take, scored.end(), asc);
        scored.resize((std::size_t)take);
    } else {
        std::sort(scored.begin(), scored.end(), asc);
    }

    std::vector<int32_t> out;
    out.reserve(scored.size());
    for (const auto& item : scored) out.push_back(item.second);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    baseline::enforce_single_thread_searcher("gem_official_hnsw_adapter");

    auto args = baseline::parse_args(argc, argv);
    if (args.doc_bundle.empty() || args.query_bundle.empty() || args.index_dir.empty()) {
        std::cerr << "Usage: searcher_gem_official --doc <doc_bundle> --query <query_bundle> "
                  << "--index_dir <dir> [--gt <gt_dir>] [--k 100] "
                  << "[--param nprobe=4] [--param ef_search=4000] "
                  << "[--param rerankK=512]\n";
        return 1;
    }

    const int k = args.k;
    const int ef_search = baseline::get_extra_int(args, "ef_search", 4000);
    const int rerankK = baseline::get_extra_int(args, "rerankK", 512);
    if (k <= 0 || ef_search <= 0 || rerankK <= 0) {
        std::cerr << "[gem-official-searcher] ERROR: k, ef_search, and rerankK must be positive.\n";
        return 1;
    }

    baseline::Timer load_timer;
    load_timer.start();
    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    OfficialRuntime rt = load_runtime(args.index_dir, doc_bundle);
    const int nprobe = baseline::get_extra_int(args, "nprobe", rt.index.nprobe);
    if (nprobe <= 0) {
        std::cerr << "[gem-official-searcher] ERROR: nprobe must be positive.\n";
        return 1;
    }

    auto query_bundle = mv::MVBundle::load(args.query_bundle);
    std::unique_ptr<mv::MVGroundTruth> gt;
    if (!args.gt_dir.empty()) {
        gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(args.gt_dir));
    }
    if (doc_bundle.dim() != rt.index.dim || query_bundle.dim() != rt.index.dim) {
        throw std::runtime_error("official GEM dimension mismatch between index/doc/query bundles");
    }
    if (doc_bundle.dtype() != mv::DType::Float16) {
        throw std::runtime_error("official GEM searcher expects fp16 doc .mvbundle input");
    }
    if (query_bundle.dtype() != mv::DType::Float32) {
        throw std::runtime_error("official GEM searcher expects fp32 query .mvbundle input");
    }
    load_timer.stop();

    const std::size_t n_queries = query_bundle.n_docs();
    std::vector<std::vector<int32_t>> all_topk(n_queries);

    baseline::Timer search_timer;
    search_timer.start();
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        std::vector<float> qtokens = active_query_tokens(query_bundle, qi, rt.index.dim);
        std::vector<int32_t> candidates =
            official_candidates(rt, qtokens, nprobe, ef_search, rerankK);
        all_topk[qi] = official_gem::rerank_raw_maxsim(
            doc_bundle,
            qtokens.data(),
            (int)(qtokens.size() / (std::size_t)rt.index.dim),
            candidates,
            k);
        if (qi % 100 == 0) {
            std::cerr << "\r[gem-official-searcher]   query " << qi << "/"
                      << n_queries << std::flush;
        }
    }
    std::cerr << "\n";
    search_timer.stop();

    const double load_sec = load_timer.elapsed_sec();
    const double search_sec = search_timer.elapsed_sec();
    const double per_query_ms = n_queries > 0 ? (search_sec * 1000.0) / (double)n_queries : 0.0;

    baseline::write_rankings_with_exact_maxsim(
        args.output, all_topk, static_cast<size_t>(args.k),
        args.doc_bundle, args.query_bundle);

    float avg_recall_10 = 0.0f;
    float avg_recall_100 = 0.0f;
    float avg_recall_1000 = 0.0f;
    float avg_mrr_10 = 0.0f;
    float avg_success_5 = 0.0f;
    float avg_success_100 = 0.0f;

    if (gt) {
        for (std::size_t qi = 0; qi < n_queries; ++qi) {
            const int32_t* gt_topk = gt->topk_indices_row(qi);
            const int have = (int)all_topk[qi].size();
            if (have >= 10) avg_recall_10 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 10);
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
            if (have >= 100) {
                avg_success_100 += baseline::success_at_k(all_topk[qi].data(), 100, qrels_set);
            }
        }
        avg_recall_10 /= (float)n_queries;
        avg_recall_100 /= (float)n_queries;
        avg_recall_1000 /= (float)n_queries;
        avg_mrr_10 /= (float)n_queries;
        avg_success_5 /= (float)n_queries;
        avg_success_100 /= (float)n_queries;

        std::cout << "[gem-official] Recall@10: " << avg_recall_10 << "\n";
        std::cout << "[gem-official] Recall@100: " << avg_recall_100 << "\n";
        std::cout << "[gem-official] Recall@1000: " << avg_recall_1000 << "\n";
        std::cout << "[gem-official] MRR@10: " << avg_mrr_10 << "\n";
        std::cout << "[gem-official] Success@5: " << avg_success_5 << "\n";
        std::cout << "[gem-official] Success@100: " << avg_success_100 << "\n";
    }

    std::cout << "[gem-official] Build time: " << rt.index.build_time_sec << "s\n";
    std::cout << "[gem-official] Load time: " << load_sec << "s\n";
    std::cout << "[gem-official] Search time: " << search_sec
              << "s (" << per_query_ms << " ms/query)\n";

    if (!args.output.empty()) {
        std::vector<std::pair<std::string, std::string>> params = {
            {"formal_baseline", "false"},
            {"official_upstream_component", "sigmod26gem/hnswlib"},
            {"mvbundle_adapter", "true"},
            {"graph_build_metric", "official_cluster_emd"},
            {"search_metric", "official_qch_cluster_search"},
            {"final_scoring", "raw_mvbundle_exact_maxsim"},
            {"shared_doc_store", "mvbundle"},
            {"n_cquant", std::to_string(rt.index.n_cquant)},
            {"n_cindex", std::to_string(rt.index.n_cindex)},
            {"fixed_ctop", std::to_string(rt.index.fixed_ctop)},
            {"M", std::to_string(rt.index.M)},
            {"ef_construction", std::to_string(rt.index.ef_construction)},
            {"nprobe", std::to_string(nprobe)},
            {"ef_search", std::to_string(ef_search)},
            {"rerankK", std::to_string(rerankK)},
        };
        const size_t index_bytes = baseline::directory_file_size(args.index_dir);
        baseline::write_results_json(
            args.output,
            "gem_official_hnsw_adapter",
            args.doc_bundle,
            k,
            params,
            avg_recall_10,
            avg_recall_100,
            avg_recall_1000,
            avg_mrr_10,
            avg_success_5,
            avg_success_100,
            rt.index.build_time_sec,
            load_sec,
            search_sec,
            per_query_ms,
            index_bytes);
    }

    return 0;
}
