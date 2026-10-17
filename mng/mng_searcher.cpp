#include "mng_index.hpp"
#include "common/ranking_artifact.hpp"
#include "exact_maxsim_rerank.hpp"
#include "rerank_config.hpp"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <vector>

#ifndef MNG_EXACT_RERANK_STAGE_TIMERS
#define MNG_EXACT_RERANK_STAGE_TIMERS 0
#endif

#ifndef MNG_EXACT_RERANK_PROGRESS
#define MNG_EXACT_RERANK_PROGRESS 0
#endif

namespace {

uint32_t meta_u32(const std::unordered_map<std::string, std::string>& meta,
                  const std::string& key,
                  uint32_t fallback) {
    auto it = meta.find(key);
    if (it == meta.end()) return fallback;
    return static_cast<uint32_t>(std::stoul(it->second));
}

uint64_t meta_u64(const std::unordered_map<std::string, std::string>& meta,
                  const std::string& key,
                  uint64_t fallback) {
    auto it = meta.find(key);
    if (it == meta.end()) return fallback;
    return static_cast<uint64_t>(std::stoull(it->second));
}

double meta_double(const std::unordered_map<std::string, std::string>& meta,
                   const std::string& key,
                   double fallback) {
    auto it = meta.find(key);
    if (it == meta.end()) return fallback;
    return std::stod(it->second);
}

bool meta_bool(const std::unordered_map<std::string, std::string>& meta,
               const std::string& key,
               bool fallback) {
    auto it = meta.find(key);
    if (it == meta.end()) return fallback;
    return it->second == "true" || it->second == "1";
}

std::string meta_string(const std::unordered_map<std::string, std::string>& meta,
                        const std::string& key,
                        const std::string& fallback) {
    auto it = meta.find(key);
    if (it == meta.end()) return fallback;
    return it->second;
}

double parse_rerank_amp(const baseline::BenchArgs& args) {
    const std::string value = baseline::get_extra_string(args, "rerank_amp", "0");
    size_t consumed = 0;
    const double amp = std::stod(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error("invalid rerank_amp: " + value);
    }
    return amp;
}

std::string precise_double(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        baseline::enforce_single_thread_searcher("mng");

        auto args = baseline::parse_args(argc, argv);
        mng::SearchConfig cfg = mng::parse_search_config(args);
        const double rerank_amp = parse_rerank_amp(args);
        const auto requested_rerank =
            mng_exact_rerank::resolve_rerank_candidates(
                rerank_amp, cfg.topk, cfg.ef_search, cfg.ef_search);
        if (cfg.doc_bundle.empty() || cfg.query_bundle.empty() || cfg.index_dir.empty()) {
            std::cerr << "Usage: searcher_mng_exp --index_dir <dir> "
                      << "--doc <bundle> "
                      << "--query <bundle> [--gt <dir>] [--k 1000] [--output file] "
                      << "[--param ef_nav_query=128] [--param ef_search=1000] "
                      << "[--param rerank_amp=2.5]\n";
            return 1;
        }
        if (cfg.nav_only) {
            throw std::runtime_error(
                "exact rerank treatment requires the standard MNG proximity search");
        }
        if (cfg.exact_score_lookup) {
            throw std::runtime_error(
                "exact_score_lookup is forbidden in the exact rerank treatment");
        }

        mng::SearchStats stats;

        double t0 = mng::now_sec();
        auto meta = baseline::read_meta(cfg.index_dir);
        // Accept indexes created before the project was renamed to MNG.
        if (meta.at("method") != "mng" && meta.at("method") != "navprox") {
            throw std::runtime_error("index method is not mng");
        }
        mng::NavGraph nav;
        mng::load_nav_graph(cfg.index_dir, meta, nav);
        const bool nav_only = cfg.nav_only || meta_bool(meta, "nav_only", false);
        if (nav_only) {
            throw std::runtime_error(
                "exact rerank treatment requires a standard MNG index");
        }
        mng::ProxGraph prox;
        const uint32_t meta_num_docs = meta_u32(meta, "num_docs", 0);
        const uint32_t meta_dim = meta_u32(meta, "dim", 0);
        const uint32_t meta_M0 = meta_u32(meta, "M0", 0);
        if (!nav_only) {
            mng::load_prox_graph(cfg.index_dir, meta_num_docs, meta_M0, prox);
        }
        stats.load_index_sec = mng::now_sec() - t0;

        t0 = mng::now_sec();
        auto doc_bundle = mv::MVBundle::load(cfg.doc_bundle);
        auto query_bundle = mv::MVBundle::load(cfg.query_bundle);
        stats.read_doc_sec = mng::now_sec() - t0;

        mng::DocTable docs;
        mng::TokenStore doc_tokens;
        t0 = mng::now_sec();
        mng::load_doc_bundle_fp32(doc_bundle, docs, doc_tokens);
        mng::DocTable queries;
        mng::TokenStore query_tokens;
        mng::materialize_query_bundle_fp32(query_bundle, queries, query_tokens);
        stats.fp32_materialize_sec = mng::now_sec() - t0;

        if (docs.num_docs != meta_num_docs || docs.dim != meta_dim) {
            throw std::runtime_error("doc bundle shape does not match mng index meta");
        }
        if (queries.dim != docs.dim) {
            throw std::runtime_error("query dim does not match doc dim");
        }

        std::unique_ptr<mv::MVGroundTruth> gt;
        if (!cfg.gt_dir.empty()) {
            t0 = mng::now_sec();
            gt = std::make_unique<mv::MVGroundTruth>(mv::MVGroundTruth::load(cfg.gt_dir));
            stats.read_doc_sec += mng::now_sec() - t0;
        }
        const double load_sec =
            stats.load_index_sec + stats.read_doc_sec + stats.fp32_materialize_sec;
        const double build_sec = meta_double(meta, "build_time_sec", 0.0);

        mng::BuildConfig kcfg;
        kcfg.seed = meta_u64(meta, "seed", 42);
        kcfg.kmeans_niter = meta_u32(meta, "kmeans_niter", 25);
        kcfg.kmeans_nredo = meta_u32(meta, "kmeans_nredo", 5);

        mng::SearchScratch scratch;
        mng::ensure_search_scratch(scratch, queries.max_doclen, docs.max_doclen, docs.dim,
                                       docs.num_docs, nav.num_anchors, cfg.ef_nav_query,
                                       cfg.ef_search, cfg.topk);

        constexpr uint32_t kRerankBlockTokens = 1024;
        mng_exact_rerank::ExactRerankScratch rerank_scratch;
        if (requested_rerank.enabled) {
            rerank_scratch.prepare(
                std::min<uint32_t>(cfg.ef_search, docs.num_docs),
                queries.max_doclen, docs.dim, kRerankBlockTokens,
                docs.max_doclen);
        }
        const mng_exact_rerank::TokenStoreView rerank_docs{
            doc_tokens.tokens.data(), docs.token_offset.data(),
            docs.token_count.data(), docs.num_docs, docs.dim};

        std::cerr << "[mng-search] mode=" << mng::build_mode_string()
                  << " backend=" << mng::backend_string()
                  << " docs=" << docs.num_docs
                  << " queries=" << queries.num_docs
                  << " dim=" << docs.dim
                  << " k=" << cfg.topk
                  << " ef_nav_query=" << cfg.ef_nav_query
                  << " ef_search=" << cfg.ef_search
                  << " rerank_amp=" << precise_double(rerank_amp)
                  << " rerank_requested=" << requested_rerank.requested_candidates
                  << " query_scorer=centroid_maxsim"
                  << " rerank_scorer="
                  << (requested_rerank.enabled ? "exact_maxsim_fp32" : "disabled")
                  << " search_mode=mng\n";
        std::cerr << "[mng-search] Loaded index/docs in " << load_sec
                  << "s (index=" << stats.load_index_sec
                  << " read=" << stats.read_doc_sec
                  << " fp32=" << stats.fp32_materialize_sec << ")\n";

        std::vector<std::vector<int32_t>> all_topk(
            queries.num_docs, std::vector<int32_t>(cfg.topk, -1));
        uint32_t rerank_effective_min = std::numeric_limits<uint32_t>::max();
        uint32_t rerank_effective_max = 0;
        uint64_t rerank_effective_total = 0;
        const double search_t0 = mng::now_sec();
        for (uint32_t qi = 0; qi < queries.num_docs; ++qi) {
#if MNG_EXACT_RERANK_PROGRESS
            if (qi % 100 == 0) {
                std::cerr << "\r[mng-search] Query " << qi
                          << "/" << queries.num_docs << std::flush;
            }
#endif

#if MNG_EXACT_RERANK_STAGE_TIMERS
            t0 = mng::now_sec();
#endif
            mng::build_query_state(qi, queries, query_tokens, kcfg, scratch);
#if MNG_EXACT_RERANK_STAGE_TIMERS
            stats.query_precompute_sec += mng::now_sec() - t0;
#endif

            if (nav_only) {
#if MNG_EXACT_RERANK_STAGE_TIMERS
                t0 = mng::now_sec();
#endif
                const uint32_t found = mng::search_anchors_for_query(
                    cfg.ef_search, cfg.topk, nav, docs, doc_tokens, scratch,
                    scratch.nav_result_ids.data(), scratch.nav_result_scores.data());
#if MNG_EXACT_RERANK_STAGE_TIMERS
                stats.nav_query_sec += mng::now_sec() - t0;
#endif
                const uint32_t keep = std::min<uint32_t>(cfg.topk, found);
                for (uint32_t i = 0; i < keep; ++i) {
                    uint32_t anchor = scratch.nav_result_ids[i];
                    all_topk[qi][i] = static_cast<int32_t>(nav.anchor_doc_ids[anchor]);
                }
            } else {
#if MNG_EXACT_RERANK_STAGE_TIMERS
                t0 = mng::now_sec();
#endif
                const uint32_t n_nav = mng::search_anchors_for_query(
                    cfg.ef_nav_query, 1, nav, docs, doc_tokens, scratch,
                    scratch.nav_result_ids.data(), scratch.nav_result_scores.data());
#if MNG_EXACT_RERANK_STAGE_TIMERS
                stats.nav_query_sec += mng::now_sec() - t0;
#endif
                uint32_t entry_anchor = n_nav ? scratch.nav_result_ids[0] : nav.entry_anchor;
                uint32_t entry_doc = nav.anchor_doc_ids[entry_anchor];

#if MNG_EXACT_RERANK_STAGE_TIMERS
                t0 = mng::now_sec();
#endif
                const uint32_t ef_eff = std::min<uint32_t>(cfg.ef_search, docs.num_docs);
                const uint32_t found = mng::prox_search(
                    entry_doc, prox, docs, doc_tokens, ef_eff, scratch);
#if MNG_EXACT_RERANK_STAGE_TIMERS
                stats.prox_search_sec += mng::now_sec() - t0;
#endif

                uint32_t keep = std::min<uint32_t>(cfg.topk, found);
                const uint32_t* final_ids = scratch.result_ids.data();
                if (requested_rerank.enabled) {
                    const uint32_t effective = std::min(
                        {requested_rerank.requested_candidates, ef_eff, found});
                    const auto rerank_stats =
                        mng_exact_rerank::
                            rerank_blocked_sgemm_token_major_auto<false>(
                                scratch.query_tokens.data(),
                                scratch.query_token_count,
                                scratch.result_ids.data(),
                                effective, cfg.topk,
                                rerank_docs, rerank_scratch,
                                scratch.nav_result_ids.data());
                    keep = rerank_stats.results_written;
                    final_ids = scratch.nav_result_ids.data();
                    rerank_effective_min = std::min(rerank_effective_min, effective);
                    rerank_effective_max = std::max(rerank_effective_max, effective);
                    rerank_effective_total += effective;
                }
                for (uint32_t i = 0; i < keep; ++i) {
                    all_topk[qi][i] = static_cast<int32_t>(final_ids[i]);
                }
            }
        }
        stats.total_search_sec = mng::now_sec() - search_t0;
        stats.per_query_ms = queries.num_docs
            ? (stats.total_search_sec * 1000.0) / (double)queries.num_docs
            : 0.0;
        std::cerr << "\n[mng-search] Done in "
                  << stats.total_search_sec
                  << "s (" << stats.per_query_ms << " ms/q)";
#if MNG_EXACT_RERANK_STAGE_TIMERS
        std::cerr
                  << " query_precompute=" << stats.query_precompute_sec
                  << " nav=" << stats.nav_query_sec
                  << " prox=" << stats.prox_search_sec;
#endif
        std::cerr << "\n";

        baseline::write_rankings_with_exact_maxsim(
            cfg.output, all_topk, static_cast<size_t>(cfg.topk),
            cfg.doc_bundle, cfg.query_bundle);

        float avg_recall_10 = 0.0f;
        float avg_recall_100 = 0.0f;
        float avg_recall_1000 = 0.0f;
        float avg_mrr_10 = 0.0f;
        float avg_success_5 = 0.0f;
        float avg_success_100 = 0.0f;
        if (gt) {
            t0 = mng::now_sec();
            for (uint32_t qi = 0; qi < queries.num_docs; ++qi) {
                const int32_t* gt_topk = gt->topk_indices_row(qi);
                if (cfg.topk >= 10 && gt->top_k() >= 10) {
                    avg_recall_10 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 10);
                }
                if (cfg.topk >= 100 && gt->top_k() >= 100) {
                    avg_recall_100 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 100);
                }
                if (cfg.topk >= 1000 && gt->top_k() >= 1000) {
                    avg_recall_1000 += baseline::recall_at_k(all_topk[qi].data(), gt_topk, 1000);
                }
                std::unordered_set<int32_t> qrels_set;
                qrels_set.insert(gt_topk[0]);
                if (cfg.topk >= 10) {
                    avg_mrr_10 += baseline::mrr_at_k(all_topk[qi].data(), 10, qrels_set);
                }
                if (cfg.topk >= 5) {
                    avg_success_5 += baseline::success_at_k(all_topk[qi].data(), 5, qrels_set);
                }
                if (cfg.topk >= 100) {
                    avg_success_100 += baseline::success_at_k(all_topk[qi].data(), 100, qrels_set);
                }
            }
            const float nq = static_cast<float>(queries.num_docs);
            avg_recall_10 /= nq;
            avg_recall_100 /= nq;
            avg_recall_1000 /= nq;
            avg_mrr_10 /= nq;
            avg_success_5 /= nq;
            avg_success_100 /= nq;
            stats.eval_sec = mng::now_sec() - t0;
            std::cout << "[mng] Recall@10: " << avg_recall_10 << "\n";
            std::cout << "[mng] Recall@100: " << avg_recall_100 << "\n";
            std::cout << "[mng] Recall@1000: " << avg_recall_1000 << "\n";
            std::cout << "[mng] MRR@10: " << avg_mrr_10 << "\n";
            std::cout << "[mng] Success@5: " << avg_success_5 << "\n";
            std::cout << "[mng] Success@100: " << avg_success_100 << "\n";
        }

        std::cout << "[mng] Build: " << build_sec
                  << "s | Load: " << load_sec
                  << "s | Search: " << stats.total_search_sec
                  << "s (" << stats.per_query_ms << " ms/q)\n";

        if (!cfg.output.empty()) {
            const std::string assignment_gamma =
                meta_string(meta, "assignment_gamma", meta_string(meta, "gamma", ""));
            const std::string assignment_gamma_eff =
                meta_string(meta, "assignment_gamma_eff", meta_string(meta, "gamma_eff", ""));
            std::vector<std::pair<std::string, std::string>> params = {
                {"M", meta.at("M")},
                {"M0", meta.at("M0")},
                {"assignment_gamma", assignment_gamma},
                {"assignment_gamma_eff", assignment_gamma_eff},
                {"anchor_ratio", meta.at("anchor_ratio")},
                {"num_anchors", meta.at("num_anchors")},
                {"ef_nav_query", std::to_string(cfg.ef_nav_query)},
                {"ef_search", std::to_string(cfg.ef_search)},
                {"rerank_enabled", requested_rerank.enabled ? "true" : "false"},
                {"rerank_amp_requested", precise_double(rerank_amp)},
                {"rerank_candidates_requested",
                 std::to_string(requested_rerank.requested_candidates)},
                {"rerank_candidates_effective_min",
                 std::to_string(requested_rerank.enabled && queries.num_docs
                                    ? rerank_effective_min : 0)},
                {"rerank_candidates_effective_max",
                 std::to_string(requested_rerank.enabled ? rerank_effective_max : 0)},
                {"rerank_candidates_effective_mean",
                 requested_rerank.enabled && queries.num_docs
                     ? precise_double(static_cast<double>(rerank_effective_total) /
                                      queries.num_docs)
                     : "0"},
                {"rerank_kernel", requested_rerank.enabled
                    ? "blocked_sgemm_token_major_auto" : "disabled"},
                {"rerank_block_tokens", std::to_string(kRerankBlockTokens)},
                {"rerank_query32_pretranspose", "true"},
                {"rerank_diagnostics", "false"},
                {"rerank_runtime_validation", "false"},
                {"rerank_stage_timers", "false"},
                {"rerank_progress_output", "false"},
                {"rerank_score_normalization", "mean_over_query_tokens"},
                {"rerank_timing", "inside_total_search"},
                {"entry_mode", nav_only ? "none" : "anchor_doc"},
                {"search_mode", nav_only ? "nav_only" : "mng"},
                {"prox_selector", meta.at("prox_selector")},
                {"scorer", meta.at("scorer")},
                {"scorer_variant", meta_string(meta, "scorer_variant", "top1_maxsim")},
                {"aggregated_gamma_nn", meta_string(meta, "aggregated_gamma_nn", "false")},
                {"query_scorer", "centroid_filtered_maxsim"},
                {"final_rerank_scorer", requested_rerank.enabled
                    ? "exact_mean_maxsim" : "disabled"},
                {"scorer_backend", mng::backend_string()},
                {"build_mode", mng::build_mode_string()}
            };
            baseline::write_results_json(
                cfg.output, "mng", cfg.doc_bundle,
                static_cast<int>(cfg.topk), params,
                avg_recall_10, avg_recall_100, avg_recall_1000,
                avg_mrr_10, avg_success_5, avg_success_100,
                build_sec, load_sec, stats.total_search_sec,
                stats.per_query_ms, baseline::directory_file_size(cfg.index_dir));
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mng-search] ERROR: " << e.what() << "\n";
        return 2;
    }
}
