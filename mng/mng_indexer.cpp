#include "mng_index.hpp"

#include <exception>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    try {
        auto args = baseline::parse_args(argc, argv);
        mng::BuildConfig cfg = mng::parse_build_config(args);
        if (cfg.doc_bundle.empty() || cfg.index_dir.empty()) {
            std::cerr << "Usage: indexer_mng_exp --doc <bundle> --index_dir <dir> "
                      << "[--param M=32] [--param M0=64] [--param assignment_gamma=M0] "
                      << "[--param anchor_ratio=32] [--param ef_nav_build=128] "
                      << "[--param ef_nav_assign=128] [--param seed=42] "
                      << "[--param kmeans_niter=25] [--param kmeans_nredo=5] "
                      << "[--param prox_builder=nearest_top_m0|rank_mutual] "
                      << "[--param nav_only=1]\n"
                      << "Legacy alias: --param gamma=<n> is accepted as assignment_gamma.\n";
            return 1;
        }

        baseline::set_env_if_possible("OMP_DYNAMIC", "FALSE");
        baseline::set_env_if_possible("MKL_NUM_THREADS", "1");
        baseline::set_env_if_possible("OPENBLAS_NUM_THREADS", "1");
        baseline::set_env_if_possible("BLIS_NUM_THREADS", "1");
        baseline::set_env_if_possible("NUMEXPR_NUM_THREADS", "1");
        omp_set_dynamic(0);
        omp_set_num_threads(MNG_INDEX_THREADS);
#if _OPENMP >= 200805
        omp_set_max_active_levels(1);
#endif

        std::cerr << "[mng-index] mode=" << mng::build_mode_string()
                  << " backend=" << mng::backend_string()
                  << " threads=" << MNG_INDEX_THREADS << "\n";
        std::cerr << "[mng-index] Params: M=" << cfg.M
                  << " M0=" << cfg.M0
                  << " assignment_gamma=" << cfg.assignment_gamma
                  << " anchor_ratio=" << cfg.anchor_ratio
                  << " ef_nav_build=" << cfg.ef_nav_build
                  << " ef_nav_assign=" << cfg.ef_nav_assign
                  << " nav_only=" << (cfg.nav_only ? 1 : 0)
                  << " seed=" << cfg.seed
                  << " kmeans_niter=" << cfg.kmeans_niter
                  << " kmeans_nredo=" << cfg.kmeans_nredo
                  << " prox_builder=" << cfg.prox_builder << "\n";
        std::cerr << "[mng-index] Loading doc bundle: " << cfg.doc_bundle << "\n";

        mng::BuildStats stats;
        const double total_t0 = mng::now_sec();

        double t0 = mng::now_sec();
        auto doc_bundle = mv::MVBundle::load(cfg.doc_bundle);
        stats.read_sec = mng::now_sec() - t0;

        mng::DocTable docs;
        mng::TokenStore tokens;
        t0 = mng::now_sec();
        mng::load_doc_bundle_fp32(doc_bundle, docs, tokens);
        stats.fp32_materialize_sec = mng::now_sec() - t0;

        if (docs.num_docs == 0 || docs.total_tokens == 0) {
            throw std::runtime_error("empty document bundle");
        }

        std::cerr << "[mng-index] docs=" << docs.num_docs
                  << " tokens=" << docs.total_tokens
                  << " dim=" << docs.dim
                  << " max_doclen=" << docs.max_doclen << "\n";

        mng::CentroidStore centroids;
        t0 = mng::now_sec();
        mng::build_centroids(docs, tokens, cfg, centroids);
        stats.centroid_build_sec = mng::now_sec() - t0;
        std::cerr << "[mng-index] Centroids: total=" << centroids.total_centroids
                  << " max_per_doc=" << centroids.max_centroids_per_doc
                  << " time=" << stats.centroid_build_sec << "s\n";

        mng::NavGraph nav;
        std::vector<uint32_t> doc_to_anchor;
        mng::init_nav_graph(nav, docs, cfg, doc_to_anchor);
        std::cerr << "[mng-index] Building navigation graph: anchors="
                  << nav.num_anchors << " max_level=" << nav.max_level << "\n";
        t0 = mng::now_sec();
        mng::build_nav_graph(nav, docs, tokens, centroids, cfg);
        stats.nav_build_sec = mng::now_sec() - t0;

        if (cfg.nav_only) {
            stats.core_build_sec = stats.centroid_build_sec + stats.nav_build_sec;
            stats.total_build_sec = mng::now_sec() - total_t0;
            std::cerr << "[mng-index] nav_only=1: skipping assignment, buckets, and proximity graph\n";
            std::cerr << "[mng-index] Build stages:"
                      << " read=" << stats.read_sec
                      << " fp32=" << stats.fp32_materialize_sec
                      << " centroid=" << stats.centroid_build_sec
                      << " nav=" << stats.nav_build_sec
                      << " assign=0 bucket=0 bound=0 prox=0"
                      << " core=" << stats.core_build_sec << "s\n";
            mng::save_nav_only_index(cfg.index_dir, cfg, docs, nav, stats);
            const size_t index_bytes = baseline::directory_file_size(cfg.index_dir);
            std::cerr << "[mng-index] Saved nav-only index to " << cfg.index_dir
                      << " save=" << stats.save_sec
                      << " total=" << stats.total_build_sec
                      << " index_bytes=" << index_bytes << "\n";
            return 0;
        }

        mng::AssignmentStore assignment;
        t0 = mng::now_sec();
        mng::build_assignment(nav, doc_to_anchor, docs, tokens, centroids, cfg, assignment);
        stats.assignment_sec = mng::now_sec() - t0;

        mng::BucketStore buckets;
        t0 = mng::now_sec();
        mng::build_buckets(assignment, buckets);
        stats.bucket_build_sec = mng::now_sec() - t0;

        t0 = mng::now_sec();
        const uint64_t candidate_bound = mng::compute_candidate_bound(assignment, buckets, stats);
        stats.candidate_bound_sec = mng::now_sec() - t0;

        std::cerr << "[mng-index] Candidate pool: mean=" << stats.candidate_mean
                  << " p50=" << stats.candidate_p50
                  << " p95=" << stats.candidate_p95
                  << " p99=" << stats.candidate_p99
                  << " max=" << stats.candidate_max << "\n";

        mng::ProxGraph prox;
        t0 = mng::now_sec();
        mng::build_prox_graph(assignment, buckets, docs, tokens, centroids,
                                  cfg, candidate_bound, prox, stats);
        stats.prox_build_sec = mng::now_sec() - t0;

        stats.core_build_sec =
            stats.centroid_build_sec + stats.nav_build_sec + stats.assignment_sec +
            stats.bucket_build_sec + stats.candidate_bound_sec + stats.prox_build_sec;
        stats.total_build_sec = mng::now_sec() - total_t0;

        std::cerr << "[mng-index] Build stages:"
                  << " read=" << stats.read_sec
                  << " fp32=" << stats.fp32_materialize_sec
                  << " centroid=" << stats.centroid_build_sec
                  << " nav=" << stats.nav_build_sec
                  << " assign=" << stats.assignment_sec
                  << " bucket=" << stats.bucket_build_sec
                  << " bound=" << stats.candidate_bound_sec
                  << " prox=" << stats.prox_build_sec
                  << " core=" << stats.core_build_sec << "s\n";
        std::cerr << "[mng-index] Edges=" << stats.selected_edges
                  << " underfilled_rows=" << stats.underfilled_rows
                  << " rank_reverse_edges=" << stats.rank_reverse_edges
                  << " rank_reverse_max=" << stats.rank_reverse_max
                  << " rank_pruned_rows=" << stats.rank_pruned_rows
                  << " prox_score_forward=" << stats.prox_score_forward_sec
                  << " prox_score_reverse=" << stats.prox_score_reverse_sec << "\n";

        mng::save_index(cfg.index_dir, cfg, docs, nav, assignment, buckets, prox, stats);
        const size_t index_bytes = baseline::directory_file_size(cfg.index_dir);
        std::cerr << "[mng-index] Saved to " << cfg.index_dir
                  << " save=" << stats.save_sec
                  << " total=" << stats.total_build_sec
                  << " index_bytes=" << index_bytes << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mng-index] ERROR: " << e.what() << "\n";
        return 2;
    }
}
