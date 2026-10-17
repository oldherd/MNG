#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/fp16_utils.hpp"
#include "common/timer.hpp"
#include "official_gem/official_gem_common.hpp"
#include "official_gem/official_gem_index.hpp"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<float> sample_training_tokens(const mv::MVBundle& doc_bundle,
                                          std::size_t max_train_tokens,
                                          int seed) {
    const int dim = doc_bundle.dim();
    const std::size_t n_tokens = doc_bundle.n_tokens();
    if (n_tokens == 0) throw std::runtime_error("official GEM sampled no training tokens");
    if (max_train_tokens == 0 || max_train_tokens > n_tokens) max_train_tokens = n_tokens;

    std::vector<std::size_t> ids;
    ids.reserve(max_train_tokens);
    if (max_train_tokens == n_tokens) {
        for (std::size_t i = 0; i < n_tokens; ++i) ids.push_back(i);
    } else {
        ids.resize(n_tokens);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 rng((uint32_t)seed);
        std::shuffle(ids.begin(), ids.end(), rng);
        ids.resize(max_train_tokens);
        std::sort(ids.begin(), ids.end());
    }

    const uint16_t* fp16 = doc_bundle.embeddings_fp16();
    std::vector<float> train(ids.size() * (std::size_t)dim);
    for (std::size_t out = 0; out < ids.size(); ++out) {
        const uint16_t* src = fp16 + ids[out] * (std::size_t)dim;
        float* dst = train.data() + out * (std::size_t)dim;
        for (int d = 0; d < dim; ++d) dst[d] = baseline::fp16_to_fp32(src[d]);
        official_gem::l2_normalize(dst, dim);
    }
    return train;
}

void normalize_rows(std::vector<float>& rows, int dim) {
    const std::size_t n = rows.size() / (std::size_t)dim;
    for (std::size_t i = 0; i < n; ++i) {
        official_gem::l2_normalize(rows.data() + i * (std::size_t)dim, dim);
    }
}

std::vector<float> train_faiss_kmeans(const std::vector<float>& train,
                                      int dim,
                                      int n_centroids,
                                      int niter,
                                      int seed,
                                      const char* name) {
    const std::size_t n_train = train.size() / (std::size_t)dim;
    if (n_train < (std::size_t)n_centroids) {
        throw std::runtime_error(std::string(name) +
                                 " k-means requires train_samples >= n_centroids");
    }

    faiss::Clustering clus(dim, n_centroids);
    clus.niter = niter;
    clus.seed = seed;
    clus.verbose = true;
    faiss::IndexFlatL2 quantizer(dim);
    clus.train((faiss::idx_t)n_train, train.data(), quantizer);

    std::vector<float> centroids = clus.centroids;
    normalize_rows(centroids, dim);
    return centroids;
}

std::vector<int32_t> assign_to_centroids_ip(const std::vector<float>& vectors,
                                            const std::vector<float>& centroids,
                                            int dim,
                                            int batch_size) {
    const std::size_t n = vectors.size() / (std::size_t)dim;
    const int n_centroids = (int)(centroids.size() / (std::size_t)dim);
    faiss::IndexFlatIP index(dim);
    index.add((faiss::idx_t)n_centroids, centroids.data());

    std::vector<int32_t> codes(n);
    std::vector<float> distances((std::size_t)batch_size);
    std::vector<faiss::idx_t> labels((std::size_t)batch_size);
    for (std::size_t start = 0; start < n; start += (std::size_t)batch_size) {
        std::size_t count = std::min<std::size_t>((std::size_t)batch_size, n - start);
        index.search((faiss::idx_t)count,
                     vectors.data() + start * (std::size_t)dim,
                     1,
                     distances.data(),
                     labels.data());
        for (std::size_t i = 0; i < count; ++i) codes[start + i] = (int32_t)labels[i];
    }
    return codes;
}

void quantize_doc_tokens(const mv::MVBundle& doc_bundle,
                         const std::vector<float>& cquant,
                         const std::vector<int32_t>& cquant_to_cindex,
                         int batch_size,
                         official_gem::OfficialGEMIndex& index,
                         std::vector<std::vector<std::pair<int32_t, int32_t>>>& doc_tf,
                         std::vector<int32_t>& df) {
    const int dim = doc_bundle.dim();
    const uint16_t* all_fp16 = doc_bundle.embeddings_fp16();
    faiss::IndexFlatIP cquant_search(dim);
    cquant_search.add((faiss::idx_t)index.n_cquant, cquant.data());

    std::vector<float> token_buf((std::size_t)batch_size * (std::size_t)dim);
    std::vector<float> distances((std::size_t)batch_size);
    std::vector<faiss::idx_t> labels((std::size_t)batch_size);

    for (std::size_t start = 0; start < index.n_tokens; start += (std::size_t)batch_size) {
        const std::size_t count = std::min<std::size_t>((std::size_t)batch_size,
                                                        index.n_tokens - start);
        for (std::size_t t = 0; t < count; ++t) {
            const uint16_t* src = all_fp16 + (start + t) * (std::size_t)dim;
            float* dst = token_buf.data() + t * (std::size_t)dim;
            for (int d = 0; d < dim; ++d) dst[d] = baseline::fp16_to_fp32(src[d]);
            official_gem::l2_normalize(dst, dim);
        }
        cquant_search.search((faiss::idx_t)count, token_buf.data(), 1,
                             distances.data(), labels.data());
        for (std::size_t t = 0; t < count; ++t) {
            index.token_codes[start + t] = (int32_t)labels[t];
        }
        if (start % (std::size_t)(batch_size * 100) == 0) {
            std::cerr << "\r[gem-official-indexer]   token " << start << "/"
                      << index.n_tokens << std::flush;
        }
    }
    std::cerr << "\n";

    for (std::size_t doc = 0; doc < index.n_docs; ++doc) {
        std::size_t begin = (std::size_t)index.offsets[doc];
        std::size_t end = (std::size_t)index.offsets[doc + 1u];
        std::vector<int32_t> coarse_codes;
        coarse_codes.reserve(end - begin);
        for (std::size_t t = begin; t < end; ++t) {
            coarse_codes.push_back(cquant_to_cindex[(std::size_t)index.token_codes[t]]);
        }
        std::sort(coarse_codes.begin(), coarse_codes.end());
        for (std::size_t i = 0; i < coarse_codes.size();) {
            std::size_t j = i + 1;
            while (j < coarse_codes.size() && coarse_codes[j] == coarse_codes[i]) ++j;
            doc_tf[doc].push_back({coarse_codes[i], (int32_t)(j - i)});
            df[(std::size_t)coarse_codes[i]] += 1;
            i = j;
        }
    }
}

void compute_tfidf_memberships(official_gem::OfficialGEMIndex& index,
                               const std::vector<std::vector<std::pair<int32_t, int32_t>>>& doc_tf,
                               const std::vector<int32_t>& df) {
    std::vector<double> idf((std::size_t)index.n_cindex);
    for (int c = 0; c < index.n_cindex; ++c) {
        idf[(std::size_t)c] = std::log((double)index.n_docs / (1.0 + (double)df[(std::size_t)c]));
    }

    index.cluster_to_docs.assign((std::size_t)index.n_cindex, {});
    for (std::size_t doc = 0; doc < index.n_docs; ++doc) {
        std::vector<std::pair<double, int32_t>> scored;
        scored.reserve(doc_tf[doc].size());
        for (auto [cluster, tf] : doc_tf[doc]) {
            scored.push_back({(double)tf * idf[(std::size_t)cluster], cluster});
        }
        auto desc = [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        };
        int take = std::min<int>(index.fixed_ctop, (int)scored.size());
        if ((int)scored.size() > take) {
            std::nth_element(scored.begin(), scored.begin() + take, scored.end(), desc);
            scored.resize((std::size_t)take);
        }
        std::sort(scored.begin(), scored.end(), desc);
        for (const auto& item : scored) {
            index.cluster_to_docs[(std::size_t)item.second].push_back((int32_t)doc);
        }
    }
	}

std::vector<int> nonempty_cluster_order(
    const std::vector<std::vector<hnswlib::labeltype>>& cluster_set) {
    std::vector<int> order;
    order.reserve(cluster_set.size());
    for (std::size_t cluster = 0; cluster < cluster_set.size(); ++cluster) {
        if (!cluster_set[cluster].empty()) order.push_back((int)cluster);
    }
    return order;
}

class OmpThreadScope {
public:
    explicit OmpThreadScope(int threads) : previous_(omp_get_max_threads()) {
        omp_set_num_threads(std::max(1, threads));
    }

    ~OmpThreadScope() {
        omp_set_num_threads(previous_);
    }

private:
    int previous_ = 1;
};

class OfficialFineClusterBuilder {
public:
    OfficialFineClusterBuilder(int m, int ef_construction, int n_graph_clusters)
        : m_(m), ef_construction_(ef_construction), n_graph_clusters_(n_graph_clusters) {}

    void build_fine_cluster(
        int d,
        const std::vector<vectorset>& base,
        const std::vector<std::vector<hnswlib::labeltype>>& cluster_set,
        const std::vector<int>& temp,
        const std::vector<float>& cluster_distance) {
        dimension_ = d;
        base_vectors_ = base;
        temp_cluster_id_ = temp;
        space_ptr_ = std::make_unique<hnswlib::L2VSSpace>((size_t)dimension_);
        cluster_entries_.assign((std::size_t)n_graph_clusters_, -1);
        alg_hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_ptr_.get(), base_vectors_.size() + 1u, (size_t)m_, (size_t)ef_construction_);

        for (int tmpi = 0; tmpi < (int)temp_cluster_id_.size(); ++tmpi) {
            const int cluster = temp_cluster_id_[(std::size_t)tmpi];
            if (cluster < 0 || cluster >= (int)cluster_set.size()) {
                throw std::runtime_error("official GEM cluster id out of range");
            }
            const auto& docs = cluster_set[(std::size_t)cluster];
            if (docs.empty()) continue;
            alg_hnsw_->entry_map.clear();
            cluster_entries_[(std::size_t)cluster] = (int)docs[0];
            alg_hnsw_->addClusterPointEntry(&base_vectors_[(std::size_t)docs[0]],
                                            cluster_distance.data(),
                                            docs[0],
                                            docs[0]);
            if (docs.size() > 1u) {
#pragma omp parallel for schedule(dynamic)
                for (std::ptrdiff_t j = 1; j < (std::ptrdiff_t)docs.size(); ++j) {
                    hnswlib::labeltype doc = docs[(std::size_t)j];
                    alg_hnsw_->addClusterPointEntry(&base_vectors_[(std::size_t)doc],
                                                    cluster_distance.data(),
                                                    doc,
                                                    docs[0]);
                }
            }
            if (tmpi % 100 == 0) {
                std::cerr << "\r[gem-official-indexer]   official cluster "
                          << cluster << "/" << n_graph_clusters_
                          << " docs=" << docs.size() << std::flush;
            }
        }
        std::cerr << "\n";
    }

    void save_fine_cluster(const std::string& hnsw_path) {
        if (!alg_hnsw_) throw std::runtime_error("official GEM HNSW graph not built");
        alg_hnsw_->saveIndex(hnsw_path);
    }

private:
    int m_ = 24;
    int ef_construction_ = 80;
    int n_graph_clusters_ = 0;
    int dimension_ = 0;
    std::vector<vectorset> base_vectors_;
    std::unique_ptr<hnswlib::L2VSSpace> space_ptr_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw_;
    std::vector<int> temp_cluster_id_;
    std::vector<int> cluster_entries_;
};

void build_official_hnsw(const official_gem::OfficialGEMIndex& index,
                         const std::vector<vectorset>& base,
                         const std::vector<std::vector<hnswlib::labeltype>>& cluster_set,
                         const std::vector<int>& temp_cluster_id,
                         std::vector<float>& cluster_distance,
                         const std::string& hnsw_path,
                         int graph_threads) {
    official_gem::set_official_cluster_count(index.n_cquant);
    OfficialFineClusterBuilder builder(index.M, index.ef_construction, index.n_cindex);
    OmpThreadScope graph_thread_scope(graph_threads);
    builder.build_fine_cluster(index.dim, base, cluster_set, temp_cluster_id, cluster_distance);
    builder.save_fine_cluster(hnsw_path);
}

}  // namespace

int main(int argc, char** argv) {
    auto args = baseline::parse_args(argc, argv);
    if (args.doc_bundle.empty() || args.index_dir.empty()) {
        std::cerr << "Usage: indexer_gem_official --doc <bundle> --index_dir <dir> "
                  << "[--param n_cquant=262144] [--param n_cindex=10000] "
                  << "[--param fixed_ctop=3] [--param M=24] "
                  << "[--param ef_construction=80] "
                  << "[--param allow_large_cluster_distance=1]\n";
        return 1;
    }

    const int fixed_ctop = baseline::get_extra_int(args, "fixed_ctop", 3);
    const int M = baseline::get_extra_int(args, "M", 24);
    const int ef_construction = baseline::get_extra_int(args, "ef_construction", 80);
	    const int nprobe = baseline::get_extra_int(args, "nprobe", 4);
	    const int kmeans_niter = baseline::get_extra_int(args, "kmeans_niter", 20);
	    const int seed = baseline::get_extra_int(args, "seed", 123);
	    const int batch_size = baseline::get_extra_int(args, "batch_size", 4096);
	    const int graph_threads = baseline::get_extra_int(args, "graph_threads", 1);
	    const int allow_large_cluster_distance =
	        baseline::get_extra_int(args, "allow_large_cluster_distance", 0);
    const float max_cluster_distance_gb =
        baseline::get_extra_float(args, "max_cluster_distance_gb", 64.0f);
    std::size_t max_train_tokens =
        (std::size_t)std::max(0, baseline::get_extra_int(args, "max_train_tokens", 0));

	    if (fixed_ctop <= 0 || M <= 0 || ef_construction <= 0 ||
	        nprobe <= 0 || batch_size <= 0 || graph_threads <= 0) {
	        std::cerr << "[gem-official-indexer] ERROR: fixed_ctop, M, ef_construction, "
	                  << "nprobe, batch_size, and graph_threads must be positive.\n";
	        return 1;
	    }

    std::cerr << "[gem-official-indexer] Loading doc bundle: " << args.doc_bundle << "\n";
    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    if (doc_bundle.dtype() != mv::DType::Float16) {
        std::cerr << "[gem-official-indexer] ERROR: expects fp16 doc .mvbundle input.\n";
        return 1;
    }
    if (doc_bundle.dim() != 128) {
        std::cerr << "[gem-official-indexer] ERROR: official GEM hnswlib path is hardwired "
                  << "for 128-d vectors; got dim=" << doc_bundle.dim() << "\n";
        return 1;
    }

    official_gem::OfficialGEMIndex index;
    index.dim = doc_bundle.dim();
    index.n_docs = doc_bundle.n_docs();
    index.n_tokens = doc_bundle.n_tokens();
    index.input_doc_bundle = args.doc_bundle;
    index.fixed_ctop = fixed_ctop;
    index.M = M;
    index.ef_construction = ef_construction;
    index.nprobe = nprobe;
    index.n_cquant = baseline::get_extra_int(args, "n_cquant", 262144);
    index.n_cquant = std::min<int>(index.n_cquant, (int)index.n_tokens);
    index.n_cindex = baseline::get_extra_int(args, "n_cindex", 10000);
    index.n_cindex = std::min<int>(index.n_cindex, index.n_cquant);
    if (index.n_docs == 0 || index.n_tokens == 0 || index.n_cquant <= 0 || index.n_cindex <= 0) {
        std::cerr << "[gem-official-indexer] ERROR: invalid empty bundle or centroid counts.\n";
        return 1;
    }
    if (max_train_tokens == 0) {
        max_train_tokens = std::min<std::size_t>(
            index.n_tokens,
            std::max<std::size_t>((std::size_t)index.n_cquant * 32u, 1000000u));
    }
    if (max_train_tokens < (std::size_t)index.n_cquant) {
        max_train_tokens = (std::size_t)index.n_cquant;
    }

    const long double cluster_distance_bytes =
        (long double)index.n_cquant * (long double)index.n_cquant * (long double)sizeof(float);
    const long double cluster_distance_gb = cluster_distance_bytes / (1024.0L * 1024.0L * 1024.0L);
    if (!allow_large_cluster_distance && cluster_distance_gb > (long double)max_cluster_distance_gb) {
        std::cerr << "[gem-official-indexer] Refusing to allocate official Cquant^2 matrix: "
                  << (double)cluster_distance_gb << " GiB. Re-run intentionally with "
                  << "--param allow_large_cluster_distance=1 or lower n_cquant/max_cluster_distance_gb.\n";
        return 2;
    }

    std::cerr << "[gem-official-indexer] Params: n_cquant=" << index.n_cquant
              << " n_cindex=" << index.n_cindex
              << " fixed_ctop=" << fixed_ctop
              << " M=" << M
              << " ef_construction=" << ef_construction
	              << " nprobe=" << nprobe
	              << " kmeans_niter=" << kmeans_niter
	              << " max_train_tokens=" << max_train_tokens
	              << " graph_threads=" << graph_threads
	              << " cluster_distance_gib=" << (double)cluster_distance_gb
	              << "\n";

    baseline::ensure_dir(args.index_dir);

    baseline::Timer build_timer;
    baseline::Timer stage_timer;
    build_timer.start();

    stage_timer.start();
    std::cerr << "[gem-official-indexer] Sampling Cquant training tokens...\n";
    std::vector<float> train = sample_training_tokens(doc_bundle, max_train_tokens, seed);
    stage_timer.stop();
    double sample_sec = stage_timer.elapsed_sec();

    stage_timer.start();
    std::cerr << "[gem-official-indexer] Training Cquant k-means...\n";
    index.cquant_centroids =
        train_faiss_kmeans(train, index.dim, index.n_cquant, kmeans_niter, seed, "Cquant");
    stage_timer.stop();
    double cquant_train_sec = stage_timer.elapsed_sec();

    stage_timer.start();
    std::cerr << "[gem-official-indexer] Training Cindex k-means over Cquant centroids...\n";
    index.cindex_centroids =
        train_faiss_kmeans(index.cquant_centroids, index.dim, index.n_cindex,
                           kmeans_niter, seed + 1, "Cindex");
    stage_timer.stop();
    double cindex_train_sec = stage_timer.elapsed_sec();

    stage_timer.start();
    std::cerr << "[gem-official-indexer] Assigning Cquant centroids to Cindex...\n";
    index.cquant_to_cindex =
        assign_to_centroids_ip(index.cquant_centroids, index.cindex_centroids,
                               index.dim, batch_size);
    stage_timer.stop();
    double cquant_to_cindex_sec = stage_timer.elapsed_sec();

    index.doclens.assign(doc_bundle.doclens(), doc_bundle.doclens() + index.n_docs);
    index.offsets.assign(doc_bundle.offsets(), doc_bundle.offsets() + index.n_docs + 1u);
    index.token_codes.resize(index.n_tokens);

    stage_timer.start();
    std::cerr << "[gem-official-indexer] Quantizing document tokens...\n";
    std::vector<std::vector<std::pair<int32_t, int32_t>>> doc_tf(index.n_docs);
    std::vector<int32_t> df((std::size_t)index.n_cindex, 0);
    quantize_doc_tokens(doc_bundle, index.cquant_centroids, index.cquant_to_cindex,
                        batch_size, index, doc_tf, df);
    stage_timer.stop();
    double quantize_sec = stage_timer.elapsed_sec();

    stage_timer.start();
    std::cerr << "[gem-official-indexer] Computing TF-IDF top-" << fixed_ctop
              << " official graph clusters...\n";
    compute_tfidf_memberships(index, doc_tf, df);
	    stage_timer.stop();
	    double tfidf_sec = stage_timer.elapsed_sec();
	    std::vector<std::vector<std::pair<int32_t, int32_t>>>().swap(doc_tf);
	    std::vector<int32_t>().swap(df);

	    stage_timer.start();
	    std::cerr << "[gem-official-indexer] Materializing official GEM base memory...\n";
	    official_gem::OfficialBaseMemory official_base =
	        official_gem::make_official_base_memory(doc_bundle, index);
	    auto cluster_set = official_gem::make_cluster_sets(index);
	    std::vector<int> temp_cluster_id = nonempty_cluster_order(cluster_set);
	    stage_timer.stop();
	    double official_base_materialize_sec = stage_timer.elapsed_sec();

	    stage_timer.start();
	    std::cerr << "[gem-official-indexer] Building official Cquant^2 centroid matrix...\n";
	    std::vector<float> cluster_distance((std::size_t)index.n_cquant *
	                                        (std::size_t)index.n_cquant);
    hnswlib::fast_dot_product_blas(index.n_cquant,
                                   index.dim,
                                   index.n_cquant,
                                   index.cquant_centroids.data(),
                                   index.cquant_centroids.data(),
                                   cluster_distance.data());
    stage_timer.stop();
    double cluster_distance_sec = stage_timer.elapsed_sec();

	    stage_timer.start();
	    std::cerr << "[gem-official-indexer] Building official GEM HNSW graph...\n";
	    build_official_hnsw(index,
	                        official_base.base,
	                        cluster_set,
	                        temp_cluster_id,
	                        cluster_distance,
	                        index.hnsw_path(args.index_dir),
	                        graph_threads);
	    stage_timer.stop();
	    double hnsw_build_sec = stage_timer.elapsed_sec();

	    build_timer.stop();
    index.build_time_sec = build_timer.elapsed_sec();
    index.save_metadata(args.index_dir, {
        {"sample_sec", std::to_string(sample_sec)},
        {"cquant_train_sec", std::to_string(cquant_train_sec)},
        {"cindex_train_sec", std::to_string(cindex_train_sec)},
        {"cquant_to_cindex_sec", std::to_string(cquant_to_cindex_sec)},
        {"quantize_sec", std::to_string(quantize_sec)},
	        {"tfidf_sec", std::to_string(tfidf_sec)},
	        {"official_base_materialize_sec", std::to_string(official_base_materialize_sec)},
	        {"cluster_distance_sec", std::to_string(cluster_distance_sec)},
	        {"hnsw_build_sec", std::to_string(hnsw_build_sec)},
	        {"cluster_distance_gib", std::to_string((double)cluster_distance_gb)},
	        {"official_base_data_bytes", std::to_string(official_base.data_bytes())},
	        {"official_base_code_bytes", std::to_string(official_base.code_bytes())},
	        {"official_nonempty_graph_clusters", std::to_string(temp_cluster_id.size())},
	        {"official_graph_build_interface", "Solution::build_fine_cluster_input_shape"},
	        {"official_graph_insert_parallel", "upstream_openmp_dynamic"},
	        {"official_graph_threads", std::to_string(graph_threads)},
	        {"kmeans_niter", std::to_string(kmeans_niter)},
	        {"seed", std::to_string(seed)},
	        {"max_train_tokens", std::to_string(max_train_tokens)},
        {"batch_size", std::to_string(batch_size)},
    });

    std::cerr << "[gem-official-indexer] Done. build_time_sec="
              << index.build_time_sec << " index_dir=" << args.index_dir << "\n";
    return 0;
}
