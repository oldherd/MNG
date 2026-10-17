// IGP vector-search indexer over a shared .mvbundle input.
//
// Builds the paper/upstream IGP structures:
//   - VQ centroids, n_c = 2^floor(log2(16 * sqrt(N_v))) by default
//   - B-bit residual scalar quantization, B=2 by default
//   - centroid -> unique passage ids inverted file
//   - HNSW proximity graph over centroids, M=64 and efConstruction=200

#include "mvbundle.hpp"
#include "common/ann_hnswlib.hpp"
#include "common/bench_utils.hpp"
#include "common/fp16_utils.hpp"
#include "common/timer.hpp"
#include "igp_index.hpp"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string get_extra_string(const baseline::BenchArgs& args,
                             const std::string& key,
                             const std::string& default_val) {
    for (const auto& kv : args.extra) {
        if (kv.first == key) return kv.second;
    }
    return default_val;
}

void fp16_token_to_fp32(const uint16_t* src, int dim, float* dst) {
    for (int d = 0; d < dim; ++d) dst[d] = baseline::fp16_to_fp32(src[d]);
}

void l2_normalize(float* x, int dim) {
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) ss += (double)x[d] * (double)x[d];
    double inv = 1.0 / std::sqrt(std::max(ss, 1e-30));
    for (int d = 0; d < dim; ++d) x[d] = (float)((double)x[d] * inv);
}

void normalize_rows(std::vector<float>& rows, int dim) {
    std::size_t n = rows.size() / (std::size_t)dim;
    for (std::size_t i = 0; i < n; ++i) {
        l2_normalize(rows.data() + i * (std::size_t)dim, dim);
    }
}

int default_num_partitions(std::size_t n_tokens) {
    if (n_tokens == 0) return 0;
    double target = 16.0 * std::sqrt((double)n_tokens);
    int pow2 = 1;
    while ((double)(pow2 * 2) <= target && pow2 < (1 << 30)) pow2 *= 2;
    return (int)std::min<std::size_t>((std::size_t)std::max(1, pow2), n_tokens);
}

std::vector<int32_t> sample_doc_ids(std::size_t n_docs,
                                    int sample_docs,
                                    int sample_seed) {
    sample_docs = std::max(1, std::min<int>(sample_docs, (int)n_docs));
    std::vector<int32_t> ids(n_docs);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng((uint32_t)sample_seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize((std::size_t)sample_docs);
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::vector<float> sample_training_docs(const mv::MVBundle& doc_bundle,
                                        int sample_docs,
                                        int sample_seed,
                                        std::size_t max_train_tokens,
                                        std::vector<int32_t>& sampled_pids) {
    const int dim = doc_bundle.dim();
    const std::size_t n_docs = doc_bundle.n_docs();
    const int64_t* offsets = doc_bundle.offsets();
    const uint16_t* all = doc_bundle.embeddings_fp16();

    sampled_pids = sample_doc_ids(n_docs, sample_docs, sample_seed);
    std::size_t n_tokens = 0;
    for (int32_t pid : sampled_pids) {
        n_tokens += (std::size_t)(offsets[(std::size_t)pid + 1u] - offsets[(std::size_t)pid]);
    }
    if (max_train_tokens > 0) n_tokens = std::min(n_tokens, max_train_tokens);
    if (n_tokens == 0) throw std::runtime_error("IGP sampled no training tokens");

    std::vector<float> train(n_tokens * (std::size_t)dim);
    std::size_t out = 0;
    for (int32_t pid : sampled_pids) {
        std::size_t begin = (std::size_t)offsets[(std::size_t)pid];
        std::size_t end = (std::size_t)offsets[(std::size_t)pid + 1u];
        for (std::size_t token = begin; token < end && out < n_tokens; ++token) {
            fp16_token_to_fp32(all + token * (std::size_t)dim, dim,
                               train.data() + out * (std::size_t)dim);
            ++out;
        }
        if (out >= n_tokens) break;
    }
    train.resize(out * (std::size_t)dim);
    return train;
}

std::vector<float> train_faiss_kmeans(const std::vector<float>& train,
                                      int dim,
                                      int n_centroids,
                                      int niter,
                                      int seed) {
    const std::size_t n_train = train.size() / (std::size_t)dim;
    if (n_train < (std::size_t)n_centroids) {
        throw std::runtime_error("IGP k-means requires train_samples >= n_centroids");
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

struct BucketStats {
    std::vector<float> cutoffs;
    std::vector<float> weights;
    double avg_abs_residual = 0.0;
};

float floor_quantile_from_sorted(const std::vector<float>& sorted, double q) {
    if (sorted.empty()) return 0.0f;
    q = std::max(0.0, std::min(1.0, q));
    std::size_t idx = (std::size_t)std::floor(q * (double)sorted.size());
    if (idx >= sorted.size()) idx = sorted.size() - 1u;
    return sorted[idx];
}

BucketStats compute_bucket_stats(const std::vector<float>& sample,
                                 const std::vector<int32_t>& sample_codes,
                                 const std::vector<float>& centroids,
                                 int dim,
                                 int nbits) {
    const std::size_t n_sample = sample.size() / (std::size_t)dim;
    if (n_sample == 0 || sample_codes.size() != n_sample) {
        throw std::runtime_error("invalid IGP SQ training sample");
    }

    std::vector<float> residual_values;
    residual_values.reserve(sample.size());
    double abs_sum = 0.0;
    for (std::size_t i = 0; i < n_sample; ++i) {
        const float* x = sample.data() + i * (std::size_t)dim;
        const float* c = centroids.data() + (std::size_t)sample_codes[i] * (std::size_t)dim;
        for (int d = 0; d < dim; ++d) {
            float r = x[d] - c[d];
            residual_values.push_back(r);
            abs_sum += std::fabs((double)r);
        }
    }

    std::sort(residual_values.begin(), residual_values.end());
    const int n_quantile = 1 << nbits;
    BucketStats stats;
    for (int i = 1; i < n_quantile; ++i) {
        stats.cutoffs.push_back(
            floor_quantile_from_sorted(residual_values, (double)i / (double)n_quantile));
    }
    for (int i = 0; i < n_quantile; ++i) {
        stats.weights.push_back(
            floor_quantile_from_sorted(residual_values, ((double)i + 0.5) / (double)n_quantile));
    }
    stats.avg_abs_residual = abs_sum / (double)residual_values.size();
    return stats;
}

uint8_t bucketize_residual(float residual, const std::vector<float>& cutoffs) {
    return (uint8_t)(std::upper_bound(cutoffs.begin(), cutoffs.end(), residual) - cutoffs.begin());
}

void pack_bucket(uint8_t* row, int dim_index, int nbits, uint8_t bucket) {
    int bit_pos = dim_index * nbits;
    int byte_index = bit_pos / 8;
    int offset_in_byte = bit_pos % 8;
    int shift = 8 - offset_in_byte - nbits;
    row[byte_index] |= (uint8_t)(bucket << shift);
}

std::vector<int32_t> assign_sample_codes(const std::vector<float>& sample,
                                         const std::vector<float>& centroids,
                                         int dim,
                                         int n_centroids,
                                         int batch_size) {
    const std::size_t n_sample = sample.size() / (std::size_t)dim;
    faiss::IndexFlatIP centroid_index(dim);
    centroid_index.add((faiss::idx_t)n_centroids, centroids.data());

    std::vector<int32_t> codes(n_sample);
    std::vector<float> distances((std::size_t)batch_size);
    std::vector<faiss::idx_t> labels((std::size_t)batch_size);
    for (std::size_t start = 0; start < n_sample; start += (std::size_t)batch_size) {
        std::size_t count = std::min<std::size_t>((std::size_t)batch_size, n_sample - start);
        centroid_index.search((faiss::idx_t)count,
                              sample.data() + start * (std::size_t)dim,
                              1, distances.data(), labels.data());
        for (std::size_t i = 0; i < count; ++i) {
            codes[start + i] = (int32_t)labels[i];
        }
    }
    return codes;
}

void assign_and_compress_all_tokens(const mv::MVBundle& doc_bundle,
                                    const std::vector<float>& centroids,
                                    const std::vector<float>& cutoffs,
                                    int n_centroids,
                                    int nbits,
                                    int batch_size,
                                    std::vector<int32_t>& assignments,
                                    std::vector<uint8_t>& residuals) {
    const int dim = doc_bundle.dim();
    const std::size_t n_tokens = doc_bundle.n_tokens();
    const std::size_t packed_dim = igp::packed_residual_dim(dim, nbits);
    const uint16_t* all = doc_bundle.embeddings_fp16();

    faiss::IndexFlatIP centroid_index(dim);
    centroid_index.add((faiss::idx_t)n_centroids, centroids.data());

    std::vector<float> batch((std::size_t)batch_size * (std::size_t)dim);
    std::vector<float> distances((std::size_t)batch_size);
    std::vector<faiss::idx_t> labels((std::size_t)batch_size);

    for (std::size_t start = 0; start < n_tokens; start += (std::size_t)batch_size) {
        std::size_t count = std::min<std::size_t>((std::size_t)batch_size, n_tokens - start);
        for (std::size_t i = 0; i < count; ++i) {
            fp16_token_to_fp32(all + (start + i) * (std::size_t)dim, dim,
                               batch.data() + i * (std::size_t)dim);
        }
        centroid_index.search((faiss::idx_t)count, batch.data(), 1, distances.data(), labels.data());
        for (std::size_t i = 0; i < count; ++i) {
            int32_t code = (int32_t)labels[i];
            assignments[start + i] = code;
            const float* x = batch.data() + i * (std::size_t)dim;
            const float* c = centroids.data() + (std::size_t)code * (std::size_t)dim;
            uint8_t* packed = residuals.data() + (start + i) * packed_dim;
            std::fill(packed, packed + packed_dim, 0);
            for (int d = 0; d < dim; ++d) {
                pack_bucket(packed, d, nbits, bucketize_residual(x[d] - c[d], cutoffs));
            }
        }
        if (start == 0 || ((start / (std::size_t)batch_size) % 20 == 0)) {
            std::cerr << "[igp-indexer] compressed " << (start + count)
                      << "/" << n_tokens << " tokens\n";
        }
    }
}

std::vector<std::vector<int32_t>> build_centroid_to_pids(const mv::MVBundle& doc_bundle,
                                                         const std::vector<int32_t>& assignments,
                                                         int n_centroids) {
    std::vector<std::vector<int32_t>> lists((std::size_t)n_centroids);
    const std::size_t n_docs = doc_bundle.n_docs();
    const int64_t* offsets = doc_bundle.offsets();
    std::vector<int32_t> doc_codes;

    for (std::size_t pid = 0; pid < n_docs; ++pid) {
        std::size_t begin = (std::size_t)offsets[pid];
        std::size_t end = (std::size_t)offsets[pid + 1u];
        doc_codes.clear();
        doc_codes.reserve(end - begin);
        for (std::size_t token = begin; token < end; ++token) {
            doc_codes.push_back(assignments[token]);
        }
        std::sort(doc_codes.begin(), doc_codes.end());
        doc_codes.erase(std::unique(doc_codes.begin(), doc_codes.end()), doc_codes.end());
        for (int32_t code : doc_codes) lists[(std::size_t)code].push_back((int32_t)pid);

        if (pid == 0 || (pid % 100000 == 0)) {
            std::cerr << "[igp-indexer] built centroid pid lists through pid "
                      << pid << "/" << n_docs << "\n";
        }
    }
    return lists;
}

std::vector<int32_t> bundle_doclens(const mv::MVBundle& doc_bundle) {
    std::vector<int32_t> doclens(doc_bundle.n_docs());
    const int64_t* offsets = doc_bundle.offsets();
    for (std::size_t pid = 0; pid < doclens.size(); ++pid) {
        doclens[pid] = (int32_t)(offsets[pid + 1u] - offsets[pid]);
    }
    return doclens;
}

std::vector<int64_t> bundle_offsets(const mv::MVBundle& doc_bundle) {
    std::vector<int64_t> offsets(doc_bundle.n_docs() + 1u);
    const int64_t* src = doc_bundle.offsets();
    std::copy(src, src + offsets.size(), offsets.begin());
    return offsets;
}

}  // namespace

int main(int argc, char** argv) {
    auto args = baseline::parse_args(argc, argv);
    if (args.doc_bundle.empty() || args.index_dir.empty()) {
        std::cerr << "Usage: indexer_igp --doc <bundle> --index_dir <dir> "
                  << "[--param nbits=2] [--param n_centroids=0] "
                  << "[--param kmeans_niters=20] [--param sample_doc_factor=8] "
                  << "[--param typical_doclen=120] [--param max_train_tokens=0] "
                  << "[--param M=64] [--param ef_construction=200]\n";
        return 1;
    }

    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    if (doc_bundle.dtype() != mv::DType::Float16) {
        throw std::runtime_error("IGP indexer expects fp16 doc .mvbundle input");
    }

    const int dim = doc_bundle.dim();
    const std::size_t n_docs = doc_bundle.n_docs();
    const std::size_t n_tokens = doc_bundle.n_tokens();
    if (n_docs == 0 || n_tokens == 0) throw std::runtime_error("empty doc bundle");

    int nbits = baseline::get_extra_int(args, "nbits", 2);
    int n_centroids = baseline::get_extra_int(args, "n_centroids", 0);
    int kmeans_niters = baseline::get_extra_int(args, "kmeans_niters", 20);
    int sample_doc_factor = baseline::get_extra_int(args, "sample_doc_factor", 8);
    int typical_doclen = baseline::get_extra_int(args, "typical_doclen", 120);
    int sample_docs_arg = baseline::get_extra_int(args, "sample_docs", 0);
    int max_train_tokens_arg = baseline::get_extra_int(args, "max_train_tokens", 0);
    int sample_seed = baseline::get_extra_int(args, "sample_seed", 12345);
    int kmeans_seed = baseline::get_extra_int(args, "kmeans_seed", 123);
    int assign_batch = baseline::get_extra_int(args, "assign_batch", 65536);
    int pg_m = baseline::get_extra_int(args, "M", 64);
    int ef_construction = baseline::get_extra_int(args, "ef_construction", 200);
    std::string profile = get_extra_string(args, "profile", "paper_igp_upstream");

    if (!(nbits == 1 || nbits == 2 || nbits == 4 || nbits == 8)) {
        throw std::runtime_error("IGP nbits must be in {1,2,4,8}");
    }
    if (8 % nbits != 0) {
        throw std::runtime_error("IGP residual packing expects nbits to divide 8");
    }
    if (n_centroids == 0) n_centroids = default_num_partitions(n_tokens);
    n_centroids = std::max(1, std::min<int>(n_centroids, (int)n_tokens));
    assign_batch = std::max(1, assign_batch);

    int sample_docs = sample_docs_arg;
    if (sample_docs <= 0) {
        sample_docs = 1 + (int)(sample_doc_factor *
            std::sqrt((double)typical_doclen * (double)n_docs));
    }
    sample_docs = std::max(1, std::min<int>(sample_docs, (int)n_docs));
    std::size_t max_train_tokens = max_train_tokens_arg > 0 ? (std::size_t)max_train_tokens_arg : 0u;

    std::cerr << "[igp-indexer] docs=" << n_docs << " tokens=" << n_tokens
              << " dim=" << dim << " n_centroids=" << n_centroids
              << " nbits=" << nbits << " kmeans_niters=" << kmeans_niters
              << " sample_docs=" << sample_docs
              << " max_train_tokens=" << max_train_tokens
              << " M=" << pg_m << " efConstruction=" << ef_construction
              << " profile=" << profile << "\n";

    baseline::Timer build_timer;
    build_timer.start();

    std::cerr << "[igp-indexer] sampling training docs...\n";
    std::vector<int32_t> sampled_pids;
    std::vector<float> train = sample_training_docs(doc_bundle, sample_docs, sample_seed,
                                                    max_train_tokens, sampled_pids);
    std::size_t train_samples = train.size() / (std::size_t)dim;
    std::cerr << "[igp-indexer] training_samples=" << train_samples << "\n";
    if (train_samples < (std::size_t)n_centroids) {
        throw std::runtime_error("IGP train sample is smaller than n_centroids; increase sample_docs");
    }

    std::cerr << "[igp-indexer] training FAISS k-means...\n";
    baseline::Timer kmeans_timer;
    kmeans_timer.start();
    std::vector<float> centroids = train_faiss_kmeans(train, dim, n_centroids,
                                                      kmeans_niters, kmeans_seed);
    kmeans_timer.stop();
    double centroid_build_sec = kmeans_timer.elapsed_sec();

    std::cerr << "[igp-indexer] assigning sample codes for SQ training...\n";
    std::vector<int32_t> train_codes = assign_sample_codes(train, centroids, dim,
                                                           n_centroids, assign_batch);
    std::cerr << "[igp-indexer] computing residual SQ buckets...\n";
    BucketStats stats = compute_bucket_stats(train, train_codes, centroids, dim, nbits);
    train.clear();
    train.shrink_to_fit();
    train_codes.clear();
    train_codes.shrink_to_fit();

    igp::IGPIndex index;
    index.dim = dim;
    index.nbits = nbits;
    index.n_centroids = n_centroids;
    index.n_docs = n_docs;
    index.n_tokens = n_tokens;
    index.packed_dim = igp::packed_residual_dim(dim, nbits);
    index.pg_m = pg_m;
    index.ef_construction = ef_construction;
    index.input_doc_bundle = args.doc_bundle;
    index.centroids = std::move(centroids);
    index.bucket_cutoffs = std::move(stats.cutoffs);
    index.bucket_weights = std::move(stats.weights);
    index.doclens = bundle_doclens(doc_bundle);
    index.offsets = bundle_offsets(doc_bundle);
    index.assignments.assign(n_tokens, 0);
    index.residuals.assign(n_tokens * index.packed_dim, 0);

    std::cerr << "[igp-indexer] assigning all tokens and packing residual SQ codes...\n";
    assign_and_compress_all_tokens(doc_bundle, index.centroids, index.bucket_cutoffs,
                                   n_centroids, nbits, assign_batch,
                                   index.assignments, index.residuals);

    std::cerr << "[igp-indexer] building centroid-to-pid inverted file...\n";
    index.centroid_to_pids = build_centroid_to_pids(doc_bundle, index.assignments, n_centroids);

    std::cerr << "[igp-indexer] building centroid proximity graph...\n";
    baseline::Timer pg_timer;
    pg_timer.start();
    baseline::HnswlibIndex pg(dim, pg_m, ef_construction);
    pg.build(index.centroids.data(), (std::size_t)n_centroids, dim);
    pg_timer.stop();
    double pg_build_sec = pg_timer.elapsed_sec();

    build_timer.stop();
    index.build_time_sec = build_timer.elapsed_sec();
    index.save(args.index_dir, pg, {
        {"profile", profile},
        {"ann_backend", "hnswlib_centroid_pg"},
        {"centroid_method", "faiss_kmeans"},
        {"centroid_rule", "paper_16sqrtNv_upstream_pow2_floor"},
        {"centroid_build_time_sec", std::to_string(centroid_build_sec)},
        {"kmeans_backend", "faiss_cpu"},
        {"kmeans_metric", "l2_then_normalize_centroids"},
        {"kmeans_iter", std::to_string(kmeans_niters)},
        {"kmeans_niters", std::to_string(kmeans_niters)},
        {"kmeans_seed", std::to_string(kmeans_seed)},
        {"sample_doc_strategy", "upstream_random_docs"},
        {"sample_doc_factor", std::to_string(sample_doc_factor)},
        {"typical_doclen", std::to_string(typical_doclen)},
        {"sample_docs", std::to_string(sample_docs)},
        {"sample_seed", std::to_string(sample_seed)},
        {"sampled_train_tokens", std::to_string(train_samples)},
        {"train_samples", std::to_string(train_samples)},
        {"max_train_tokens", std::to_string(max_train_tokens)},
        {"assign_batch", std::to_string(assign_batch)},
        {"M", std::to_string(pg_m)},
        {"B", std::to_string(nbits)},
        {"seed", std::to_string(kmeans_seed)},
        {"pg_backend", "hnswlib_inner_product"},
        {"pg_build_time_sec", std::to_string(pg_build_sec)},
        {"paper_pg_M", "64"},
        {"paper_pg_efConstruction", "200"},
        {"paper_B", "2"},
        {"avg_abs_residual", std::to_string(stats.avg_abs_residual)},
        {"formal_baseline", "true"},
        {"shared_doc_store", "none"},
    });

    std::cerr << "[igp-indexer] done in " << index.build_time_sec
              << "s; logical bytes=" << igp::logical_index_bytes(args.index_dir) << "\n";
    return 0;
}
