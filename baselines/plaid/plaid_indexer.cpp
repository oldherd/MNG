// PLAID/ColBERTv2 vector-search indexer over a shared .mvbundle input.
//
// This implements the PLAID storage path used by ColBERTv2: FAISS k-means
// centroids, per-token centroid ids, quantized residuals, and centroid->pid IVF.

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/fp16_utils.hpp"
#include "common/timer.hpp"
#include "plaid_index.hpp"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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
    for (int d = 0; d < dim; ++d) {
        dst[d] = baseline::fp16_to_fp32(src[d]);
    }
}

void l2_normalize(float* x, int dim) {
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) ss += (double)x[d] * (double)x[d];
    double inv = 1.0 / std::sqrt(std::max(ss, 1e-30));
    for (int d = 0; d < dim; ++d) x[d] = (float)((double)x[d] * inv);
}

void normalize_rows(std::vector<float>& rows, int dim) {
    const std::size_t n = rows.size() / (std::size_t)dim;
    for (std::size_t i = 0; i < n; ++i) {
        l2_normalize(rows.data() + i * (std::size_t)dim, dim);
    }
}

int default_num_partitions(std::size_t n_tokens) {
    if (n_tokens == 0) return 0;
    double target = 16.0 * std::sqrt((double)n_tokens);
    int pow2 = 1;
    while ((double)(pow2 * 2) <= target && pow2 < (1 << 30)) pow2 *= 2;
    pow2 = std::max(1, pow2);
    return (int)std::min<std::size_t>((std::size_t)pow2, n_tokens);
}

std::vector<float> sample_tokens(const mv::MVBundle& doc_bundle,
                                 std::size_t sample_n,
                                 int seed) {
    const int dim = doc_bundle.dim();
    const std::size_t n_tokens = doc_bundle.n_tokens();
    if (n_tokens == 0 || sample_n == 0) return {};

    std::vector<float> sample(sample_n * (std::size_t)dim);
    const uint16_t* all = doc_bundle.embeddings_fp16();

    std::mt19937_64 rng((uint64_t)seed);
    std::uniform_int_distribution<std::size_t> dist(0, n_tokens - 1);
    for (std::size_t i = 0; i < sample_n; ++i) {
        std::size_t token_id = dist(rng);
        fp16_token_to_fp32(all + token_id * (std::size_t)dim, dim,
                           sample.data() + i * (std::size_t)dim);
    }
    return sample;
}

std::vector<float> train_faiss_kmeans(const std::vector<float>& train,
                                      int dim,
                                      int n_centroids,
                                      int niter,
                                      int seed) {
    if (train.empty()) throw std::runtime_error("empty PLAID k-means sample");
    const std::size_t n_train = train.size() / (std::size_t)dim;
    if (n_train < (std::size_t)n_centroids) {
        throw std::runtime_error("PLAID k-means requires train_samples >= n_centroids");
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

float quantile_sorted(const std::vector<float>& sorted, double q) {
    if (sorted.empty()) return 0.0f;
    q = std::max(0.0, std::min(1.0, q));
    double pos = q * (double)(sorted.size() - 1);
    std::size_t lo = (std::size_t)std::floor(pos);
    std::size_t hi = (std::size_t)std::ceil(pos);
    if (lo == hi) return sorted[lo];
    double t = pos - (double)lo;
    return (float)((1.0 - t) * sorted[lo] + t * sorted[hi]);
}

struct BucketStats {
    std::vector<float> cutoffs;
    std::vector<float> weights;
    double avg_abs_residual = 0.0;
};

BucketStats compute_bucket_stats(const std::vector<float>& heldout,
                                 const std::vector<float>& centroids,
                                 int dim,
                                 int n_centroids,
                                 int nbits,
                                 int batch_size) {
    const std::size_t n_heldout = heldout.size() / (std::size_t)dim;
    if (n_heldout == 0) throw std::runtime_error("empty PLAID heldout sample");

    faiss::IndexFlatIP centroid_index(dim);
    centroid_index.add((faiss::idx_t)n_centroids, centroids.data());

    std::vector<float> residual_values;
    residual_values.reserve(n_heldout * (std::size_t)dim);
    double abs_sum = 0.0;

    std::vector<float> distances((std::size_t)batch_size);
    std::vector<faiss::idx_t> labels((std::size_t)batch_size);
    for (std::size_t start = 0; start < n_heldout; start += (std::size_t)batch_size) {
        std::size_t count = std::min<std::size_t>((std::size_t)batch_size, n_heldout - start);
        centroid_index.search((faiss::idx_t)count,
                              heldout.data() + start * (std::size_t)dim,
                              1, distances.data(), labels.data());
        for (std::size_t i = 0; i < count; ++i) {
            const float* x = heldout.data() + (start + i) * (std::size_t)dim;
            const float* c = centroids.data() + (std::size_t)labels[i] * (std::size_t)dim;
            for (int d = 0; d < dim; ++d) {
                float r = x[d] - c[d];
                residual_values.push_back(r);
                abs_sum += std::fabs((double)r);
            }
        }
    }

    std::sort(residual_values.begin(), residual_values.end());
    int num_options = 1 << nbits;
    BucketStats stats;
    stats.cutoffs.reserve((std::size_t)num_options - 1u);
    stats.weights.reserve((std::size_t)num_options);
    for (int i = 1; i < num_options; ++i) {
        stats.cutoffs.push_back(quantile_sorted(residual_values, (double)i / (double)num_options));
    }
    for (int i = 0; i < num_options; ++i) {
        stats.weights.push_back(quantile_sorted(residual_values,
            ((double)i + 0.5) / (double)num_options));
    }
    stats.avg_abs_residual = abs_sum / (double)residual_values.size();
    return stats;
}

uint8_t bucketize_residual(float residual, const std::vector<float>& cutoffs) {
    uint8_t bucket = 0;
    while (bucket < cutoffs.size() && residual > cutoffs[bucket]) ++bucket;
    return bucket;
}

void pack_bucket(uint8_t* row, int dim_index, int nbits, uint8_t bucket) {
    const int bit_pos = dim_index * nbits;
    const int byte_index = bit_pos / 8;
    const int offset_in_byte = bit_pos % 8;
    const int shift = 8 - offset_in_byte - nbits;
    row[byte_index] |= (uint8_t)(bucket << shift);
}

void compress_all_tokens(const mv::MVBundle& doc_bundle,
                         const std::vector<float>& centroids,
                         const std::vector<float>& cutoffs,
                         int n_centroids,
                         int nbits,
                         int batch_size,
                         std::vector<int32_t>& codes,
                         std::vector<uint8_t>& residuals) {
    const int dim = doc_bundle.dim();
    const std::size_t n_tokens = doc_bundle.n_tokens();
    const std::size_t packed_dim = plaid::packed_residual_dim(dim, nbits);
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
            codes[start + i] = code;
            const float* x = batch.data() + i * (std::size_t)dim;
            const float* c = centroids.data() + (std::size_t)code * (std::size_t)dim;
            uint8_t* packed = residuals.data() + (start + i) * packed_dim;
            std::fill(packed, packed + packed_dim, 0);
            for (int d = 0; d < dim; ++d) {
                uint8_t bucket = bucketize_residual(x[d] - c[d], cutoffs);
                pack_bucket(packed, d, nbits, bucket);
            }
        }

        if (start == 0 || ((start / (std::size_t)batch_size) % 20 == 0)) {
            std::cerr << "[plaid-indexer] compressed " << (start + count)
                      << "/" << n_tokens << " tokens\n";
        }
    }
}

std::vector<std::vector<int32_t>> build_ivf_pid_lists(const std::vector<int32_t>& codes,
                                                       const std::vector<int64_t>& offsets,
                                                       int n_centroids) {
    std::vector<std::vector<int32_t>> ivf((std::size_t)n_centroids);
    const std::size_t n_docs = offsets.size() - 1;
    std::vector<int32_t> unique_codes;

    for (std::size_t pid = 0; pid < n_docs; ++pid) {
        std::size_t start = (std::size_t)offsets[pid];
        std::size_t end = (std::size_t)offsets[pid + 1];
        unique_codes.clear();
        unique_codes.reserve(end - start);
        for (std::size_t t = start; t < end; ++t) unique_codes.push_back(codes[t]);
        std::sort(unique_codes.begin(), unique_codes.end());
        unique_codes.erase(std::unique(unique_codes.begin(), unique_codes.end()), unique_codes.end());
        for (int32_t code : unique_codes) {
            if (code >= 0 && code < n_centroids) ivf[(std::size_t)code].push_back((int32_t)pid);
        }
        if (pid == 0 || (pid % 100000 == 0)) {
            std::cerr << "[plaid-indexer] IVF pid " << pid << "/" << n_docs << "\n";
        }
    }
    return ivf;
}

}  // namespace

int main(int argc, char** argv) {
    auto args = baseline::parse_args(argc, argv);
    if (args.doc_bundle.empty() || args.index_dir.empty()) {
        std::cerr << "Usage: indexer_plaid --doc <bundle> --index_dir <dir> "
                  << "[--param nbits=2] [--param n_centroids=0] "
                  << "[--param kmeans_niters=4] [--param sample_factor=100] "
                  << "[--param heldout_max=50000] [--param seed=123] "
                  << "[--param assign_batch=65536]\n";
        return 1;
    }

    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    if (doc_bundle.dtype() != mv::DType::Float16) {
        throw std::runtime_error("PLAID indexer expects fp16 doc .mvbundle input");
    }

    const int dim = doc_bundle.dim();
    const std::size_t n_docs = doc_bundle.n_docs();
    const std::size_t n_tokens = doc_bundle.n_tokens();

    int nbits = baseline::get_extra_int(args, "nbits", 2);
    int n_centroids = baseline::get_extra_int(args, "n_centroids", 0);
    int kmeans_niters = baseline::get_extra_int(args, "kmeans_niters", 4);
    int sample_factor = baseline::get_extra_int(args, "sample_factor", 100);
    int heldout_max = baseline::get_extra_int(args, "heldout_max", 50000);
    int seed = baseline::get_extra_int(args, "seed", 123);
    int assign_batch = baseline::get_extra_int(args, "assign_batch", 65536);
    std::string profile = get_extra_string(args, "profile", "paper_lotte_colbertv2");

    if (nbits <= 0 || nbits > 8 || (8 % nbits) != 0) {
        throw std::runtime_error("PLAID nbits must divide 8 and be in [1,8]");
    }
    if (dim % 8 != 0 || dim % (nbits * 8) != 0) {
        throw std::runtime_error("PLAID residual packing expects dim % (nbits*8) == 0");
    }
    if (n_tokens == 0) throw std::runtime_error("empty doc bundle");
    if (n_centroids == 0) n_centroids = default_num_partitions(n_tokens);
    n_centroids = std::max(1, std::min<int>(n_centroids, (int)n_tokens));
    assign_batch = std::max(1, assign_batch);

    const std::size_t train_target = std::max<std::size_t>(
        (std::size_t)n_centroids,
        std::min<std::size_t>(n_tokens, (std::size_t)sample_factor * (std::size_t)n_centroids));
    const std::size_t heldout_target = std::min<std::size_t>(
        (std::size_t)heldout_max,
        std::max<std::size_t>(1, train_target / 20));

    std::cerr << "[plaid-indexer] docs=" << n_docs << " tokens=" << n_tokens
              << " dim=" << dim << " n_centroids=" << n_centroids
              << " nbits=" << nbits << " kmeans_niters=" << kmeans_niters
              << " train_samples=" << train_target << " heldout=" << heldout_target
              << " profile=" << profile << "\n";

    baseline::Timer build_timer;
    build_timer.start();

    std::cerr << "[plaid-indexer] sampling training tokens...\n";
    std::vector<float> train = sample_tokens(doc_bundle, train_target, seed);

    std::cerr << "[plaid-indexer] training FAISS k-means...\n";
    baseline::Timer kmeans_timer;
    kmeans_timer.start();
    std::vector<float> centroids = train_faiss_kmeans(
        train, dim, n_centroids, kmeans_niters, seed);
    kmeans_timer.stop();
    double centroid_build_sec = kmeans_timer.elapsed_sec();
    train.clear();
    train.shrink_to_fit();

    std::cerr << "[plaid-indexer] sampling heldout tokens for residual buckets...\n";
    std::vector<float> heldout = sample_tokens(doc_bundle, heldout_target, seed + 1);
    std::cerr << "[plaid-indexer] computing residual bucket cutoffs/weights...\n";
    BucketStats stats = compute_bucket_stats(
        heldout, centroids, dim, n_centroids, nbits, assign_batch);
    heldout.clear();
    heldout.shrink_to_fit();

    plaid::PLAIDIndex index;
    index.dim = dim;
    index.nbits = nbits;
    index.n_centroids = n_centroids;
    index.n_docs = n_docs;
    index.n_tokens = n_tokens;
    index.packed_dim = plaid::packed_residual_dim(dim, nbits);
    index.input_doc_bundle = args.doc_bundle;
    index.centroids = std::move(centroids);
    index.bucket_cutoffs = std::move(stats.cutoffs);
    index.bucket_weights = std::move(stats.weights);

    index.doclens.assign(doc_bundle.doclens(), doc_bundle.doclens() + n_docs);
    index.offsets.assign(doc_bundle.offsets(), doc_bundle.offsets() + n_docs + 1);
    index.codes.assign(n_tokens, 0);
    index.residuals.assign(n_tokens * index.packed_dim, 0);

    std::cerr << "[plaid-indexer] assigning centroids and packing residuals...\n";
    compress_all_tokens(doc_bundle, index.centroids, index.bucket_cutoffs,
                        n_centroids, nbits, assign_batch,
                        index.codes, index.residuals);

    std::cerr << "[plaid-indexer] building centroid->pid IVF lists...\n";
    index.ivf_pid_lists = build_ivf_pid_lists(index.codes, index.offsets, n_centroids);

    build_timer.stop();
    index.build_time_sec = build_timer.elapsed_sec();

    index.save(args.index_dir, {
        {"profile", profile},
        {"ann_backend", "ivf_pid_lists"},
        {"num_partitions_rule", "2^floor(log2(16*sqrt(n_tokens))) unless overridden"},
        {"centroid_method", "kmeans"},
        {"centroid_build_time_sec", std::to_string(centroid_build_sec)},
        {"kmeans_backend", "faiss_cpu"},
        {"kmeans_metric", "l2_then_normalize_centroids"},
        {"kmeans_iter", std::to_string(kmeans_niters)},
        {"kmeans_niters", std::to_string(kmeans_niters)},
        {"sample_factor", std::to_string(sample_factor)},
        {"train_samples", std::to_string(train_target)},
        {"heldout_samples", std::to_string(heldout_target)},
        {"sample_seed", std::to_string(seed)},
        {"seed", std::to_string(seed)},
        {"avg_abs_residual", std::to_string(stats.avg_abs_residual)},
        {"formal_baseline", "true"},
        {"shared_doc_store", "none"},
    });

    std::cerr << "[plaid-indexer] done in " << index.build_time_sec
              << "s; logical bytes=" << plaid::logical_index_bytes(args.index_dir) << "\n";
    return 0;
}
