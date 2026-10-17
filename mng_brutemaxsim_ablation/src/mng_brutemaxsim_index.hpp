#pragma once

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/eval.hpp"
#include "common/fp16_utils.hpp"
#include "common/index_io.hpp"
#include "common/search_runtime.hpp"
#include "common/timer.hpp"
#include "brutemaxsim_score.hpp"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <omp.h>

#if defined(MNG_BACKEND_AVX512)
#include <immintrin.h>
#elif defined(MNG_BACKEND_AVX2)
#include <immintrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mng {

constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();
// MNG v1 uses standard top-1 MaxSim semantics. Aggregated gamma-NN
// scoring from MV-HNSW/XTR is intentionally not part of the official path.

#ifndef MNG_INDEX_THREADS
#define MNG_INDEX_THREADS 1
#endif

#ifndef MNG_QUERY_THREADS
#define MNG_QUERY_THREADS 1
#endif

#if defined(MNG_MODE_DEBUG) && defined(MNG_MODE_EXP)
#error "Define only one MNG mode macro"
#endif

inline const char* build_mode_string() {
#if defined(MNG_MODE_DEBUG)
    return "debug";
#elif defined(MNG_MODE_EXP)
    return "exp";
#else
    return "normal";
#endif
}

inline const char* backend_string() {
#if defined(MNG_BACKEND_AVX512)
    return "avx512";
#elif defined(MNG_BACKEND_AVX2)
    return "avx2";
#else
    return "scalar";
#endif
}

inline double now_sec() {
    return omp_get_wtime();
}

template <typename T, size_t Align = 64>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t n) { resize_uninitialized(n); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept { move_from(other); }
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }
    ~AlignedBuffer() { reset(); }

    void resize_uninitialized(size_t n) {
        if (n <= capacity_) {
            size_ = n;
            return;
        }
        reset();
        if (n == 0) return;
        void* p = nullptr;
        const int rc = posix_memalign(&p, Align, n * sizeof(T));
        if (rc != 0 || p == nullptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<T*>(p);
        size_ = n;
        capacity_ = n;
    }

    void reset() {
        std::free(data_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    T& operator[](size_t i) noexcept { return data_[i]; }
    const T& operator[](size_t i) const noexcept { return data_[i]; }

private:
    void move_from(AlignedBuffer& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    T* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
};

template <typename T>
inline void write_aligned_bin(const std::string& path, const AlignedBuffer<T>& buf) {
    baseline::write_bin(path, buf.data(), buf.size() * sizeof(T));
}

template <typename T>
inline void read_aligned_bin_exact(const std::string& path, AlignedBuffer<T>& buf, size_t count) {
    const size_t bytes = baseline::file_size(path);
    if (bytes != count * sizeof(T)) {
        throw std::runtime_error("MNG artifact size mismatch: " + path);
    }
    buf.resize_uninitialized(count);
    baseline::read_bin(path, buf.data(), bytes);
}

inline void ensure_dir2(const std::string& dir) {
    baseline::ensure_dir(dir);
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint32_t sqrt_count(uint32_t n) {
    return std::max<uint32_t>(1, static_cast<uint32_t>(std::floor(std::sqrt((double)n))));
}

inline uint32_t beta_for_target(uint32_t n) {
    return std::min<uint32_t>(n, sqrt_count(n));
}

inline void normalize_row(float* x, uint32_t dim) {
    double ss = 0.0;
    for (uint32_t i = 0; i < dim; ++i) ss += (double)x[i] * (double)x[i];
    if (ss <= 0.0) return;
    const float inv = 1.0f / std::sqrt((float)ss);
    for (uint32_t i = 0; i < dim; ++i) x[i] *= inv;
}

inline float dot_row(const float* a, const float* b, uint32_t dim) {
#if defined(MNG_BACKEND_AVX512)
    if (dim == 128) {
        __m512 acc = _mm512_setzero_ps();
        for (int i = 0; i < 128; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        return _mm512_reduce_add_ps(acc);
    }
#elif defined(MNG_BACKEND_AVX2)
    if (dim == 128) {
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < 128; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
#endif
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

struct DocTable {
    uint32_t num_docs = 0;
    uint32_t dim = 0;
    uint64_t total_tokens = 0;
    uint32_t max_doclen = 0;
    AlignedBuffer<uint64_t> token_offset;
    AlignedBuffer<uint32_t> token_count;
    AlignedBuffer<uint64_t> external_doc_ids;
};

struct TokenStore {
    AlignedBuffer<float> tokens;
};

struct CentroidStore {
    uint64_t total_centroids = 0;
    uint32_t max_centroids_per_doc = 0;
    AlignedBuffer<uint64_t> centroid_offset;
    AlignedBuffer<uint16_t> centroid_count;
    AlignedBuffer<float> centroids;
    AlignedBuffer<uint64_t> group_offsets;
    AlignedBuffer<uint16_t> token_order;
};

struct NavLayer {
    uint32_t capacity = 0;
    AlignedBuffer<uint32_t> neighbors;
    AlignedBuffer<uint16_t> degree;
};

struct NavGraph {
    uint32_t num_anchors = 0;
    uint32_t M_nav = 0;
    uint32_t maxM_nav = 0;
    uint32_t maxM0_nav = 0;
    uint32_t entry_anchor = kInvalidId;
    uint16_t max_level = 0;
    AlignedBuffer<uint32_t> anchor_doc_ids;
    AlignedBuffer<uint16_t> levels;
    std::vector<NavLayer> layers;
};

struct AssignmentStore {
    uint32_t num_docs = 0;
    uint32_t assignment_gamma_eff = 0;
    AlignedBuffer<uint32_t> primary_anchor;
    AlignedBuffer<uint16_t> degree;
    AlignedBuffer<uint32_t> anchor_ids;
    AlignedBuffer<float> anchor_scores;
};

struct BucketStore {
    uint32_t num_anchors = 0;
    AlignedBuffer<uint64_t> offsets;
    AlignedBuffer<uint32_t> doc_ids;
};

struct ProxGraph {
    uint32_t num_docs = 0;
    uint32_t M0 = 0;
    AlignedBuffer<uint32_t> neighbors;
    AlignedBuffer<uint16_t> degree;
#if defined(MNG_MODE_DEBUG)
    AlignedBuffer<float> scores;
#endif
};

struct HeapItem {
    float score = 0.0f;
    uint32_t id = 0;
};

struct MaxHeapBetter {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.score != b.score) return a.score < b.score;
        return a.id > b.id;
    }
};

struct MinHeapBetter {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
    }
};

template <typename Compare>
struct ReusableHeap {
    std::vector<HeapItem> data;
    Compare comp;
    void reserve(size_t n) { data.reserve(n); }
    void clear() { data.clear(); }
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
    const HeapItem& top() const { return data.front(); }
    void push(const HeapItem& x) {
        data.push_back(x);
        std::push_heap(data.begin(), data.end(), comp);
    }
    HeapItem pop() {
        std::pop_heap(data.begin(), data.end(), comp);
        HeapItem x = data.back();
        data.pop_back();
        return x;
    }
};

struct BuildWorkerScratch {
    AlignedBuffer<uint32_t> candidate_ids;
    AlignedBuffer<float> candidate_scores;
    AlignedBuffer<uint32_t> candidate_order;
    AlignedBuffer<float> row_scores;
    AlignedBuffer<uint16_t> row_order;
    AlignedBuffer<uint16_t> top_beta_ids;
    AlignedBuffer<uint16_t> visited_anchor;
    uint16_t visited_anchor_tag = 1;
    ReusableHeap<MaxHeapBetter> nav_candidates;
    ReusableHeap<MinHeapBetter> nav_top;
    AlignedBuffer<uint32_t> result_ids;
    AlignedBuffer<float> result_scores;
    AlignedBuffer<uint32_t> selector_order;
    AlignedBuffer<float> selector_scores;
    uint64_t local_edges = 0;
    uint64_t local_underfilled = 0;
    uint64_t local_candidates = 0;
    uint64_t local_symmetric_pairs = 0;
    double local_forward_sec = 0.0;
    double local_reverse_sec = 0.0;
};

struct SearchScratch {
    uint32_t query_token_count = 0;
    uint16_t query_centroid_count = 0;
    AlignedBuffer<float> query_tokens;
    AlignedBuffer<float> query_centroids;
    AlignedBuffer<uint64_t> query_group_offsets;
    AlignedBuffer<uint16_t> query_token_order;
    AlignedBuffer<uint16_t> query_labels;
    AlignedBuffer<uint32_t> query_cluster_counts;
    AlignedBuffer<uint32_t> query_cluster_cursor;
    AlignedBuffer<float> row_scores;
    AlignedBuffer<uint16_t> row_order;
    AlignedBuffer<uint16_t> top_beta_ids;
    AlignedBuffer<uint16_t> visited_anchor;
    uint16_t visited_anchor_tag = 1;
    AlignedBuffer<uint16_t> visited_doc;
    uint16_t visited_doc_tag = 1;
    ReusableHeap<MaxHeapBetter> nav_candidates;
    ReusableHeap<MinHeapBetter> nav_top;
    ReusableHeap<MaxHeapBetter> candidate_queue;
    ReusableHeap<MinHeapBetter> top_candidates;
    AlignedBuffer<uint32_t> result_ids;
    AlignedBuffer<float> result_scores;
    AlignedBuffer<uint32_t> nav_result_ids;
    AlignedBuffer<float> nav_result_scores;
    AlignedBuffer<uint32_t> selector_order;
    AlignedBuffer<float> selector_scores;
};

struct BuildConfig {
    std::string doc_bundle;
    std::string index_dir;
    uint32_t M = 32;
    uint32_t M0 = 64;
    uint32_t assignment_gamma = 64;
    uint32_t anchor_ratio = 32;
    uint32_t ef_nav_build = 128;
    uint32_t ef_nav_assign = 128;
    uint64_t seed = 42;
    uint32_t kmeans_niter = 25;
    uint32_t kmeans_nredo = 5;
    std::string prox_builder = "nearest_top_m0";
    bool nav_only = false;
};

struct SearchConfig {
    std::string doc_bundle;
    std::string query_bundle;
    std::string gt_dir;
    std::string index_dir;
    std::string output;
    uint32_t topk = 1000;
    uint32_t ef_nav_query = 128;
    uint32_t ef_search = 1000;
    bool nav_only = false;
    bool exact_score_lookup = false;
};

struct BuildStats {
    double read_sec = 0.0;
    double fp32_materialize_sec = 0.0;
    double centroid_build_sec = 0.0;
    double nav_build_sec = 0.0;
    double assignment_sec = 0.0;
    double bucket_build_sec = 0.0;
    double candidate_bound_sec = 0.0;
    double prox_build_sec = 0.0;
    double prox_score_forward_sec = 0.0;
    double prox_score_reverse_sec = 0.0;
    double save_sec = 0.0;
    double total_build_sec = 0.0;
    double core_build_sec = 0.0;
    uint64_t candidate_mean = 0;
    uint64_t candidate_p50 = 0;
    uint64_t candidate_p95 = 0;
    uint64_t candidate_p99 = 0;
    uint64_t candidate_max = 0;
    uint64_t selected_edges = 0;
    uint64_t underfilled_rows = 0;
    uint64_t rank_reverse_edges = 0;
    uint64_t rank_reverse_max = 0;
    uint64_t rank_pruned_rows = 0;
};

struct SearchStats {
    double load_index_sec = 0.0;
    double read_doc_sec = 0.0;
    double fp32_materialize_sec = 0.0;
    double query_precompute_sec = 0.0;
    double nav_query_sec = 0.0;
    double prox_search_sec = 0.0;
    double eval_sec = 0.0;
    double total_search_sec = 0.0;
    double per_query_ms = 0.0;
};

inline BuildConfig parse_build_config(const baseline::BenchArgs& args) {
    BuildConfig cfg;
    cfg.doc_bundle = args.doc_bundle;
    cfg.index_dir = args.index_dir;
    cfg.M = (uint32_t)baseline::get_extra_int(args, "M", 32);
    cfg.M0 = (uint32_t)baseline::get_extra_int(args, "M0", (int)(2 * cfg.M));
    const int legacy_assignment_gamma =
        baseline::get_extra_int(args, "gamma", (int)cfg.M0);
    cfg.assignment_gamma =
        (uint32_t)baseline::get_extra_int(args, "assignment_gamma", legacy_assignment_gamma);
    cfg.anchor_ratio = (uint32_t)baseline::get_extra_int(args, "anchor_ratio", (int)cfg.M);
    cfg.ef_nav_build = (uint32_t)baseline::get_extra_int(args, "ef_nav_build", 128);
    cfg.ef_nav_assign = (uint32_t)baseline::get_extra_int(args, "ef_nav_assign", 128);
    cfg.seed = (uint64_t)baseline::get_extra_int(args, "seed", 42);
    cfg.kmeans_niter = (uint32_t)baseline::get_extra_int(args, "kmeans_niter", 25);
    cfg.kmeans_nredo = (uint32_t)baseline::get_extra_int(args, "kmeans_nredo", 5);
    cfg.prox_builder = baseline::get_extra_string(args, "prox_builder", "nearest_top_m0");
    if (cfg.prox_builder == "rank_mutual_v1") cfg.prox_builder = "rank_mutual";
    cfg.nav_only = baseline::get_extra_int(args, "nav_only", 0) != 0;
    if (cfg.M == 0 || cfg.M0 == 0 || cfg.assignment_gamma == 0 || cfg.anchor_ratio == 0 ||
        cfg.ef_nav_build == 0 || cfg.ef_nav_assign == 0) {
        throw std::runtime_error("MNG build params must be positive");
    }
    if (cfg.prox_builder != "nearest_top_m0" && cfg.prox_builder != "rank_mutual") {
        throw std::runtime_error("unsupported prox_builder: " + cfg.prox_builder);
    }
    if (cfg.prox_builder == "rank_mutual" && cfg.M > cfg.M0) {
        throw std::runtime_error("rank_mutual prox_builder requires M <= M0");
    }
    return cfg;
}

inline SearchConfig parse_search_config(const baseline::BenchArgs& args) {
    SearchConfig cfg;
    cfg.doc_bundle = args.doc_bundle;
    cfg.query_bundle = args.query_bundle;
    cfg.gt_dir = args.gt_dir;
    cfg.index_dir = args.index_dir;
    cfg.output = args.output;
    cfg.topk = (uint32_t)args.k;
    cfg.ef_nav_query = (uint32_t)baseline::get_extra_int(args, "ef_nav_query", 128);
    cfg.ef_search = (uint32_t)baseline::get_extra_int(args, "ef_search", (int)cfg.topk);
    cfg.nav_only = baseline::get_extra_int(args, "nav_only", 0) != 0;
    cfg.exact_score_lookup =
        baseline::get_extra_int(args, "exact_score", 0) != 0 ||
        baseline::get_extra_int(args, "exact_score_lookup", 0) != 0;
    if (cfg.topk == 0 || cfg.ef_nav_query == 0 || cfg.ef_search < cfg.topk) {
        throw std::runtime_error("MNG search requires ef_search >= k > 0");
    }
    return cfg;
}

inline void load_doc_bundle_fp32(const mv::MVBundle& bundle,
                                 DocTable& docs,
                                 TokenStore& store) {
    if (bundle.is_query() || bundle.dtype() != mv::DType::Float16) {
        throw std::runtime_error("MNG expects fp16 doc .mvbundle input");
    }
    docs.num_docs = (uint32_t)bundle.n_docs();
    docs.dim = (uint32_t)bundle.dim();
    docs.total_tokens = (uint64_t)bundle.n_tokens();
    docs.token_offset.resize_uninitialized((size_t)docs.num_docs + 1);
    docs.token_count.resize_uninitialized(docs.num_docs);
    docs.external_doc_ids.resize_uninitialized(docs.num_docs);

    const int32_t* lens = bundle.doclens();
    const int32_t* ids = bundle.docids();
    docs.max_doclen = 0;
    uint64_t off = 0;
    for (uint32_t i = 0; i < docs.num_docs; ++i) {
        if (lens[i] <= 0) throw std::runtime_error("MNG does not support empty documents");
        if (lens[i] > 65535) throw std::runtime_error("MNG first version requires doclen <= 65535");
        docs.token_offset[i] = off;
        docs.token_count[i] = (uint32_t)lens[i];
        docs.external_doc_ids[i] = (uint64_t)(uint32_t)ids[i];
        docs.max_doclen = std::max<uint32_t>(docs.max_doclen, (uint32_t)lens[i]);
        off += (uint64_t)lens[i];
    }
    docs.token_offset[docs.num_docs] = off;
    if (off != docs.total_tokens) throw std::runtime_error("doc token count mismatch");

    store.tokens.resize_uninitialized((size_t)docs.total_tokens * docs.dim);
    const uint16_t* src = bundle.embeddings_fp16();
    float* dst = store.tokens.data();
    const size_t total = (size_t)docs.total_tokens * docs.dim;
    #pragma omp parallel for schedule(static) num_threads(MNG_INDEX_THREADS)
    for (size_t i = 0; i < total; ++i) {
        dst[i] = baseline::fp16_to_fp32(src[i]);
    }
}

inline void materialize_query_bundle_fp32(const mv::MVBundle& bundle,
                                          DocTable& queries,
                                          TokenStore& store) {
    if (!bundle.is_query() || bundle.dtype() != mv::DType::Float32) {
        throw std::runtime_error("MNG expects fp32 query .mvbundle input");
    }
    queries.num_docs = (uint32_t)bundle.n_docs();
    queries.dim = (uint32_t)bundle.dim();
    queries.total_tokens = (uint64_t)bundle.n_tokens();
    queries.token_offset.resize_uninitialized((size_t)queries.num_docs + 1);
    queries.token_count.resize_uninitialized(queries.num_docs);
    queries.external_doc_ids.resize_uninitialized(queries.num_docs);
    const int32_t* lens = bundle.doclens();
    const int32_t* ids = bundle.docids();
    uint64_t off = 0;
    queries.max_doclen = 0;
    for (uint32_t i = 0; i < queries.num_docs; ++i) {
        if (lens[i] <= 0) throw std::runtime_error("MNG does not support empty queries");
        if (lens[i] > 65535) throw std::runtime_error("MNG first version requires query len <= 65535");
        queries.token_offset[i] = off;
        queries.token_count[i] = (uint32_t)lens[i];
        queries.external_doc_ids[i] = (uint64_t)(uint32_t)ids[i];
        queries.max_doclen = std::max<uint32_t>(queries.max_doclen, (uint32_t)lens[i]);
        off += (uint64_t)lens[i];
    }
    queries.token_offset[queries.num_docs] = off;
    store.tokens.resize_uninitialized((size_t)queries.total_tokens * queries.dim);
    std::memcpy(store.tokens.data(), bundle.embeddings_fp32(),
                (size_t)queries.total_tokens * queries.dim * sizeof(float));
}

inline void faiss_spherical_kmeans(const float* x,
                                   uint32_t n,
                                   uint32_t dim,
                                   uint32_t k,
                                   uint32_t niter,
                                   uint32_t nredo,
                                   uint64_t seed,
                                   float* centroids_out) {
    if (n == 0 || k == 0 || k > n) throw std::runtime_error("invalid kmeans shape");
    if (k == 1) {
        std::fill(centroids_out, centroids_out + dim, 0.0f);
        for (uint32_t i = 0; i < n; ++i) {
            const float* row = x + (size_t)i * dim;
            for (uint32_t d = 0; d < dim; ++d) centroids_out[d] += row[d];
        }
        for (uint32_t d = 0; d < dim; ++d) centroids_out[d] /= (float)n;
        normalize_row(centroids_out, dim);
        return;
    }
    faiss::Clustering clus((int)dim, (int)k);
    clus.niter = (int)niter;
    clus.nredo = (int)nredo;
    clus.seed = (int)(seed & 0x7fffffffU);
    clus.spherical = true;
    clus.verbose = false;
    clus.min_points_per_centroid = 1;
    clus.max_points_per_centroid = std::max<int>(256, (int)n);
    faiss::IndexFlatIP quantizer(dim);
    clus.train((faiss::idx_t)n, x, quantizer);
    std::memcpy(centroids_out, clus.centroids.data(), (size_t)k * dim * sizeof(float));
    for (uint32_t c = 0; c < k; ++c) normalize_row(centroids_out + (size_t)c * dim, dim);
}

inline void assign_tokens_to_centroids(const float* tokens,
                                       uint32_t n,
                                       uint32_t dim,
                                       const float* centroids,
                                       uint32_t k,
                                       uint16_t* labels) {
    for (uint32_t i = 0; i < n; ++i) {
        const float* row = tokens + (size_t)i * dim;
        float best = -std::numeric_limits<float>::infinity();
        uint32_t best_c = 0;
        for (uint32_t c = 0; c < k; ++c) {
            float s = dot_row(row, centroids + (size_t)c * dim, dim);
            if (s > best) {
                best = s;
                best_c = c;
            }
        }
        labels[i] = (uint16_t)best_c;
    }
}

inline void build_centroids(const DocTable& docs,
                            const TokenStore& tokens,
                            const BuildConfig& cfg,
                            CentroidStore& centroids) {
    const uint32_t n = docs.num_docs;
    const uint32_t dim = docs.dim;
    centroids.centroid_offset.resize_uninitialized((size_t)n + 1);
    centroids.centroid_count.resize_uninitialized(n);
    uint64_t total_c = 0;
    centroids.max_centroids_per_doc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t k = sqrt_count(docs.token_count[i]);
        centroids.centroid_offset[i] = total_c;
        centroids.centroid_count[i] = (uint16_t)k;
        centroids.max_centroids_per_doc = std::max(centroids.max_centroids_per_doc, k);
        total_c += k;
    }
    centroids.centroid_offset[n] = total_c;
    centroids.total_centroids = total_c;
    centroids.centroids.resize_uninitialized((size_t)total_c * dim);
    centroids.group_offsets.resize_uninitialized((size_t)total_c + 1);
    centroids.token_order.resize_uninitialized((size_t)docs.total_tokens);

    const float* all_tokens = tokens.tokens.data();
    #pragma omp parallel num_threads(MNG_INDEX_THREADS)
    {
        std::vector<uint16_t> labels;
        std::vector<uint32_t> counts;
        std::vector<uint32_t> cursor;
        #pragma omp for schedule(dynamic, 128)
        for (uint32_t doc = 0; doc < n; ++doc) {
            const uint64_t tbase = docs.token_offset[doc];
            const uint32_t len = docs.token_count[doc];
            const uint64_t cbase = centroids.centroid_offset[doc];
            const uint32_t k = centroids.centroid_count[doc];
            const float* doc_tokens = all_tokens + (size_t)tbase * dim;
            float* doc_centroids = centroids.centroids.data() + (size_t)cbase * dim;
            faiss_spherical_kmeans(doc_tokens, len, dim, k, cfg.kmeans_niter,
                                   cfg.kmeans_nredo, splitmix64(cfg.seed ^ doc), doc_centroids);

            labels.resize(len);
            counts.assign(k, 0);
            cursor.assign(k, 0);
            assign_tokens_to_centroids(doc_tokens, len, dim, doc_centroids, k, labels.data());
            for (uint32_t i = 0; i < len; ++i) counts[labels[i]]++;

            uint64_t pos = tbase;
            for (uint32_t c = 0; c < k; ++c) {
                centroids.group_offsets[cbase + c] = pos;
                cursor[c] = (uint32_t)(pos - tbase);
                pos += counts[c];
            }
            for (uint32_t i = 0; i < len; ++i) {
                uint32_t c = labels[i];
                uint32_t local_pos = cursor[c]++;
                centroids.token_order[tbase + local_pos] = (uint16_t)i;
            }
        }
    }
    centroids.group_offsets[total_c] = docs.total_tokens;
}

inline void build_query_state(uint32_t qid,
                              const DocTable& queries,
                              const TokenStore& query_tokens,
                              const BuildConfig& kcfg,
                              SearchScratch& scratch) {
    const uint32_t dim = queries.dim;
    const uint64_t qbase = queries.token_offset[qid];
    const uint32_t qlen = queries.token_count[qid];
    const uint32_t k = sqrt_count(qlen);
    scratch.query_token_count = qlen;
    scratch.query_centroid_count = (uint16_t)k;
    scratch.query_tokens.resize_uninitialized((size_t)queries.max_doclen * dim);
    scratch.query_centroids.resize_uninitialized((size_t)sqrt_count(queries.max_doclen) * dim);
    scratch.query_group_offsets.resize_uninitialized((size_t)sqrt_count(queries.max_doclen) + 1);
    scratch.query_token_order.resize_uninitialized(queries.max_doclen);
    scratch.query_labels.resize_uninitialized(queries.max_doclen);
    scratch.query_cluster_counts.resize_uninitialized(sqrt_count(queries.max_doclen));
    scratch.query_cluster_cursor.resize_uninitialized(sqrt_count(queries.max_doclen));
    const float* src = query_tokens.tokens.data() + (size_t)qbase * dim;
    std::memcpy(scratch.query_tokens.data(), src, (size_t)qlen * dim * sizeof(float));
    faiss_spherical_kmeans(scratch.query_tokens.data(), qlen, dim, k,
                           kcfg.kmeans_niter, kcfg.kmeans_nredo,
                           splitmix64(kcfg.seed ^ (0xabc00000ULL + qid)),
                           scratch.query_centroids.data());
    assign_tokens_to_centroids(scratch.query_tokens.data(), qlen, dim,
                               scratch.query_centroids.data(), k, scratch.query_labels.data());
    uint32_t* counts = scratch.query_cluster_counts.data();
    uint32_t* cursor = scratch.query_cluster_cursor.data();
    std::fill(counts, counts + k, 0);
    uint64_t pos = 0;
    for (uint32_t i = 0; i < qlen; ++i) counts[scratch.query_labels[i]]++;
    for (uint32_t c = 0; c < k; ++c) {
        scratch.query_group_offsets[c] = pos;
        cursor[c] = (uint32_t)pos;
        pos += counts[c];
    }
    scratch.query_group_offsets[k] = qlen;
    for (uint32_t i = 0; i < qlen; ++i) {
        uint32_t c = scratch.query_labels[i];
        scratch.query_token_order[cursor[c]++] = (uint16_t)i;
    }
}

inline void build_brutemaxsim_query_state(uint32_t qid,
                                          const DocTable& queries,
                                          const TokenStore& query_tokens,
                                          SearchScratch& scratch) {
    if (qid >= queries.num_docs) {
        throw std::out_of_range("brute MaxSim query id is out of range");
    }
    const uint32_t dim = queries.dim;
    const uint64_t query_base = queries.token_offset[qid];
    const uint32_t query_len = queries.token_count[qid];
    if (query_len == 0 || dim == 0) {
        throw std::invalid_argument("brute MaxSim requires a non-empty query");
    }
    scratch.query_token_count = query_len;
    scratch.query_centroid_count = 0;
    scratch.query_tokens.resize_uninitialized(
        static_cast<size_t>(queries.max_doclen) * dim);
    const float* source =
        query_tokens.tokens.data() + static_cast<size_t>(query_base) * dim;
    std::memcpy(
        scratch.query_tokens.data(),
        source,
        static_cast<size_t>(query_len) * dim * sizeof(float));
}

inline void ensure_build_scratch(BuildWorkerScratch& s,
                                 uint32_t candidate_bound,
                                 uint32_t max_doclen,
                                 uint32_t max_centroids,
                                 uint32_t beta_max,
                                 uint32_t num_anchors,
                                 uint32_t max_ef) {
    s.candidate_ids.resize_uninitialized(candidate_bound);
    s.candidate_scores.resize_uninitialized(candidate_bound);
    s.candidate_order.resize_uninitialized(candidate_bound);
    s.row_scores.resize_uninitialized(max_doclen);
    s.row_order.resize_uninitialized(max_doclen);
    s.top_beta_ids.resize_uninitialized((size_t)max_centroids * beta_max);
    s.visited_anchor.resize_uninitialized(num_anchors);
    std::fill(s.visited_anchor.data(), s.visited_anchor.data() + num_anchors, 0);
    s.nav_candidates.reserve((size_t)max_ef + 8);
    s.nav_top.reserve((size_t)max_ef + 8);
    s.result_ids.resize_uninitialized((size_t)max_ef + 8);
    s.result_scores.resize_uninitialized((size_t)max_ef + 8);
    s.selector_order.resize_uninitialized((size_t)std::max(max_ef + 8, candidate_bound));
    s.selector_scores.resize_uninitialized((size_t)std::max(max_ef + 8, candidate_bound));
}

inline void ensure_search_scratch(SearchScratch& s,
                                  uint32_t query_max_doclen,
                                  uint32_t doc_max_doclen,
                                  uint32_t dim,
                                  uint32_t num_docs,
                                  uint32_t num_anchors,
                                  uint32_t ef_nav_query,
                                  uint32_t ef_search,
                                  uint32_t topk) {
    const uint32_t qmax_centroids = sqrt_count(query_max_doclen);
    const uint32_t beta_max = beta_for_target(doc_max_doclen);
    const uint32_t heap_cap = std::max<uint32_t>({ef_nav_query, ef_search, topk}) + 8;
    s.query_tokens.resize_uninitialized((size_t)query_max_doclen * dim);
    s.query_centroids.resize_uninitialized((size_t)qmax_centroids * dim);
    s.query_group_offsets.resize_uninitialized((size_t)qmax_centroids + 1);
    s.query_token_order.resize_uninitialized(query_max_doclen);
    s.query_labels.resize_uninitialized(query_max_doclen);
    s.query_cluster_counts.resize_uninitialized(qmax_centroids);
    s.query_cluster_cursor.resize_uninitialized(qmax_centroids);
    s.row_scores.resize_uninitialized(doc_max_doclen);
    s.row_order.resize_uninitialized(doc_max_doclen);
    s.top_beta_ids.resize_uninitialized((size_t)qmax_centroids * beta_max);
    s.visited_anchor.resize_uninitialized(num_anchors);
    s.visited_doc.resize_uninitialized(num_docs);
    std::fill(s.visited_anchor.data(), s.visited_anchor.data() + num_anchors, 0);
    std::fill(s.visited_doc.data(), s.visited_doc.data() + num_docs, 0);
    s.nav_candidates.reserve(heap_cap);
    s.nav_top.reserve(heap_cap);
    s.candidate_queue.reserve(heap_cap);
    s.top_candidates.reserve(heap_cap);
    s.result_ids.resize_uninitialized(heap_cap);
    s.result_scores.resize_uninitialized(heap_cap);
    s.nav_result_ids.resize_uninitialized(heap_cap);
    s.nav_result_scores.resize_uninitialized(heap_cap);
    s.selector_order.resize_uninitialized(heap_cap);
    s.selector_scores.resize_uninitialized(heap_cap);
}

template <typename Scratch>
inline void sort_scratch_results_desc(Scratch& scratch, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) scratch.selector_order[i] = i;
    std::sort(scratch.selector_order.data(), scratch.selector_order.data() + count,
              [&](uint32_t a, uint32_t b) {
                  if (scratch.result_scores[a] != scratch.result_scores[b]) {
                      return scratch.result_scores[a] > scratch.result_scores[b];
                  }
                  return scratch.result_ids[a] < scratch.result_ids[b];
              });
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = scratch.selector_order[i];
        scratch.selector_order[i] = scratch.result_ids[idx];
        scratch.selector_scores[i] = scratch.result_scores[idx];
    }
    for (uint32_t i = 0; i < count; ++i) {
        scratch.result_ids[i] = scratch.selector_order[i];
        scratch.result_scores[i] = scratch.selector_scores[i];
    }
}

inline void bump_anchor_visit(BuildWorkerScratch& s, uint32_t n) {
    ++s.visited_anchor_tag;
    if (s.visited_anchor_tag == 0) {
        std::fill(s.visited_anchor.data(), s.visited_anchor.data() + n, 0);
        s.visited_anchor_tag = 1;
    }
}

inline void bump_anchor_visit(SearchScratch& s, uint32_t n) {
    ++s.visited_anchor_tag;
    if (s.visited_anchor_tag == 0) {
        std::fill(s.visited_anchor.data(), s.visited_anchor.data() + n, 0);
        s.visited_anchor_tag = 1;
    }
}

inline float directed_doc_score(uint32_t src,
                                uint32_t dst,
                                const DocTable& docs,
                                const TokenStore& tokens,
                                const CentroidStore& centroids,
                                BuildWorkerScratch& scratch) {
    const uint32_t dim = docs.dim;
    const float* token_data = tokens.tokens.data();
    const float* centroid_data = centroids.centroids.data();
    const uint64_t* token_offset = docs.token_offset.data();
    const uint32_t* token_count = docs.token_count.data();
    const uint64_t* centroid_offset = centroids.centroid_offset.data();
    const uint16_t* centroid_count = centroids.centroid_count.data();
    const uint64_t* group_offsets = centroids.group_offsets.data();
    const uint16_t* token_order = centroids.token_order.data();

    const uint64_t src_tbase = token_offset[src];
    const uint32_t src_len = token_count[src];
    const uint64_t dst_tbase = token_offset[dst];
    const uint32_t dst_len = token_count[dst];
    const uint64_t src_cbase = centroid_offset[src];
    const uint32_t src_k = centroid_count[src];
    const uint32_t beta = beta_for_target(dst_len);
    float total = 0.0f;

    for (uint32_t c = 0; c < src_k; ++c) {
        const float* cent = centroid_data + (size_t)(src_cbase + c) * dim;
        for (uint32_t j = 0; j < dst_len; ++j) {
            scratch.row_scores[j] = dot_row(cent, token_data + (size_t)(dst_tbase + j) * dim, dim);
            scratch.row_order[j] = (uint16_t)j;
        }
        auto cmp_row = [&](uint16_t a, uint16_t b) {
            float sa = scratch.row_scores[a], sb = scratch.row_scores[b];
            if (sa != sb) return sa > sb;
            return a < b;
        };
        if (dst_len > beta) {
            std::nth_element(scratch.row_order.data(),
                             scratch.row_order.data() + beta,
                             scratch.row_order.data() + dst_len,
                             cmp_row);
        }
        uint16_t* top_ids = scratch.top_beta_ids.data() + (size_t)c * beta;
        for (uint32_t j = 0; j < beta; ++j) top_ids[j] = scratch.row_order[j];

        const uint64_t gb = group_offsets[src_cbase + c];
        const uint64_t ge = group_offsets[src_cbase + c + 1];
        for (uint64_t p = gb; p < ge; ++p) {
            const uint16_t local_tok = token_order[p];
            const float* qtok = token_data + (size_t)(src_tbase + local_tok) * dim;
            float best = -std::numeric_limits<float>::infinity();
            for (uint32_t j = 0; j < beta; ++j) {
                const uint32_t dtok = top_ids[j];
                float s = dot_row(qtok, token_data + (size_t)(dst_tbase + dtok) * dim, dim);
                if (s > best) best = s;
            }
            total += best;
        }
    }
    return total / (float)src_len;
}

inline float directed_query_score(uint32_t dst,
                                  const DocTable& docs,
                                  const TokenStore& tokens,
                                  const SearchScratch& q,
                                  SearchScratch& scratch) {
    (void)scratch;
    if (dst >= docs.num_docs) {
        throw std::out_of_range("brute MaxSim destination document id is out of range");
    }
    const uint32_t dim = docs.dim;
    const float* token_data = tokens.tokens.data();
    const uint64_t* token_offset = docs.token_offset.data();
    const uint32_t* token_count = docs.token_count.data();
    const uint64_t dst_tbase = token_offset[dst];
    const uint32_t dst_len = token_count[dst];
    return mng_brutemaxsim_ablation::directed_mean_maxsim(
        q.query_tokens.data(),
        q.query_token_count,
        token_data + static_cast<size_t>(dst_tbase) * dim,
        dst_len,
        dim,
        [](const float* a, const float* b, uint32_t n) {
            return dot_row(a, b, n);
        });
}

inline float symmetric_doc_score(uint32_t a,
                                 uint32_t b,
                                 const DocTable& docs,
                                 const TokenStore& tokens,
                                 const CentroidStore& centroids,
                                 BuildWorkerScratch& scratch) {
    float ab = directed_doc_score(a, b, docs, tokens, centroids, scratch);
    float ba = directed_doc_score(b, a, docs, tokens, centroids, scratch);
    return 0.5f * (ab + ba);
}

inline void score_symmetric_doc_to_docs(uint32_t src,
                                        const uint32_t* candidates,
                                        uint32_t n,
                                        float* out,
                                        const DocTable& docs,
                                        const TokenStore& tokens,
                                        const CentroidStore& centroids,
                                        BuildWorkerScratch& scratch) {
    const double t0 = now_sec();
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = directed_doc_score(src, candidates[i], docs, tokens, centroids, scratch);
    }
    const double t1 = now_sec();
    for (uint32_t i = 0; i < n; ++i) {
        float rev = directed_doc_score(candidates[i], src, docs, tokens, centroids, scratch);
        out[i] = 0.5f * (out[i] + rev);
    }
    const double t2 = now_sec();
    scratch.local_forward_sec += (t1 - t0);
    scratch.local_reverse_sec += (t2 - t1);
    scratch.local_symmetric_pairs += n;
}

inline void init_nav_graph(NavGraph& nav,
                           const DocTable& docs,
                           const BuildConfig& cfg,
                           std::vector<uint32_t>& doc_to_anchor) {
    nav.M_nav = cfg.M;
    nav.maxM_nav = cfg.M;
    nav.maxM0_nav = 2 * cfg.M;
    nav.num_anchors = std::max<uint32_t>(1, (docs.num_docs + cfg.anchor_ratio - 1) / cfg.anchor_ratio);
    nav.num_anchors = std::min<uint32_t>(nav.num_anchors, docs.num_docs);
    std::vector<uint32_t> ids(docs.num_docs);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937_64 rng(splitmix64(cfg.seed ^ 0xA53A));
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(nav.num_anchors);
    std::sort(ids.begin(), ids.end());
    nav.anchor_doc_ids.resize_uninitialized(nav.num_anchors);
    nav.levels.resize_uninitialized(nav.num_anchors);
    doc_to_anchor.assign(docs.num_docs, kInvalidId);
    const double mult = 1.0 / std::log((double)std::max<uint32_t>(2, cfg.M));
    uint16_t max_level = 0;
    for (uint32_t i = 0; i < nav.num_anchors; ++i) {
        nav.anchor_doc_ids[i] = ids[i];
        doc_to_anchor[ids[i]] = i;
        double u = std::generate_canonical<double, 53>(rng);
        if (u <= 0.0) u = 1e-12;
        uint16_t level = (uint16_t)std::floor(-std::log(u) * mult);
        nav.levels[i] = level;
        max_level = std::max(max_level, level);
    }
    nav.max_level = max_level;
    nav.layers.resize((size_t)max_level + 1);
    for (uint32_t l = 0; l <= max_level; ++l) {
        NavLayer& layer = nav.layers[l];
        layer.capacity = (l == 0) ? nav.maxM0_nav : nav.maxM_nav;
        layer.neighbors.resize_uninitialized((size_t)nav.num_anchors * layer.capacity);
        layer.degree.resize_uninitialized(nav.num_anchors);
        std::fill(layer.neighbors.data(),
                  layer.neighbors.data() + (size_t)nav.num_anchors * layer.capacity,
                  kInvalidId);
        std::fill(layer.degree.data(), layer.degree.data() + nav.num_anchors, 0);
    }
}

inline uint32_t select_top_ids(const uint32_t* ids,
                               const float* scores,
                               uint32_t n,
                               uint32_t keep,
                               uint32_t* order,
                               uint32_t* out_ids) {
    keep = std::min(keep, n);
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    auto cmp = [&](uint32_t a, uint32_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return ids[a] < ids[b];
    };
    if (n > keep) std::nth_element(order, order + keep, order + n, cmp);
    std::sort(order, order + keep, cmp);
    for (uint32_t i = 0; i < keep; ++i) out_ids[i] = ids[order[i]];
    return keep;
}

template <typename ScoreFn, typename Scratch>
inline uint32_t search_nav_layer(const NavGraph& nav,
                                 uint32_t entry,
                                 uint32_t layer_id,
                                 uint32_t ef,
                                 ScoreFn score_fn,
                                 Scratch& scratch) {
    bump_anchor_visit(scratch, nav.num_anchors);
    scratch.nav_candidates.clear();
    scratch.nav_top.clear();
    const uint16_t tag = scratch.visited_anchor_tag;
    float s0 = score_fn(entry);
    scratch.nav_candidates.push({s0, entry});
    scratch.nav_top.push({s0, entry});
    scratch.visited_anchor[entry] = tag;
    const NavLayer& layer = nav.layers[layer_id];
    const uint32_t* neighbors = layer.neighbors.data();
    const uint16_t* degree = layer.degree.data();
    const uint32_t cap = layer.capacity;
    while (!scratch.nav_candidates.empty()) {
        HeapItem cur = scratch.nav_candidates.pop();
        float worst = scratch.nav_top.top().score;
        if (scratch.nav_top.size() >= ef && cur.score < worst) break;
        const uint32_t base = cur.id * cap;
        const uint16_t deg = degree[cur.id];
        for (uint16_t i = 0; i < deg; ++i) {
            uint32_t nb = neighbors[base + i];
            if (scratch.visited_anchor[nb] == tag) continue;
            scratch.visited_anchor[nb] = tag;
            float s = score_fn(nb);
            if (scratch.nav_top.size() < ef || s > scratch.nav_top.top().score) {
                scratch.nav_candidates.push({s, nb});
                scratch.nav_top.push({s, nb});
                if (scratch.nav_top.size() > ef) scratch.nav_top.pop();
            }
        }
    }
    uint32_t count = 0;
    while (!scratch.nav_top.empty()) {
        HeapItem x = scratch.nav_top.pop();
        scratch.result_ids[count] = x.id;
        scratch.result_scores[count] = x.score;
        ++count;
    }
    sort_scratch_results_desc(scratch, count);
    return count;
}

inline void rewrite_nav_neighbors(NavGraph& nav,
                                  uint32_t layer_id,
                                  uint32_t owner,
                                  const uint32_t* selected,
                                  uint32_t n) {
    NavLayer& layer = nav.layers[layer_id];
    uint32_t* row = layer.neighbors.data() + (size_t)owner * layer.capacity;
    layer.degree[owner] = (uint16_t)n;
    for (uint32_t i = 0; i < n; ++i) row[i] = selected[i];
    for (uint32_t i = n; i < layer.capacity; ++i) row[i] = kInvalidId;
}

inline void append_or_prune_nav(NavGraph& nav,
                                uint32_t layer_id,
                                uint32_t owner,
                                uint32_t nb,
                                const DocTable& docs,
                                const TokenStore& tokens,
                                const CentroidStore& centroids,
                                BuildWorkerScratch& scratch) {
    if (owner == nb) return;
    NavLayer& layer = nav.layers[layer_id];
    uint32_t* row = layer.neighbors.data() + (size_t)owner * layer.capacity;
    uint16_t deg = layer.degree[owner];
    for (uint16_t i = 0; i < deg; ++i) if (row[i] == nb) return;
    if (deg < layer.capacity) {
        row[deg] = nb;
        layer.degree[owner] = deg + 1;
        return;
    }
    const uint32_t n = deg + 1;
    for (uint32_t i = 0; i < deg; ++i) scratch.selector_order[i] = row[i];
    scratch.selector_order[deg] = nb;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t cand = scratch.selector_order[i];
        uint32_t odoc = nav.anchor_doc_ids[owner];
        uint32_t cdoc = nav.anchor_doc_ids[cand];
        scratch.candidate_scores[i] = symmetric_doc_score(odoc, cdoc, docs, tokens, centroids, scratch);
    }
    select_top_ids(scratch.selector_order.data(), scratch.candidate_scores.data(), n,
                   layer.capacity, scratch.candidate_order.data(), row);
    layer.degree[owner] = (uint16_t)layer.capacity;
}

inline void build_nav_graph(NavGraph& nav,
                            const DocTable& docs,
                            const TokenStore& tokens,
                            const CentroidStore& centroids,
                            const BuildConfig& cfg) {
    BuildWorkerScratch scratch;
    ensure_build_scratch(scratch, std::max<uint32_t>(cfg.ef_nav_build + 8, nav.maxM0_nav + 8),
                         docs.max_doclen, centroids.max_centroids_per_doc,
                         beta_for_target(docs.max_doclen), nav.num_anchors,
                         std::max(cfg.ef_nav_build, cfg.ef_nav_assign) + 8);
    nav.entry_anchor = 0;
    uint16_t current_max = nav.levels[0];
    nav.max_level = current_max;
#if defined(MNG_MODE_EXP) || defined(MNG_MODE_DEBUG)
    const double nav_build_start = now_sec();
#endif
    for (uint32_t a = 1; a < nav.num_anchors; ++a) {
#if defined(MNG_MODE_EXP) || defined(MNG_MODE_DEBUG)
        if (a == 1 || a % 100 == 0) {
            const double elapsed = now_sec() - nav_build_start;
            const double rate = elapsed > 0.0 ? (double)a / elapsed : 0.0;
            std::cerr << "[mng-index] nav_inserted " << a << "/"
                      << nav.num_anchors << " elapsed=" << elapsed
                      << "s rate=" << rate << " anchors/s\n";
        }
#endif
        uint16_t alevel = nav.levels[a];
        uint32_t curr = nav.entry_anchor;
        uint32_t adoc = nav.anchor_doc_ids[a];
        if (current_max > alevel) {
            for (int l = current_max; l > (int)alevel; --l) {
                bool changed = true;
                float best = symmetric_doc_score(adoc, nav.anchor_doc_ids[curr], docs, tokens, centroids, scratch);
                while (changed) {
                    changed = false;
                    const NavLayer& layer = nav.layers[l];
                    const uint32_t* neigh = layer.neighbors.data() + (size_t)curr * layer.capacity;
                    uint16_t deg = layer.degree[curr];
                    for (uint16_t i = 0; i < deg; ++i) {
                        uint32_t nb = neigh[i];
                        float s = symmetric_doc_score(adoc, nav.anchor_doc_ids[nb], docs, tokens, centroids, scratch);
                        if (s > best) {
                            best = s;
                            curr = nb;
                            changed = true;
                        }
                    }
                }
            }
        }
        int top_layer = std::min<int>(alevel, current_max);
        for (int l = top_layer; l >= 0; --l) {
            auto score_fn = [&](uint32_t anchor) {
                return symmetric_doc_score(adoc, nav.anchor_doc_ids[anchor], docs, tokens, centroids, scratch);
            };
            uint32_t found = search_nav_layer(nav, curr, (uint32_t)l, cfg.ef_nav_build, score_fn, scratch);
            uint32_t keep = std::min<uint32_t>(found, nav.M_nav);
            rewrite_nav_neighbors(nav, (uint32_t)l, a, scratch.result_ids.data(), keep);
            for (uint32_t i = 0; i < keep; ++i) {
                append_or_prune_nav(nav, (uint32_t)l, scratch.result_ids[i], a,
                                    docs, tokens, centroids, scratch);
            }
            if (found > 0) curr = scratch.result_ids[0];
        }
        if (alevel > current_max) {
            nav.entry_anchor = a;
            current_max = alevel;
            nav.max_level = current_max;
        }
    }
}

inline uint32_t search_anchors_for_doc(uint32_t doc,
                                       uint32_t ef,
                                       uint32_t topk,
                                       const NavGraph& nav,
                                       const DocTable& docs,
                                       const TokenStore& tokens,
                                       const CentroidStore& centroids,
                                       BuildWorkerScratch& scratch,
                                       uint32_t* out_ids,
                                       float* out_scores) {
    uint32_t curr = nav.entry_anchor;
    auto score_fn = [&](uint32_t anchor) {
        return directed_doc_score(doc, nav.anchor_doc_ids[anchor], docs, tokens, centroids, scratch);
    };
    for (int l = nav.max_level; l > 0; --l) {
        bool changed = true;
        float best = score_fn(curr);
        while (changed) {
            changed = false;
            const NavLayer& layer = nav.layers[l];
            const uint32_t* neigh = layer.neighbors.data() + (size_t)curr * layer.capacity;
            uint16_t deg = layer.degree[curr];
            for (uint16_t i = 0; i < deg; ++i) {
                uint32_t nb = neigh[i];
                float s = score_fn(nb);
                if (s > best) {
                    best = s;
                    curr = nb;
                    changed = true;
                }
            }
        }
    }
    uint32_t found = search_nav_layer(nav, curr, 0, ef, score_fn, scratch);
    uint32_t keep = std::min(topk, found);
    for (uint32_t i = 0; i < keep; ++i) {
        out_ids[i] = scratch.result_ids[i];
        if (out_scores) out_scores[i] = scratch.result_scores[i];
    }
    return keep;
}

template <typename ScoreFn>
inline uint32_t search_anchors_for_query_with_score(uint32_t ef,
                                                    uint32_t topk,
                                                    const NavGraph& nav,
                                                    SearchScratch& scratch,
                                                    uint32_t* out_ids,
                                                    float* out_scores,
                                                    ScoreFn score_fn) {
    uint32_t curr = nav.entry_anchor;
    for (int l = nav.max_level; l > 0; --l) {
        bool changed = true;
        float best = score_fn(curr);
        while (changed) {
            changed = false;
            const NavLayer& layer = nav.layers[l];
            const uint32_t* neigh = layer.neighbors.data() + (size_t)curr * layer.capacity;
            uint16_t deg = layer.degree[curr];
            for (uint16_t i = 0; i < deg; ++i) {
                uint32_t nb = neigh[i];
                float s = score_fn(nb);
                if (s > best) {
                    best = s;
                    curr = nb;
                    changed = true;
                }
            }
        }
    }
    uint32_t found = search_nav_layer(nav, curr, 0, ef, score_fn, scratch);
    uint32_t keep = std::min(topk, found);
    for (uint32_t i = 0; i < keep; ++i) {
        out_ids[i] = scratch.result_ids[i];
        if (out_scores) out_scores[i] = scratch.result_scores[i];
    }
    return keep;
}

inline uint32_t search_anchors_for_query(uint32_t ef,
                                         uint32_t topk,
                                         const NavGraph& nav,
                                         const DocTable& docs,
                                         const TokenStore& tokens,
                                         SearchScratch& scratch,
                                         uint32_t* out_ids,
                                         float* out_scores) {
    auto score_fn = [&](uint32_t anchor) {
        return directed_query_score(nav.anchor_doc_ids[anchor], docs, tokens, scratch, scratch);
    };
    return search_anchors_for_query_with_score(
        ef, topk, nav, scratch, out_ids, out_scores, score_fn);
}

inline void build_assignment(const NavGraph& nav,
                             const std::vector<uint32_t>& doc_to_anchor,
                             const DocTable& docs,
                             const TokenStore& tokens,
                             const CentroidStore& centroids,
                             const BuildConfig& cfg,
                             AssignmentStore& assignment) {
    assignment.num_docs = docs.num_docs;
    assignment.assignment_gamma_eff =
        std::min<uint32_t>(cfg.assignment_gamma, nav.num_anchors);
    assignment.primary_anchor.resize_uninitialized(docs.num_docs);
    assignment.degree.resize_uninitialized(docs.num_docs);
    assignment.anchor_ids.resize_uninitialized((size_t)docs.num_docs *
                                               assignment.assignment_gamma_eff);
    assignment.anchor_scores.resize_uninitialized((size_t)docs.num_docs *
                                                  assignment.assignment_gamma_eff);
    std::fill(assignment.anchor_ids.data(),
              assignment.anchor_ids.data() + (size_t)docs.num_docs * assignment.assignment_gamma_eff,
              kInvalidId);
    std::fill(assignment.anchor_scores.data(),
              assignment.anchor_scores.data() + (size_t)docs.num_docs * assignment.assignment_gamma_eff,
              -std::numeric_limits<float>::infinity());
    const int threads = MNG_INDEX_THREADS;
    std::vector<BuildWorkerScratch> scratches(threads);
    for (int t = 0; t < threads; ++t) {
        ensure_build_scratch(scratches[t], assignment.assignment_gamma_eff + 8, docs.max_doclen,
                             centroids.max_centroids_per_doc, beta_for_target(docs.max_doclen),
                             nav.num_anchors,
                             std::max<uint32_t>(cfg.ef_nav_assign, assignment.assignment_gamma_eff) + 8);
    }
    #pragma omp parallel for schedule(dynamic, 256) num_threads(MNG_INDEX_THREADS)
    for (uint32_t doc = 0; doc < docs.num_docs; ++doc) {
        int tid = omp_get_thread_num();
        BuildWorkerScratch& scratch = scratches[tid];
        uint32_t* row = assignment.anchor_ids.data() + (size_t)doc * assignment.assignment_gamma_eff;
        float* score_row = assignment.anchor_scores.data() + (size_t)doc * assignment.assignment_gamma_eff;
        float* tmp_scores = scratch.result_scores.data();
        uint32_t count = 0;
        uint32_t self_anchor = doc_to_anchor[doc];
        if (self_anchor != kInvalidId) {
            row[count++] = self_anchor;
            score_row[count - 1] = directed_doc_score(doc, nav.anchor_doc_ids[self_anchor],
                                                      docs, tokens, centroids, scratch);
        }
        uint32_t found = search_anchors_for_doc(doc, cfg.ef_nav_assign,
                                                assignment.assignment_gamma_eff,
                                                nav, docs, tokens, centroids, scratch,
                                                scratch.result_ids.data(), tmp_scores);
        for (uint32_t i = 0; i < found && count < assignment.assignment_gamma_eff; ++i) {
            uint32_t a = scratch.result_ids[i];
            bool seen = false;
            for (uint32_t j = 0; j < count; ++j) if (row[j] == a) seen = true;
            if (!seen) {
                row[count] = a;
                score_row[count] = tmp_scores[i];
                ++count;
            }
        }
        assignment.degree[doc] = (uint16_t)count;
        assignment.primary_anchor[doc] = count ? row[0] : 0;
    }
}

inline void build_buckets(const AssignmentStore& assignment, BucketStore& buckets) {
    buckets.num_anchors = 0;
    for (uint32_t i = 0; i < assignment.num_docs; ++i) {
        const uint32_t* row =
            assignment.anchor_ids.data() + (size_t)i * assignment.assignment_gamma_eff;
        for (uint16_t j = 0; j < assignment.degree[i]; ++j) {
            buckets.num_anchors = std::max(buckets.num_anchors, row[j] + 1);
        }
    }
    buckets.offsets.resize_uninitialized((size_t)buckets.num_anchors + 1);
    buckets.doc_ids.resize_uninitialized(assignment.num_docs);
    std::vector<uint64_t> counts(buckets.num_anchors, 0);
    for (uint32_t doc = 0; doc < assignment.num_docs; ++doc) counts[assignment.primary_anchor[doc]]++;
    buckets.offsets[0] = 0;
    for (uint32_t a = 0; a < buckets.num_anchors; ++a) {
        buckets.offsets[a + 1] = buckets.offsets[a] + counts[a];
    }
    std::vector<uint64_t> cursor(buckets.num_anchors);
    for (uint32_t a = 0; a < buckets.num_anchors; ++a) cursor[a] = buckets.offsets[a];
    for (uint32_t doc = 0; doc < assignment.num_docs; ++doc) {
        uint32_t a = assignment.primary_anchor[doc];
        buckets.doc_ids[cursor[a]++] = doc;
    }
}

inline uint64_t materialize_candidates(uint32_t doc,
                                       const uint32_t* assignment_ids,
                                       const uint16_t* assignment_degree,
                                       const uint64_t* bucket_offsets,
                                       const uint32_t* bucket_doc_ids,
                                       uint32_t assignment_gamma_eff,
                                       uint32_t* out) {
    uint64_t n = 0;
    const uint32_t* anchors = assignment_ids + (size_t)doc * assignment_gamma_eff;
    const uint16_t deg = assignment_degree[doc];
    for (uint16_t i = 0; i < deg; ++i) {
        uint32_t a = anchors[i];
        for (uint64_t p = bucket_offsets[a]; p < bucket_offsets[a + 1]; ++p) {
            uint32_t v = bucket_doc_ids[p];
            if (v != doc) out[n++] = v;
        }
    }
    return n;
}

inline uint64_t compute_candidate_bound(const AssignmentStore& assignment,
                                        const BucketStore& buckets,
                                        BuildStats& stats) {
    std::vector<uint64_t> sizes;
    sizes.reserve(assignment.num_docs);
    uint64_t sum = 0, maxv = 0;
    for (uint32_t doc = 0; doc < assignment.num_docs; ++doc) {
        const uint32_t* anchors =
            assignment.anchor_ids.data() + (size_t)doc * assignment.assignment_gamma_eff;
        uint64_t sz = 0;
        bool hit_self = false;
        for (uint16_t i = 0; i < assignment.degree[doc]; ++i) {
            uint32_t a = anchors[i];
            sz += buckets.offsets[a + 1] - buckets.offsets[a];
            if (a == assignment.primary_anchor[doc]) hit_self = true;
        }
        if (hit_self && sz > 0) --sz;
        sizes.push_back(sz);
        sum += sz;
        maxv = std::max(maxv, sz);
    }
    std::sort(sizes.begin(), sizes.end());
    auto pct = [&](double q) -> uint64_t {
        if (sizes.empty()) return 0;
        size_t idx = std::min<size_t>(sizes.size() - 1, (size_t)std::floor(q * (sizes.size() - 1)));
        return sizes[idx];
    };
    stats.candidate_mean = sizes.empty() ? 0 : sum / sizes.size();
    stats.candidate_p50 = pct(0.50);
    stats.candidate_p95 = pct(0.95);
    stats.candidate_p99 = pct(0.99);
    stats.candidate_max = maxv;
    return maxv;
}

inline uint16_t select_top_m0(const uint32_t* cand,
                              const float* scores,
                              uint32_t n,
                              uint32_t M0,
                              uint32_t* order,
                              uint32_t* out_neighbors,
                              float* out_scores) {
    uint32_t keep = std::min<uint32_t>(n, M0);
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    auto cmp = [&](uint32_t a, uint32_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return cand[a] < cand[b];
    };
    if (n > keep) std::nth_element(order, order + keep, order + n, cmp);
    std::sort(order, order + keep, cmp);
    for (uint32_t i = 0; i < keep; ++i) {
        uint32_t idx = order[i];
        out_neighbors[i] = cand[idx];
        if (out_scores) out_scores[i] = scores[idx];
    }
    return (uint16_t)keep;
}

inline void build_prox_graph_nearest_top_m0(const AssignmentStore& assignment,
                                            const BucketStore& buckets,
                                            const DocTable& docs,
                                            const TokenStore& tokens,
                                            const CentroidStore& centroids,
                                            const BuildConfig& cfg,
                                            uint64_t candidate_bound,
                                            ProxGraph& prox,
                                            BuildStats& stats) {
    prox.num_docs = docs.num_docs;
    prox.M0 = cfg.M0;
    prox.neighbors.resize_uninitialized((size_t)docs.num_docs * cfg.M0);
    prox.degree.resize_uninitialized(docs.num_docs);
    std::fill(prox.neighbors.data(), prox.neighbors.data() + (size_t)docs.num_docs * cfg.M0, kInvalidId);
    std::fill(prox.degree.data(), prox.degree.data() + docs.num_docs, 0);
#if defined(MNG_MODE_DEBUG)
    prox.scores.resize_uninitialized((size_t)docs.num_docs * cfg.M0);
    std::fill(prox.scores.data(), prox.scores.data() + (size_t)docs.num_docs * cfg.M0,
              -std::numeric_limits<float>::infinity());
#endif
    const int threads = MNG_INDEX_THREADS;
    std::vector<BuildWorkerScratch> scratches(threads);
    for (int t = 0; t < threads; ++t) {
        ensure_build_scratch(scratches[t], (uint32_t)candidate_bound, docs.max_doclen,
                             centroids.max_centroids_per_doc, beta_for_target(docs.max_doclen),
                             1, std::max<uint32_t>(cfg.ef_nav_assign, cfg.assignment_gamma) + 8);
    }
    const uint32_t* assignment_ids = assignment.anchor_ids.data();
    const uint16_t* assignment_degree = assignment.degree.data();
    const uint64_t* bucket_offsets = buckets.offsets.data();
    const uint32_t* bucket_doc_ids = buckets.doc_ids.data();
    const uint32_t assignment_gamma_eff = assignment.assignment_gamma_eff;
    uint32_t* prox_neighbors = prox.neighbors.data();
    uint16_t* prox_degree = prox.degree.data();
    const uint32_t block_size = 128;
    const uint32_t n_blocks = (docs.num_docs + block_size - 1) / block_size;
    #pragma omp parallel for schedule(dynamic) num_threads(MNG_INDEX_THREADS)
    for (uint32_t block = 0; block < n_blocks; ++block) {
        int tid = omp_get_thread_num();
        BuildWorkerScratch& scratch = scratches[tid];
        uint32_t begin = block * block_size;
        uint32_t end = std::min<uint32_t>(docs.num_docs, begin + block_size);
        for (uint32_t doc = begin; doc < end; ++doc) {
            uint64_t n64 = materialize_candidates(doc, assignment_ids, assignment_degree,
                                                  bucket_offsets, bucket_doc_ids,
                                                  assignment_gamma_eff,
                                                  scratch.candidate_ids.data());
            uint32_t n = (uint32_t)n64;
            scratch.local_candidates += n;
            if (n > 0) {
                score_symmetric_doc_to_docs(doc, scratch.candidate_ids.data(), n,
                                            scratch.candidate_scores.data(),
                                            docs, tokens, centroids, scratch);
            }
            uint32_t* out = prox_neighbors + (size_t)doc * cfg.M0;
#if defined(MNG_MODE_DEBUG)
            float* out_scores = prox.scores.data() + (size_t)doc * cfg.M0;
#else
            float* out_scores = nullptr;
#endif
            uint16_t deg = select_top_m0(scratch.candidate_ids.data(),
                                         scratch.candidate_scores.data(), n, cfg.M0,
                                         scratch.candidate_order.data(), out, out_scores);
            prox_degree[doc] = deg;
            scratch.local_edges += deg;
            if (deg < cfg.M0) scratch.local_underfilled++;
        }
    }
    for (auto& s : scratches) {
        stats.selected_edges += s.local_edges;
        stats.underfilled_rows += s.local_underfilled;
        stats.prox_score_forward_sec += s.local_forward_sec;
        stats.prox_score_reverse_sec += s.local_reverse_sec;
    }
}

inline void choose_rank_mutual_active_edges(uint32_t doc,
                                            const uint32_t* candidates,
                                            uint32_t n,
                                            const DocTable& docs,
                                            const TokenStore& tokens,
                                            const CentroidStore& centroids,
                                            const BuildConfig& cfg,
                                            BuildWorkerScratch& scratch,
                                            uint32_t* out,
                                            uint16_t& out_degree) {
    const uint32_t M = cfg.M;
    if (n <= M) {
        for (uint32_t i = 0; i < n; ++i) out[i] = candidates[i];
        out_degree = (uint16_t)n;
        return;
    }

    uint32_t earlier_count = 0;
    uint32_t later_count = 0;
    uint32_t* earlier = scratch.selector_order.data();
    uint32_t* later = scratch.candidate_order.data();
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t v = candidates[i];
        if (v < doc) {
            earlier[earlier_count++] = v;
        } else {
            later[later_count++] = v;
        }
    }

    if (earlier_count > M) {
        score_symmetric_doc_to_docs(doc, earlier, earlier_count, scratch.candidate_scores.data(),
                                    docs, tokens, centroids, scratch);
        out_degree = select_top_m0(earlier, scratch.candidate_scores.data(), earlier_count, M,
                                   scratch.candidate_ids.data(), out, nullptr);
        return;
    }

    uint32_t degree = 0;
    for (; degree < earlier_count; ++degree) out[degree] = earlier[degree];
    const uint32_t need = M - degree;
    if (need > 0 && later_count > 0) {
        score_symmetric_doc_to_docs(doc, later, later_count, scratch.candidate_scores.data(),
                                    docs, tokens, centroids, scratch);
        uint16_t add = select_top_m0(later, scratch.candidate_scores.data(), later_count, need,
                                     scratch.candidate_ids.data(), out + degree, nullptr);
        degree += add;
    }
    out_degree = (uint16_t)degree;
}

inline void build_prox_graph_rank_mutual(const AssignmentStore& assignment,
                                         const BucketStore& buckets,
                                         const DocTable& docs,
                                         const TokenStore& tokens,
                                         const CentroidStore& centroids,
                                         const BuildConfig& cfg,
                                         uint64_t candidate_bound,
                                         ProxGraph& prox,
                                         BuildStats& stats) {
    prox.num_docs = docs.num_docs;
    prox.M0 = cfg.M0;
    prox.neighbors.resize_uninitialized((size_t)docs.num_docs * cfg.M0);
    prox.degree.resize_uninitialized(docs.num_docs);
    std::fill(prox.neighbors.data(), prox.neighbors.data() + (size_t)docs.num_docs * cfg.M0, kInvalidId);
    std::fill(prox.degree.data(), prox.degree.data() + docs.num_docs, 0);
#if defined(MNG_MODE_DEBUG)
    prox.scores.resize_uninitialized((size_t)docs.num_docs * cfg.M0);
    std::fill(prox.scores.data(), prox.scores.data() + (size_t)docs.num_docs * cfg.M0,
              -std::numeric_limits<float>::infinity());
#endif

    AlignedBuffer<uint32_t> own_neighbors;
    AlignedBuffer<uint16_t> own_degree;
    AlignedBuffer<uint32_t> reverse_counts;
    own_neighbors.resize_uninitialized((size_t)docs.num_docs * cfg.M);
    own_degree.resize_uninitialized(docs.num_docs);
    reverse_counts.resize_uninitialized(docs.num_docs);
    std::fill(own_neighbors.data(), own_neighbors.data() + (size_t)docs.num_docs * cfg.M, kInvalidId);
    std::fill(own_degree.data(), own_degree.data() + docs.num_docs, 0);
    std::fill(reverse_counts.data(), reverse_counts.data() + docs.num_docs, 0);

    const int threads = MNG_INDEX_THREADS;
    std::vector<BuildWorkerScratch> select_scratches(threads);
    for (int t = 0; t < threads; ++t) {
        ensure_build_scratch(select_scratches[t], (uint32_t)candidate_bound, docs.max_doclen,
                             centroids.max_centroids_per_doc, beta_for_target(docs.max_doclen),
                             1, std::max<uint32_t>(cfg.ef_nav_assign, cfg.assignment_gamma) + 8);
    }

    const uint32_t* assignment_ids = assignment.anchor_ids.data();
    const uint16_t* assignment_degree = assignment.degree.data();
    const uint64_t* bucket_offsets = buckets.offsets.data();
    const uint32_t* bucket_doc_ids = buckets.doc_ids.data();
    const uint32_t assignment_gamma_eff = assignment.assignment_gamma_eff;
    const uint32_t block_size = 128;
    const uint32_t n_blocks = (docs.num_docs + block_size - 1) / block_size;

    #pragma omp parallel for schedule(dynamic) num_threads(MNG_INDEX_THREADS)
    for (uint32_t block = 0; block < n_blocks; ++block) {
        int tid = omp_get_thread_num();
        BuildWorkerScratch& scratch = select_scratches[tid];
        uint32_t begin = block * block_size;
        uint32_t end = std::min<uint32_t>(docs.num_docs, begin + block_size);
        for (uint32_t doc = begin; doc < end; ++doc) {
            uint64_t n64 = materialize_candidates(doc, assignment_ids, assignment_degree,
                                                  bucket_offsets, bucket_doc_ids,
                                                  assignment_gamma_eff,
                                                  scratch.candidate_ids.data());
            uint32_t n = (uint32_t)n64;
            scratch.local_candidates += n;
            uint32_t* out = own_neighbors.data() + (size_t)doc * cfg.M;
            uint16_t deg = 0;
            choose_rank_mutual_active_edges(doc, scratch.candidate_ids.data(), n,
                                            docs, tokens, centroids, cfg, scratch,
                                            out, deg);
            own_degree[doc] = deg;
            for (uint16_t i = 0; i < deg; ++i) {
                uint32_t v = out[i];
                if (v < doc) {
                    #pragma omp atomic update
                    reverse_counts[v]++;
                }
            }
        }
    }

    AlignedBuffer<uint64_t> reverse_offsets;
    reverse_offsets.resize_uninitialized((size_t)docs.num_docs + 1);
    reverse_offsets[0] = 0;
    uint64_t total_reverse = 0;
    uint32_t max_reverse = 0;
    for (uint32_t doc = 0; doc < docs.num_docs; ++doc) {
        uint32_t c = reverse_counts[doc];
        max_reverse = std::max(max_reverse, c);
        total_reverse += c;
        reverse_offsets[doc + 1] = total_reverse;
    }
    stats.rank_reverse_edges = total_reverse;
    stats.rank_reverse_max = max_reverse;

    AlignedBuffer<uint32_t> reverse_neighbors;
    AlignedBuffer<uint32_t> reverse_cursor;
    reverse_neighbors.resize_uninitialized(total_reverse == 0 ? 1 : total_reverse);
    reverse_cursor.resize_uninitialized(docs.num_docs);
    std::fill(reverse_cursor.data(), reverse_cursor.data() + docs.num_docs, 0);

    #pragma omp parallel for schedule(static) num_threads(MNG_INDEX_THREADS)
    for (uint32_t doc = 0; doc < docs.num_docs; ++doc) {
        const uint32_t* row = own_neighbors.data() + (size_t)doc * cfg.M;
        uint16_t deg = own_degree[doc];
        for (uint16_t i = 0; i < deg; ++i) {
            uint32_t v = row[i];
            if (v < doc) {
                uint32_t pos;
                #pragma omp atomic capture
                pos = reverse_cursor[v]++;
                reverse_neighbors[reverse_offsets[v] + pos] = doc;
            }
        }
    }

    const uint32_t final_bound = std::max<uint32_t>(cfg.M + max_reverse, cfg.M0);
    std::vector<BuildWorkerScratch> prune_scratches(threads);
    for (int t = 0; t < threads; ++t) {
        ensure_build_scratch(prune_scratches[t], final_bound + 8, docs.max_doclen,
                             centroids.max_centroids_per_doc, beta_for_target(docs.max_doclen),
                             1, std::max<uint32_t>(cfg.ef_nav_assign, cfg.assignment_gamma) + 8);
    }

    uint32_t* prox_neighbors = prox.neighbors.data();
    uint16_t* prox_degree = prox.degree.data();
    #pragma omp parallel for schedule(dynamic, 128) num_threads(MNG_INDEX_THREADS)
    for (uint32_t doc = 0; doc < docs.num_docs; ++doc) {
        int tid = omp_get_thread_num();
        BuildWorkerScratch& scratch = prune_scratches[tid];
        uint32_t n = 0;
        const uint32_t* own = own_neighbors.data() + (size_t)doc * cfg.M;
        for (uint16_t i = 0; i < own_degree[doc]; ++i) {
            scratch.candidate_ids[n++] = own[i];
        }
        for (uint64_t p = reverse_offsets[doc]; p < reverse_offsets[doc + 1]; ++p) {
            scratch.candidate_ids[n++] = reverse_neighbors[p];
        }
        if (n > 1) {
            std::sort(scratch.candidate_ids.data(), scratch.candidate_ids.data() + n);
            n = (uint32_t)(std::unique(scratch.candidate_ids.data(),
                                       scratch.candidate_ids.data() + n) -
                           scratch.candidate_ids.data());
        }

        uint32_t* out = prox_neighbors + (size_t)doc * cfg.M0;
#if defined(MNG_MODE_DEBUG)
        float* out_scores = prox.scores.data() + (size_t)doc * cfg.M0;
#else
        float* out_scores = nullptr;
#endif
        uint16_t deg = 0;
        if (n > cfg.M0) {
            score_symmetric_doc_to_docs(doc, scratch.candidate_ids.data(), n,
                                        scratch.candidate_scores.data(),
                                        docs, tokens, centroids, scratch);
            deg = select_top_m0(scratch.candidate_ids.data(), scratch.candidate_scores.data(),
                                n, cfg.M0, scratch.candidate_order.data(), out, out_scores);
            #pragma omp atomic update
            stats.rank_pruned_rows++;
        } else {
            for (uint32_t i = 0; i < n; ++i) out[i] = scratch.candidate_ids[i];
            for (uint32_t i = n; i < cfg.M0; ++i) out[i] = kInvalidId;
            deg = (uint16_t)n;
        }
        prox_degree[doc] = deg;
        scratch.local_edges += deg;
        if (deg < cfg.M0) scratch.local_underfilled++;
    }

    for (auto& s : select_scratches) {
        stats.prox_score_forward_sec += s.local_forward_sec;
        stats.prox_score_reverse_sec += s.local_reverse_sec;
    }
    for (auto& s : prune_scratches) {
        stats.selected_edges += s.local_edges;
        stats.underfilled_rows += s.local_underfilled;
        stats.prox_score_forward_sec += s.local_forward_sec;
        stats.prox_score_reverse_sec += s.local_reverse_sec;
    }
}

inline void build_prox_graph(const AssignmentStore& assignment,
                             const BucketStore& buckets,
                             const DocTable& docs,
                             const TokenStore& tokens,
                             const CentroidStore& centroids,
                             const BuildConfig& cfg,
                             uint64_t candidate_bound,
                             ProxGraph& prox,
                             BuildStats& stats) {
    if (cfg.prox_builder == "rank_mutual") {
        build_prox_graph_rank_mutual(assignment, buckets, docs, tokens, centroids,
                                     cfg, candidate_bound, prox, stats);
    } else {
        build_prox_graph_nearest_top_m0(assignment, buckets, docs, tokens, centroids,
                                        cfg, candidate_bound, prox, stats);
    }
}

inline void save_nav_graph(const std::string& dir, const NavGraph& nav) {
    ensure_dir2(dir + "/nav_graph");
    write_aligned_bin(dir + "/nav_graph/anchor_doc_ids.u32.bin", nav.anchor_doc_ids);
    write_aligned_bin(dir + "/nav_graph/levels.u16.bin", nav.levels);
    for (uint32_t l = 0; l < nav.layers.size(); ++l) {
        write_aligned_bin(dir + "/nav_graph/neighbors_l" + std::to_string(l) + ".u32.bin",
                          nav.layers[l].neighbors);
        write_aligned_bin(dir + "/nav_graph/degree_l" + std::to_string(l) + ".u16.bin",
                          nav.layers[l].degree);
    }
}

inline void load_nav_graph(const std::string& dir,
                           const std::unordered_map<std::string, std::string>& meta,
                           NavGraph& nav) {
    nav.num_anchors = (uint32_t)std::stoul(meta.at("num_anchors"));
    nav.M_nav = (uint32_t)std::stoul(meta.at("M_nav"));
    nav.maxM_nav = (uint32_t)std::stoul(meta.at("maxM_nav"));
    nav.maxM0_nav = (uint32_t)std::stoul(meta.at("maxM0_nav"));
    nav.entry_anchor = (uint32_t)std::stoul(meta.at("nav_entry_anchor"));
    nav.max_level = (uint16_t)std::stoul(meta.at("nav_max_level"));
    read_aligned_bin_exact(dir + "/nav_graph/anchor_doc_ids.u32.bin", nav.anchor_doc_ids, nav.num_anchors);
    read_aligned_bin_exact(dir + "/nav_graph/levels.u16.bin", nav.levels, nav.num_anchors);
    nav.layers.resize((size_t)nav.max_level + 1);
    for (uint32_t l = 0; l <= nav.max_level; ++l) {
        nav.layers[l].capacity = (l == 0) ? nav.maxM0_nav : nav.maxM_nav;
        read_aligned_bin_exact(dir + "/nav_graph/neighbors_l" + std::to_string(l) + ".u32.bin",
                               nav.layers[l].neighbors, (size_t)nav.num_anchors * nav.layers[l].capacity);
        read_aligned_bin_exact(dir + "/nav_graph/degree_l" + std::to_string(l) + ".u16.bin",
                               nav.layers[l].degree, nav.num_anchors);
    }
}

inline void save_prox_graph(const std::string& dir, const ProxGraph& prox) {
    write_aligned_bin(dir + "/prox_neighbors.u32.bin", prox.neighbors);
    write_aligned_bin(dir + "/prox_degree.u16.bin", prox.degree);
#if defined(MNG_MODE_DEBUG)
    ensure_dir2(dir + "/debug");
    write_aligned_bin(dir + "/debug/prox_scores.f32.bin", prox.scores);
#endif
}

inline void load_prox_graph(const std::string& dir,
                            uint32_t num_docs,
                            uint32_t M0,
                            ProxGraph& prox) {
    prox.num_docs = num_docs;
    prox.M0 = M0;
    read_aligned_bin_exact(dir + "/prox_neighbors.u32.bin", prox.neighbors, (size_t)num_docs * M0);
    read_aligned_bin_exact(dir + "/prox_degree.u16.bin", prox.degree, num_docs);
}

inline void save_debug_assignment_bucket(const std::string& dir,
                                         const AssignmentStore& assignment,
                                         const BucketStore& buckets) {
#if defined(MNG_MODE_DEBUG)
    ensure_dir2(dir + "/debug");
    write_aligned_bin(dir + "/debug/primary_anchor.u32.bin", assignment.primary_anchor);
    write_aligned_bin(dir + "/debug/assignment_anchor_ids.u32.bin", assignment.anchor_ids);
    write_aligned_bin(dir + "/debug/assignment_anchor_scores.f32.bin", assignment.anchor_scores);
    write_aligned_bin(dir + "/debug/assignment_anchor_degree.u16.bin", assignment.degree);
    write_aligned_bin(dir + "/debug/bucket_offsets.u64.bin", buckets.offsets);
    write_aligned_bin(dir + "/debug/bucket_doc_ids.u32.bin", buckets.doc_ids);
#else
    (void)dir; (void)assignment; (void)buckets;
#endif
}

inline void write_build_stats_json(const std::string& path, const BuildStats& s) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"read_sec\": " << s.read_sec << ",\n";
    f << "  \"fp32_materialize_sec\": " << s.fp32_materialize_sec << ",\n";
    f << "  \"centroid_build_sec\": " << s.centroid_build_sec << ",\n";
    f << "  \"nav_build_sec\": " << s.nav_build_sec << ",\n";
    f << "  \"assignment_sec\": " << s.assignment_sec << ",\n";
    f << "  \"bucket_build_sec\": " << s.bucket_build_sec << ",\n";
    f << "  \"candidate_bound_sec\": " << s.candidate_bound_sec << ",\n";
    f << "  \"prox_build_sec\": " << s.prox_build_sec << ",\n";
    f << "  \"prox_score_forward_sec\": " << s.prox_score_forward_sec << ",\n";
    f << "  \"prox_score_reverse_sec\": " << s.prox_score_reverse_sec << ",\n";
    f << "  \"save_sec\": " << s.save_sec << ",\n";
    f << "  \"core_build_sec\": " << s.core_build_sec << ",\n";
    f << "  \"total_build_sec\": " << s.total_build_sec << ",\n";
    f << "  \"candidate_mean\": " << s.candidate_mean << ",\n";
    f << "  \"candidate_p50\": " << s.candidate_p50 << ",\n";
    f << "  \"candidate_p95\": " << s.candidate_p95 << ",\n";
    f << "  \"candidate_p99\": " << s.candidate_p99 << ",\n";
    f << "  \"candidate_max\": " << s.candidate_max << ",\n";
    f << "  \"selected_edges\": " << s.selected_edges << ",\n";
    f << "  \"underfilled_rows\": " << s.underfilled_rows << ",\n";
    f << "  \"rank_reverse_edges\": " << s.rank_reverse_edges << ",\n";
    f << "  \"rank_reverse_max\": " << s.rank_reverse_max << ",\n";
    f << "  \"rank_pruned_rows\": " << s.rank_pruned_rows << "\n";
    f << "}\n";
}

inline void save_index_meta(const std::string& dir,
                            const BuildConfig& cfg,
                            const DocTable& docs,
                            const NavGraph& nav,
                            const ProxGraph& prox,
                            const BuildStats& stats) {
    std::vector<std::pair<std::string, std::string>> kv = {
        {"method", "mng"},
        {"layout", "mng_v1"},
        {"schema_version", "1"},
        {"index_static", "true"},
        {"nav_only", cfg.nav_only ? "true" : "false"},
        {"dim", std::to_string(docs.dim)},
        {"num_docs", std::to_string(docs.num_docs)},
        {"num_tokens", std::to_string(docs.total_tokens)},
        {"metric", "ip"},
        {"score_semantics", "larger_is_better"},
        {"scorer", "centroid_maxsim"},
        {"scorer_variant", "top1_maxsim"},
        {"aggregated_gamma_nn", "false"},
        {"scorer_backend", backend_string()},
        {"doc_doc_score", "symmetric_average_directed"},
        {"query_doc_score", "directed_query_to_doc"},
        {"centroid_policy", "sqrt_doclen"},
        {"kmeans_policy", "faiss_spherical"},
        {"kmeans_niter", std::to_string(cfg.kmeans_niter)},
        {"kmeans_nredo", std::to_string(cfg.kmeans_nredo)},
        {"kmeans_spherical", "true"},
        {"token_store", "fp32_materialized"},
        {"input_dtype", "fp16_doc_bundle"},
        {"M", std::to_string(cfg.M)},
        {"M0", std::to_string(cfg.M0)},
        {"assignment_gamma", std::to_string(cfg.assignment_gamma)},
        {"assignment_gamma_eff",
         std::to_string(std::min<uint32_t>(cfg.assignment_gamma, nav.num_anchors))},
        {"anchor_ratio", std::to_string(cfg.anchor_ratio)},
        {"num_anchors", std::to_string(nav.num_anchors)},
        {"M_nav", std::to_string(nav.M_nav)},
        {"maxM_nav", std::to_string(nav.maxM_nav)},
        {"maxM0_nav", std::to_string(nav.maxM0_nav)},
        {"ef_nav_build", std::to_string(cfg.ef_nav_build)},
        {"ef_nav_assign", std::to_string(cfg.ef_nav_assign)},
        {"ef_construction", std::to_string(cfg.ef_nav_build)},
        {"nav_selector", "nearest"},
        {"nav_level_type", "u16"},
        {"nav_max_level", std::to_string(nav.max_level)},
        {"nav_entry_anchor", std::to_string(nav.entry_anchor)},
        {"prox_selector", cfg.nav_only ? "disabled" :
            (cfg.prox_builder == "rank_mutual" ? "rank_mutual_v1" : "nearest_top_m0")},
        {"prox_builder", cfg.nav_only ? "disabled" : cfg.prox_builder},
        {"prox_degree_type", cfg.nav_only ? "none" : "u16"},
        {"assignment_saved", "false"},
        {"bucket_saved", "false"},
        {"assignment_scores_saved", std::string(build_mode_string()) == "debug" ? "true" : "false"},
        {"prox_scores_saved", std::string(build_mode_string()) == "debug" ? "true" : "false"},
        {"shared_doc_store", "true"},
        {"index_threads", std::to_string(MNG_INDEX_THREADS)},
        {"build_mode", build_mode_string()},
        {"seed", std::to_string(cfg.seed)},
        {"build_time_sec", std::to_string(stats.total_build_sec)},
        {"core_build_sec", std::to_string(stats.core_build_sec)}
    };
    (void)prox;
    baseline::write_meta(dir, kv);
}

inline void save_index(const std::string& dir,
                       const BuildConfig& cfg,
                       const DocTable& docs,
                       const NavGraph& nav,
                       const AssignmentStore& assignment,
                       const BucketStore& buckets,
                       const ProxGraph& prox,
                       BuildStats& stats) {
    ensure_dir2(dir);
    double t0 = now_sec();
    save_nav_graph(dir, nav);
    save_prox_graph(dir, prox);
    save_debug_assignment_bucket(dir, assignment, buckets);
    stats.save_sec = now_sec() - t0;
    stats.total_build_sec += stats.save_sec;
    save_index_meta(dir, cfg, docs, nav, prox, stats);
    if (std::string(build_mode_string()) != "normal") {
        write_build_stats_json(dir + "/build_stats.json", stats);
    }
}

inline void save_nav_only_index(const std::string& dir,
                                const BuildConfig& cfg,
                                const DocTable& docs,
                                const NavGraph& nav,
                                BuildStats& stats) {
    ensure_dir2(dir);
    double t0 = now_sec();
    save_nav_graph(dir, nav);
    stats.save_sec = now_sec() - t0;
    stats.total_build_sec += stats.save_sec;
    ProxGraph empty_prox;
    save_index_meta(dir, cfg, docs, nav, empty_prox, stats);
    if (std::string(build_mode_string()) != "normal") {
        write_build_stats_json(dir + "/build_stats.json", stats);
    }
}

template <typename ScoreFn>
inline uint32_t prox_search_with_score(uint32_t entry_doc,
                                       const ProxGraph& prox,
                                       uint32_t num_docs,
                                       uint32_t ef,
                                       SearchScratch& scratch,
                                       ScoreFn score_fn) {
    ++scratch.visited_doc_tag;
    if (scratch.visited_doc_tag == 0) {
        std::fill(scratch.visited_doc.data(), scratch.visited_doc.data() + num_docs, 0);
        scratch.visited_doc_tag = 1;
    }
    const uint16_t tag = scratch.visited_doc_tag;
    scratch.candidate_queue.clear();
    scratch.top_candidates.clear();
    float s0 = score_fn(entry_doc);
    scratch.candidate_queue.push({s0, entry_doc});
    scratch.top_candidates.push({s0, entry_doc});
    scratch.visited_doc[entry_doc] = tag;
    const uint32_t* neighbors = prox.neighbors.data();
    const uint16_t* degree = prox.degree.data();
    const uint32_t M0 = prox.M0;
    while (!scratch.candidate_queue.empty()) {
        HeapItem cur = scratch.candidate_queue.pop();
        float worst = scratch.top_candidates.top().score;
        if (scratch.top_candidates.size() >= ef && cur.score < worst) break;
        const uint32_t base = cur.id * M0;
        const uint16_t deg = degree[cur.id];
        for (uint16_t i = 0; i < deg; ++i) {
            uint32_t nb = neighbors[base + i];
            if (scratch.visited_doc[nb] == tag) continue;
            scratch.visited_doc[nb] = tag;
            float s = score_fn(nb);
            if (scratch.top_candidates.size() < ef || s > scratch.top_candidates.top().score) {
                scratch.candidate_queue.push({s, nb});
                scratch.top_candidates.push({s, nb});
                if (scratch.top_candidates.size() > ef) scratch.top_candidates.pop();
            }
        }
    }
    uint32_t count = 0;
    while (!scratch.top_candidates.empty()) {
        HeapItem x = scratch.top_candidates.pop();
        scratch.result_ids[count] = x.id;
        scratch.result_scores[count] = x.score;
        ++count;
    }
    sort_scratch_results_desc(scratch, count);
    return count;
}

inline uint32_t prox_search(uint32_t entry_doc,
                            const ProxGraph& prox,
                            const DocTable& docs,
                            const TokenStore& tokens,
                            uint32_t ef,
                            SearchScratch& scratch) {
    auto score_fn = [&](uint32_t doc) {
        return directed_query_score(doc, docs, tokens, scratch, scratch);
    };
    return prox_search_with_score(entry_doc, prox, docs.num_docs, ef, scratch, score_fn);
}

}  // namespace mng
