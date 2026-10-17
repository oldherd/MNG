// mvbundle.hpp — read interface for multi-vector bundles produced by encode.py.
//
// Bundle directory layout:
//   embeddings.fp16.bin   doc mode: fp16 (n_tokens, dim) row-major
//   embeddings.fp32.bin   query mode: fp32 (n_tokens, dim) row-major
//   doclens.i32.bin       int32 (n_rows,) token count per row
//   docids.i32.bin        int32 (n_rows,) original id per row (doc id / query id)
//   meta.json             dtype, dim, counts, sha256, encode args
//
// The embedding file is mmap'd: opening a 100 GB bundle is O(1), accesses
// page in on demand. Per-row slicing is constant-time via a precomputed
// cumulative offset table.
//
// Doc bundle:   each row is one document, variable token count.
// Query bundle: each row is one query. With keep_padding=true, every row
//               has exactly query_maxlen tokens (= 32 by default); otherwise
//               rows are variable-length.
//
// Usage:
//
//   #include "mvbundle.hpp"
//
//   auto b = mv::MVBundle::load("/path/to/lotte_lifestyle_dev.mvbundle");
//
//   // metadata
//   b.n_docs(); b.n_tokens(); b.dim(); b.dtype(); b.is_query();
//
//   // typed view of full storage
//   const uint16_t* emb = b.embeddings_fp16();        // doc bundle
//   const float*    emb = b.embeddings_fp32();        // query bundle
//
//   // per-row slice
//   auto view = b.get(42);                            // {ptr, n_tokens, dim, dtype}
//   auto view = b.get_by_id(12345);                   // lookup by original id
//
//   // raw arrays
//   const int32_t* dl  = b.doclens();
//   const int32_t* did = b.docids();
//   const int64_t* off = b.offsets();
//
// Build:
//   g++ -std=c++17 -O3 -Wall -c mvbundle.cpp
//   (link with your binary; no external deps beyond POSIX + libc)

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mv {

enum class DType : uint8_t {
    Float16 = 0,
    Float32 = 1,
};

constexpr std::size_t dtype_bytes(DType d) noexcept {
    return d == DType::Float16 ? 2u : 4u;
}

constexpr const char* dtype_name(DType d) noexcept {
    return d == DType::Float16 ? "float16" : "float32";
}

// Lightweight error type so callers can distinguish IO / format / shape problems.
class BundleError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Parsed fields from meta.json. We only model what we actually consume;
// callers wanting raw meta strings should read meta.json themselves.
struct Meta {
    std::string magic;          // expect "mvbin"
    int         version = 0;    // expect 1
    DType       dtype = DType::Float16;
    int         dim = 0;
    std::size_t n_docs = 0;
    std::size_t n_tokens = 0;
    std::string mode;           // "doc" or "query"
    bool        is_query = false;
    bool        keep_padding = false;
    int         doc_maxlen = 0;
    int         query_maxlen = 0;
    std::string encoder_ckpt;
    std::string input_tsv;
    std::string dataset;
    std::string split;
    double      elapsed_sec = 0.0;
};

// Per-row slice into the embedding storage. Pointer is mmap-backed, read-only.
// Stride between consecutive token vectors is dim * dtype_bytes(dtype).
struct DocView {
    const void* data;        // tokens * dim * dtype_bytes(dtype) bytes
    std::size_t tokens;
    int         dim;
    DType       dtype;

    // Typed convenience accessors. Throw BundleError if dtype doesn't match.
    const std::uint16_t* as_fp16() const;
    const float*         as_fp32() const;
};

// Forward declaration; ground-truth class defined below.
class MVGroundTruth;

class MVBundle {
public:
    // Open + memory-map a bundle directory. Validates magic, version,
    // file sizes, and doclens.sum() == n_tokens.
    static MVBundle load(const std::filesystem::path& dir);

    // Non-copyable, movable.
    MVBundle(const MVBundle&) = delete;
    MVBundle& operator=(const MVBundle&) = delete;
    MVBundle(MVBundle&&) noexcept;
    MVBundle& operator=(MVBundle&&) noexcept;
    ~MVBundle();

    // --- metadata ---------------------------------------------------------
    const Meta& meta() const noexcept { return meta_; }
    const std::filesystem::path& path() const noexcept { return path_; }
    std::size_t n_docs()  const noexcept { return meta_.n_docs;  }
    std::size_t n_tokens() const noexcept { return meta_.n_tokens; }
    int         dim()     const noexcept { return meta_.dim;     }
    DType       dtype()   const noexcept { return meta_.dtype;   }
    bool        is_query() const noexcept { return meta_.is_query; }

    // --- raw storage (mmap-backed, read-only, owned by this object) -------
    // Embeddings: (n_tokens, dim) row-major contiguous.
    const void* embeddings_raw() const noexcept { return emb_ptr_; }
    std::size_t embeddings_bytes() const noexcept { return emb_size_; }

    // Typed views; throw BundleError if dtype mismatches.
    const std::uint16_t* embeddings_fp16() const;
    const float*         embeddings_fp32() const;

    // Per-row metadata, all sized (n_docs,) (offsets is n_docs+1).
    const std::int32_t* doclens() const noexcept { return doclens_.data(); }
    const std::int32_t* docids()  const noexcept { return docids_.data();  }
    const std::int64_t* offsets() const noexcept { return offsets_.data(); }

    // --- per-row slice ----------------------------------------------------
    DocView get(std::size_t row) const;

    // Build an id->row hash table lazily on first call. O(n_docs) once.
    DocView get_by_id(std::int32_t original_id);

    // Returns SIZE_MAX if not found. Builds the index lazily, same as get_by_id.
    std::size_t row_of_id(std::int32_t original_id);

private:
    MVBundle() = default;
    void build_id_index_();
    void release_mmap_() noexcept;

    Meta meta_;
    std::filesystem::path path_;

    // mmap state for embeddings
    const void* emb_ptr_  = nullptr;
    std::size_t emb_size_ = 0;
    int         emb_fd_   = -1;

    // small in-RAM tables (doclens + docids + offsets are typically MB-scale)
    std::vector<std::int32_t> doclens_;
    std::vector<std::int32_t> docids_;
    std::vector<std::int64_t> offsets_;

    std::unordered_map<std::int32_t, std::size_t> id_to_row_;
    bool id_index_built_ = false;
};

// ============================================================================
// Ground truth (output of compute_topk.py)
// ============================================================================
//
// Directory layout (one ground-truth set per (doc_bundle, query_bundle) pair):
//
//   <gt_dir>/
//     scores.fp32.bin       (n_q, n_d) row-major, score_dtype
//     scores.fp16.bin       — alternative when meta.score_dtype = float16
//     topk_indices.i32.bin  (n_q, top_k) int32, descending by score
//     topk_scores.fp32.bin  (n_q, top_k) matching scores
//     meta.json             magic="mvgt" v1
//
// Large datasets: scores.bin is mmap'd; topk_* are slurped into RAM
// (n_q × top_k × 4B is typically MB-scale — fits comfortably).
//
// Typical use: ANN benchmark Recall@k = |ann_topk ∩ gt_topk| / k.

struct GTMeta {
    std::string magic;      // expect "mvgt"
    int         version = 0;
    std::size_t n_queries = 0;
    std::size_t n_docs = 0;
    int         top_k = 0;
    int         dim = 0;
    int         q_maxlen = 0;
    DType       score_dtype = DType::Float32;
    std::string sort_order;  // "descending"
    double      elapsed_sec = 0.0;
};

class MVGroundTruth {
public:
    static MVGroundTruth load(const std::filesystem::path& dir);

    MVGroundTruth(const MVGroundTruth&) = delete;
    MVGroundTruth& operator=(const MVGroundTruth&) = delete;
    MVGroundTruth(MVGroundTruth&&) noexcept;
    MVGroundTruth& operator=(MVGroundTruth&&) noexcept;
    ~MVGroundTruth();

    // ---- metadata --------------------------------------------------------
    const GTMeta& meta() const noexcept { return meta_; }
    const std::filesystem::path& path() const noexcept { return path_; }
    std::size_t n_queries()  const noexcept { return meta_.n_queries; }
    std::size_t n_docs()     const noexcept { return meta_.n_docs;    }
    int         top_k()      const noexcept { return meta_.top_k;     }
    DType       score_dtype() const noexcept { return meta_.score_dtype; }

    // ---- full score matrix (mmap-backed) --------------------------------
    // (n_queries, n_docs) row-major. May be 10+ GB; OS pages on demand.
    const void* scores_raw() const noexcept { return scores_ptr_; }
    std::size_t scores_bytes() const noexcept { return scores_size_; }
    bool has_full_scores() const noexcept { return scores_ptr_ != nullptr; }

    // Typed views; throw BundleError if dtype mismatches.
    const float*         scores_fp32() const;
    const std::uint16_t* scores_fp16() const;

    // Per-query slice into the score matrix.
    // Returns a pointer to n_docs scalars of dtype score_dtype();
    // caller casts based on score_dtype().
    const void* scores_row(std::size_t q) const;

    // ---- top-k (in-RAM, fully loaded) ------------------------------------
    // Sorted descending. Use top-k for any k <= meta.top_k.
    const std::int32_t* topk_indices() const noexcept { return topk_indices_.data(); }
    const std::int32_t* topk_indices_row(std::size_t q) const;

    // topk_scores stored byte-blob; cast based on score_dtype()
    const void*          topk_scores_raw() const noexcept { return topk_scores_.data(); }
    const float*         topk_scores_fp32() const;
    const std::uint16_t* topk_scores_fp16() const;
    const void*          topk_scores_row(std::size_t q) const;

    // Recall@k against this GT's stored top-k for query q.
    // approx_topk must point to at least k int32 doc ids (caller's ANN top-k).
    // Throws if k > meta.top_k.
    float recall_at_k(const std::int32_t* approx_topk,
                      std::size_t q, std::size_t k) const;

private:
    MVGroundTruth() = default;
    void release_mmap_() noexcept;

    GTMeta meta_;
    std::filesystem::path path_;

    // mmap'd full score matrix
    const void* scores_ptr_  = nullptr;
    std::size_t scores_size_ = 0;
    int         scores_fd_   = -1;

    // in-RAM top-k tables
    std::vector<std::int32_t> topk_indices_;   // (n_q × top_k) flat
    std::vector<std::uint8_t> topk_scores_;    // (n_q × top_k × bytes) flat
};

// Module-level Recall@k helper. Both arrays of size k (treated as sets).
// O(k) average via small hashset on `approx_topk`.
float recall_at_k(const std::int32_t* approx_topk,
                  const std::int32_t* gt_topk,
                  std::size_t k);

}  // namespace mv
