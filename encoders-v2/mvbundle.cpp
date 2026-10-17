// mvbundle.cpp — implementation of mv::MVBundle.

#include "mvbundle.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace mv {

namespace {

// ---- minimal JSON value extractor ---------------------------------------
//
// meta.json is a flat object (apart from the "files" sub-object, which we
// ignore here). We only need to read scalar fields by key:
//
//   "key": "string"
//   "key": 12345
//   "key": 3.14
//   "key": true|false
//
// This is intentionally not a real JSON parser. It's robust against
// whitespace, but it does not handle escaped quotes inside string values or
// nested arrays. encode.py never writes such values into the fields we read.

std::string slurp_file_(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw BundleError("failed to open " + p.string());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Find the position right after `"key":` in `text`, skipping over whitespace.
// Returns std::string::npos if the key isn't present.
std::size_t find_key_value_pos_(std::string_view text, std::string_view key) {
    std::string needle = "\"";
    needle.append(key.data(), key.size());
    needle += "\"";
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string_view::npos) {
        std::size_t after = pos + needle.size();
        // require this to be followed (with optional whitespace) by ':'
        while (after < text.size() && (text[after] == ' ' || text[after] == '\t' ||
                                       text[after] == '\n' || text[after] == '\r')) ++after;
        if (after < text.size() && text[after] == ':') {
            ++after;
            while (after < text.size() && (text[after] == ' ' || text[after] == '\t' ||
                                           text[after] == '\n' || text[after] == '\r')) ++after;
            return after;
        }
        pos = after;
    }
    return std::string_view::npos;
}

bool get_string_(std::string_view text, std::string_view key, std::string& out) {
    auto p = find_key_value_pos_(text, key);
    if (p == std::string_view::npos) return false;
    if (p >= text.size() || text[p] != '"') return false;
    ++p;
    auto end = text.find('"', p);
    if (end == std::string_view::npos) return false;
    out.assign(text.data() + p, end - p);
    return true;
}

bool get_int_(std::string_view text, std::string_view key, long long& out) {
    auto p = find_key_value_pos_(text, key);
    if (p == std::string_view::npos) return false;
    const char* start = text.data() + p;
    char* endp = nullptr;
    long long v = std::strtoll(start, &endp, 10);
    if (endp == start) return false;
    out = v;
    return true;
}

bool get_double_(std::string_view text, std::string_view key, double& out) {
    auto p = find_key_value_pos_(text, key);
    if (p == std::string_view::npos) return false;
    const char* start = text.data() + p;
    char* endp = nullptr;
    double v = std::strtod(start, &endp);
    if (endp == start) return false;
    out = v;
    return true;
}

bool get_bool_(std::string_view text, std::string_view key, bool& out) {
    auto p = find_key_value_pos_(text, key);
    if (p == std::string_view::npos) return false;
    if (text.compare(p, 4, "true") == 0)  { out = true;  return true; }
    if (text.compare(p, 5, "false") == 0) { out = false; return true; }
    return false;
}

Meta parse_meta_(const std::filesystem::path& meta_path) {
    std::string text = slurp_file_(meta_path);
    Meta m;

    if (!get_string_(text, "magic", m.magic))
        throw BundleError("meta.json missing 'magic' in " + meta_path.string());
    if (m.magic != "mvbin")
        throw BundleError("bad magic '" + m.magic + "' in " + meta_path.string());

    long long iv = 0;
    if (!get_int_(text, "version", iv))
        throw BundleError("meta.json missing 'version'");
    m.version = static_cast<int>(iv);
    if (m.version != 1)
        throw BundleError("unsupported bundle version " + std::to_string(m.version));

    std::string dtype_str;
    if (!get_string_(text, "dtype", dtype_str))
        throw BundleError("meta.json missing 'dtype'");
    if      (dtype_str == "float16") m.dtype = DType::Float16;
    else if (dtype_str == "float32") m.dtype = DType::Float32;
    else throw BundleError("unknown dtype '" + dtype_str + "'");

    if (!get_int_(text, "dim", iv)) throw BundleError("meta.json missing 'dim'");
    m.dim = static_cast<int>(iv);

    if (!get_int_(text, "n_docs", iv)) throw BundleError("meta.json missing 'n_docs'");
    m.n_docs = static_cast<std::size_t>(iv);

    if (!get_int_(text, "n_tokens", iv)) throw BundleError("meta.json missing 'n_tokens'");
    m.n_tokens = static_cast<std::size_t>(iv);

    get_string_(text, "mode", m.mode);
    get_bool_(text, "is_query", m.is_query);
    get_bool_(text, "keep_padding", m.keep_padding);

    if (get_int_(text, "doc_maxlen", iv))   m.doc_maxlen   = static_cast<int>(iv);
    if (get_int_(text, "query_maxlen", iv)) m.query_maxlen = static_cast<int>(iv);

    get_string_(text, "encoder_ckpt", m.encoder_ckpt);
    get_string_(text, "input_tsv",    m.input_tsv);
    get_string_(text, "dataset",      m.dataset);
    get_string_(text, "split",        m.split);
    get_double_(text, "elapsed_sec",  m.elapsed_sec);

    return m;
}

// Read an entire little-endian int32/int64 array file into a vector.
template <class T>
std::vector<T> read_pod_file_(const std::filesystem::path& p, std::size_t expected_count) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw BundleError("failed to open " + p.string());
    std::streamsize sz = f.tellg();
    if (sz < 0) throw BundleError("tellg failed for " + p.string());
    if (static_cast<std::size_t>(sz) != expected_count * sizeof(T)) {
        throw BundleError(
            "size mismatch for " + p.filename().string() +
            ": got " + std::to_string(sz) +
            " bytes, expected " + std::to_string(expected_count * sizeof(T)));
    }
    std::vector<T> out(expected_count);
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char*>(out.data()), sz))
        throw BundleError("failed to read " + p.string());
    return out;
}

}  // namespace

// ---- DocView typed accessors --------------------------------------------

const std::uint16_t* DocView::as_fp16() const {
    if (dtype != DType::Float16)
        throw BundleError("DocView::as_fp16: dtype is float32, not float16");
    return static_cast<const std::uint16_t*>(data);
}

const float* DocView::as_fp32() const {
    if (dtype != DType::Float32)
        throw BundleError("DocView::as_fp32: dtype is float16, not float32");
    return static_cast<const float*>(data);
}

// ---- MVBundle lifecycle --------------------------------------------------

MVBundle::MVBundle(MVBundle&& other) noexcept { *this = std::move(other); }

MVBundle& MVBundle::operator=(MVBundle&& other) noexcept {
    if (this != &other) {
        release_mmap_();
        meta_           = std::move(other.meta_);
        path_           = std::move(other.path_);
        emb_ptr_        = other.emb_ptr_;
        emb_size_       = other.emb_size_;
        emb_fd_         = other.emb_fd_;
        doclens_        = std::move(other.doclens_);
        docids_         = std::move(other.docids_);
        offsets_        = std::move(other.offsets_);
        id_to_row_      = std::move(other.id_to_row_);
        id_index_built_ = other.id_index_built_;
        other.emb_ptr_  = nullptr;
        other.emb_size_ = 0;
        other.emb_fd_   = -1;
    }
    return *this;
}

MVBundle::~MVBundle() { release_mmap_(); }

void MVBundle::release_mmap_() noexcept {
    if (emb_ptr_ && emb_size_ > 0) {
        ::munmap(const_cast<void*>(emb_ptr_), emb_size_);
    }
    if (emb_fd_ >= 0) {
        ::close(emb_fd_);
    }
    emb_ptr_ = nullptr;
    emb_size_ = 0;
    emb_fd_ = -1;
}

MVBundle MVBundle::load(const std::filesystem::path& dir) {
    MVBundle b;
    b.path_ = dir;

    const auto meta_path = dir / "meta.json";
    if (!std::filesystem::exists(meta_path))
        throw BundleError("meta.json not found in " + dir.string());
    b.meta_ = parse_meta_(meta_path);

    const char* emb_fname =
        (b.meta_.is_query) ? "embeddings.fp32.bin" : "embeddings.fp16.bin";
    const auto emb_path = dir / emb_fname;
    if (!std::filesystem::exists(emb_path)) {
        const char* other =
            (b.meta_.is_query) ? "embeddings.fp16.bin" : "embeddings.fp32.bin";
        if (std::filesystem::exists(dir / other)) {
            throw BundleError(
                std::string("meta says is_query=") + (b.meta_.is_query ? "true" : "false") +
                " (dtype=" + dtype_name(b.meta_.dtype) + ") but found " + other +
                " instead of " + emb_fname + " in " + dir.string());
        }
        throw BundleError(std::string(emb_fname) + " not found in " + dir.string());
    }

    const std::size_t expected_emb_bytes =
        b.meta_.n_tokens * static_cast<std::size_t>(b.meta_.dim) * dtype_bytes(b.meta_.dtype);
    const auto actual_emb_bytes =
        static_cast<std::size_t>(std::filesystem::file_size(emb_path));
    if (actual_emb_bytes != expected_emb_bytes) {
        throw BundleError(
            std::string(emb_fname) + " size mismatch: got " +
            std::to_string(actual_emb_bytes) + " bytes, expected " +
            std::to_string(expected_emb_bytes) + " (n_tokens=" +
            std::to_string(b.meta_.n_tokens) + " × dim=" + std::to_string(b.meta_.dim) +
            " × " + std::to_string(dtype_bytes(b.meta_.dtype)) + ")");
    }

    // mmap embeddings read-only, sequential hint
    b.emb_fd_ = ::open(emb_path.c_str(), O_RDONLY);
    if (b.emb_fd_ < 0) {
        throw BundleError("open(" + emb_path.string() +
                          ") failed: " + std::strerror(errno));
    }
    void* p = ::mmap(nullptr, actual_emb_bytes, PROT_READ, MAP_PRIVATE, b.emb_fd_, 0);
    if (p == MAP_FAILED) {
        ::close(b.emb_fd_);
        b.emb_fd_ = -1;
        throw BundleError("mmap(" + emb_path.string() +
                          ") failed: " + std::strerror(errno));
    }
    b.emb_ptr_  = p;
    b.emb_size_ = actual_emb_bytes;
#if defined(MADV_SEQUENTIAL) && defined(MADV_DONTDUMP)
    // hints — failure is non-fatal
    ::madvise(p, actual_emb_bytes, MADV_SEQUENTIAL);
    ::madvise(p, actual_emb_bytes, MADV_DONTDUMP);
#endif

    // doclens + docids (small, kept in RAM)
    b.doclens_ = read_pod_file_<std::int32_t>(dir / "doclens.i32.bin", b.meta_.n_docs);
    b.docids_  = read_pod_file_<std::int32_t>(dir / "docids.i32.bin",  b.meta_.n_docs);

    // sanity: doclens.sum() must equal n_tokens
    std::size_t sum = 0;
    for (auto v : b.doclens_) {
        if (v < 0) throw BundleError("negative doclen at row " + std::to_string(&v - b.doclens_.data()));
        sum += static_cast<std::size_t>(v);
    }
    if (sum != b.meta_.n_tokens) {
        throw BundleError(
            "doclens.sum() = " + std::to_string(sum) +
            " != n_tokens = " + std::to_string(b.meta_.n_tokens));
    }

    // cumulative offsets (n_docs + 1)
    b.offsets_.resize(b.meta_.n_docs + 1);
    b.offsets_[0] = 0;
    for (std::size_t i = 0; i < b.meta_.n_docs; ++i) {
        b.offsets_[i + 1] = b.offsets_[i] + b.doclens_[i];
    }

    return b;
}

// ---- typed embedding views ----------------------------------------------

const std::uint16_t* MVBundle::embeddings_fp16() const {
    if (meta_.dtype != DType::Float16)
        throw BundleError("embeddings_fp16(): bundle dtype is float32");
    return static_cast<const std::uint16_t*>(emb_ptr_);
}

const float* MVBundle::embeddings_fp32() const {
    if (meta_.dtype != DType::Float32)
        throw BundleError("embeddings_fp32(): bundle dtype is float16");
    return static_cast<const float*>(emb_ptr_);
}

// ---- per-row slicing -----------------------------------------------------

DocView MVBundle::get(std::size_t row) const {
    if (row >= meta_.n_docs) {
        throw BundleError("row " + std::to_string(row) +
                          " out of range (n_docs=" + std::to_string(meta_.n_docs) + ")");
    }
    const std::size_t start_token = static_cast<std::size_t>(offsets_[row]);
    const std::size_t tokens      = static_cast<std::size_t>(doclens_[row]);
    const std::size_t bytes       = dtype_bytes(meta_.dtype);
    const auto* base = static_cast<const std::uint8_t*>(emb_ptr_);
    return DocView{
        base + start_token * meta_.dim * bytes,
        tokens,
        meta_.dim,
        meta_.dtype,
    };
}

void MVBundle::build_id_index_() {
    id_to_row_.reserve(meta_.n_docs * 2);
    for (std::size_t r = 0; r < meta_.n_docs; ++r) {
        id_to_row_.emplace(docids_[r], r);
    }
    id_index_built_ = true;
}

std::size_t MVBundle::row_of_id(std::int32_t original_id) {
    if (!id_index_built_) build_id_index_();
    auto it = id_to_row_.find(original_id);
    return it == id_to_row_.end() ? static_cast<std::size_t>(-1) : it->second;
}

DocView MVBundle::get_by_id(std::int32_t original_id) {
    auto row = row_of_id(original_id);
    if (row == static_cast<std::size_t>(-1))
        throw BundleError("id " + std::to_string(original_id) + " not in bundle");
    return get(row);
}

// ============================================================================
// MVGroundTruth implementation
// ============================================================================

namespace {

GTMeta parse_gt_meta_(const std::filesystem::path& meta_path) {
    std::string text = slurp_file_(meta_path);
    GTMeta m;

    if (!get_string_(text, "magic", m.magic))
        throw BundleError("gt meta.json missing 'magic' in " + meta_path.string());
    if (m.magic != "mvgt")
        throw BundleError("bad gt magic '" + m.magic + "' in " + meta_path.string());

    long long iv = 0;
    if (!get_int_(text, "version", iv))
        throw BundleError("gt meta.json missing 'version'");
    m.version = static_cast<int>(iv);
    if (m.version != 1)
        throw BundleError("unsupported gt version " + std::to_string(m.version));

    if (!get_int_(text, "n_queries", iv)) throw BundleError("gt meta missing 'n_queries'");
    m.n_queries = static_cast<std::size_t>(iv);
    if (!get_int_(text, "n_docs", iv))    throw BundleError("gt meta missing 'n_docs'");
    m.n_docs = static_cast<std::size_t>(iv);
    if (!get_int_(text, "top_k", iv))     throw BundleError("gt meta missing 'top_k'");
    m.top_k = static_cast<int>(iv);

    if (get_int_(text, "dim", iv))      m.dim      = static_cast<int>(iv);
    if (get_int_(text, "q_maxlen", iv)) m.q_maxlen = static_cast<int>(iv);

    std::string dt;
    if (!get_string_(text, "score_dtype", dt))
        throw BundleError("gt meta missing 'score_dtype'");
    if      (dt == "fp32" || dt == "float32") m.score_dtype = DType::Float32;
    else if (dt == "fp16" || dt == "float16") m.score_dtype = DType::Float16;
    else throw BundleError("unknown gt score_dtype '" + dt + "'");

    get_string_(text, "sort_order", m.sort_order);
    get_double_(text, "elapsed_sec", m.elapsed_sec);
    return m;
}

}  // namespace

MVGroundTruth::MVGroundTruth(MVGroundTruth&& other) noexcept { *this = std::move(other); }

MVGroundTruth& MVGroundTruth::operator=(MVGroundTruth&& other) noexcept {
    if (this != &other) {
        release_mmap_();
        meta_           = std::move(other.meta_);
        path_           = std::move(other.path_);
        scores_ptr_     = other.scores_ptr_;
        scores_size_    = other.scores_size_;
        scores_fd_      = other.scores_fd_;
        topk_indices_   = std::move(other.topk_indices_);
        topk_scores_    = std::move(other.topk_scores_);
        other.scores_ptr_  = nullptr;
        other.scores_size_ = 0;
        other.scores_fd_   = -1;
    }
    return *this;
}

MVGroundTruth::~MVGroundTruth() { release_mmap_(); }

void MVGroundTruth::release_mmap_() noexcept {
    if (scores_ptr_ && scores_size_ > 0) {
        ::munmap(const_cast<void*>(scores_ptr_), scores_size_);
    }
    if (scores_fd_ >= 0) {
        ::close(scores_fd_);
    }
    scores_ptr_ = nullptr;
    scores_size_ = 0;
    scores_fd_ = -1;
}

MVGroundTruth MVGroundTruth::load(const std::filesystem::path& dir) {
    MVGroundTruth gt;
    gt.path_ = dir;

    const auto meta_path = dir / "meta.json";
    if (!std::filesystem::exists(meta_path))
        throw BundleError("gt meta.json not found in " + dir.string());
    gt.meta_ = parse_gt_meta_(meta_path);

    const std::size_t bytes_per_score = dtype_bytes(gt.meta_.score_dtype);
    const char* scores_fname =
        (gt.meta_.score_dtype == DType::Float32) ? "scores.fp32.bin" : "scores.fp16.bin";
    const char* tops_fname =
        (gt.meta_.score_dtype == DType::Float32) ? "topk_scores.fp32.bin" : "topk_scores.fp16.bin";

    // ---- scores.bin: mmap when present ----
    // Recall evaluation only needs topk_indices. Some archived GT bundles omit
    // the much larger full score matrix while retaining top-k IDs and scores.
    const auto scores_path = dir / scores_fname;
    if (std::filesystem::exists(scores_path)) {
        const std::size_t expected_scores_bytes =
            gt.meta_.n_queries * gt.meta_.n_docs * bytes_per_score;
        const std::size_t actual_scores_bytes =
            static_cast<std::size_t>(std::filesystem::file_size(scores_path));
        if (actual_scores_bytes != expected_scores_bytes) {
            throw BundleError(
                std::string(scores_fname) + " size mismatch: got " +
                std::to_string(actual_scores_bytes) + " bytes, expected " +
                std::to_string(expected_scores_bytes));
        }

        gt.scores_fd_ = ::open(scores_path.c_str(), O_RDONLY);
        if (gt.scores_fd_ < 0)
            throw BundleError("open(" + scores_path.string() +
                              ") failed: " + std::strerror(errno));
        void* p = ::mmap(nullptr, actual_scores_bytes, PROT_READ, MAP_PRIVATE,
                         gt.scores_fd_, 0);
        if (p == MAP_FAILED) {
            ::close(gt.scores_fd_);
            gt.scores_fd_ = -1;
            throw BundleError("mmap(" + scores_path.string() +
                              ") failed: " + std::strerror(errno));
        }
        gt.scores_ptr_  = p;
        gt.scores_size_ = actual_scores_bytes;
#if defined(MADV_RANDOM)
        // GT access patterns are typically per-query (whole row); RANDOM is the
        // safer default since callers may read scattered rows.
        ::madvise(p, actual_scores_bytes, MADV_RANDOM);
#endif
    }

    // ---- topk_indices.i32.bin: in-RAM ----
    gt.topk_indices_ = read_pod_file_<std::int32_t>(
        dir / "topk_indices.i32.bin",
        gt.meta_.n_queries * static_cast<std::size_t>(gt.meta_.top_k));

    // ---- topk_scores: in-RAM, dtype-varying ----
    const auto tops_path = dir / tops_fname;
    if (!std::filesystem::exists(tops_path))
        throw BundleError(std::string(tops_fname) + " not found in " + dir.string());
    const std::size_t expected_tops_bytes =
        gt.meta_.n_queries * static_cast<std::size_t>(gt.meta_.top_k) * bytes_per_score;
    const std::size_t actual_tops_bytes =
        static_cast<std::size_t>(std::filesystem::file_size(tops_path));
    if (actual_tops_bytes != expected_tops_bytes) {
        throw BundleError(
            std::string(tops_fname) + " size mismatch: got " +
            std::to_string(actual_tops_bytes) + " bytes, expected " +
            std::to_string(expected_tops_bytes));
    }
    gt.topk_scores_.resize(expected_tops_bytes);
    {
        std::ifstream f(tops_path, std::ios::binary);
        if (!f) throw BundleError("open " + tops_path.string() + " failed");
        if (!f.read(reinterpret_cast<char*>(gt.topk_scores_.data()),
                    static_cast<std::streamsize>(expected_tops_bytes))) {
            throw BundleError("read " + tops_path.string() + " failed");
        }
    }

    return gt;
}

const float* MVGroundTruth::scores_fp32() const {
    if (meta_.score_dtype != DType::Float32)
        throw BundleError("scores_fp32(): score dtype is float16");
    if (!scores_ptr_)
        throw BundleError("scores_fp32(): full score matrix is unavailable");
    return static_cast<const float*>(scores_ptr_);
}

const std::uint16_t* MVGroundTruth::scores_fp16() const {
    if (meta_.score_dtype != DType::Float16)
        throw BundleError("scores_fp16(): score dtype is float32");
    if (!scores_ptr_)
        throw BundleError("scores_fp16(): full score matrix is unavailable");
    return static_cast<const std::uint16_t*>(scores_ptr_);
}

const void* MVGroundTruth::scores_row(std::size_t q) const {
    if (q >= meta_.n_queries) {
        throw BundleError("scores_row: q=" + std::to_string(q) +
                          " out of range (n_queries=" +
                          std::to_string(meta_.n_queries) + ")");
    }
    if (!scores_ptr_)
        throw BundleError("scores_row: full score matrix is unavailable");
    const std::size_t row_bytes = meta_.n_docs * dtype_bytes(meta_.score_dtype);
    const auto* base = static_cast<const std::uint8_t*>(scores_ptr_);
    return base + q * row_bytes;
}

const std::int32_t* MVGroundTruth::topk_indices_row(std::size_t q) const {
    if (q >= meta_.n_queries) {
        throw BundleError("topk_indices_row: q=" + std::to_string(q) + " OOR");
    }
    return topk_indices_.data() + q * static_cast<std::size_t>(meta_.top_k);
}

const float* MVGroundTruth::topk_scores_fp32() const {
    if (meta_.score_dtype != DType::Float32)
        throw BundleError("topk_scores_fp32(): score dtype is float16");
    return reinterpret_cast<const float*>(topk_scores_.data());
}

const std::uint16_t* MVGroundTruth::topk_scores_fp16() const {
    if (meta_.score_dtype != DType::Float16)
        throw BundleError("topk_scores_fp16(): score dtype is float32");
    return reinterpret_cast<const std::uint16_t*>(topk_scores_.data());
}

const void* MVGroundTruth::topk_scores_row(std::size_t q) const {
    if (q >= meta_.n_queries) {
        throw BundleError("topk_scores_row: q=" + std::to_string(q) + " OOR");
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(meta_.top_k) * dtype_bytes(meta_.score_dtype);
    return topk_scores_.data() + q * row_bytes;
}

float MVGroundTruth::recall_at_k(const std::int32_t* approx_topk,
                                 std::size_t q, std::size_t k) const {
    if (k == 0) return 0.0f;
    if (static_cast<int>(k) > meta_.top_k) {
        throw BundleError("recall_at_k: k=" + std::to_string(k) +
                          " > stored top_k=" + std::to_string(meta_.top_k));
    }
    return mv::recall_at_k(approx_topk, topk_indices_row(q), k);
}

float recall_at_k(const std::int32_t* approx_topk,
                  const std::int32_t* gt_topk,
                  std::size_t k) {
    if (k == 0) return 0.0f;
    std::unordered_set<std::int32_t> s(approx_topk, approx_topk + k);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < k; ++i) {
        if (s.find(gt_topk[i]) != s.end()) ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(k);
}

}  // namespace mv
