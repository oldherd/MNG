// DESSERT indexer — IVF pre-filter + per-document TinyTable sketches.

#include "mvbundle.hpp"
#include "common/fp16_utils.hpp"
#include "common/kmeans.hpp"
#include "common/timer.hpp"
#include "common/bench_utils.hpp"
#include "dessert_index.hpp"
#include "dessert_tinytable.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {

class MappedU16Output {
public:
    MappedU16Output(const std::string& final_path, uint64_t entries)
        : final_path_(final_path), temp_path_(final_path + ".tmp"), entries_(entries) {
        fd_ = ::open(temp_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) fail("open");

        const uint64_t bytes = entries_ * sizeof(uint16_t);
        if (bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("DESSERT TinyTable output exceeds off_t range");
        }
        if (::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) fail("ftruncate");
        if (bytes > 0) {
            void* mapped = ::mmap(nullptr, static_cast<size_t>(bytes), PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd_, 0);
            if (mapped == MAP_FAILED) fail("mmap");
            data_ = static_cast<uint16_t*>(mapped);
        }
    }

    MappedU16Output(const MappedU16Output&) = delete;
    MappedU16Output& operator=(const MappedU16Output&) = delete;

    ~MappedU16Output() {
        close_noexcept();
    }

    uint16_t* data() noexcept { return data_; }

    void finish() {
        const size_t bytes = static_cast<size_t>(entries_ * sizeof(uint16_t));
        if (data_ != nullptr) {
            if (::msync(data_, bytes, MS_SYNC) != 0) fail("msync");
            if (::munmap(data_, bytes) != 0) fail("munmap");
            data_ = nullptr;
        }
        if (::fsync(fd_) != 0) fail("fsync");
        if (::close(fd_) != 0) {
            fd_ = -1;
            fail("close");
        }
        fd_ = -1;
        if (::rename(temp_path_.c_str(), final_path_.c_str()) != 0) fail("rename");
        finished_ = true;
    }

private:
    [[noreturn]] void fail(const char* operation) {
        const int error = errno;
        close_noexcept();
        throw std::runtime_error(std::string("DESSERT TinyTable ") + operation +
                                 " failed for " + temp_path_ + ": " +
                                 std::strerror(error));
    }

    void close_noexcept() noexcept {
        const size_t bytes = static_cast<size_t>(entries_ * sizeof(uint16_t));
        if (data_ != nullptr) {
            ::munmap(data_, bytes);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        if (!finished_) ::unlink(temp_path_.c_str());
    }

    std::string final_path_;
    std::string temp_path_;
    uint64_t entries_ = 0;
    int fd_ = -1;
    uint16_t* data_ = nullptr;
    bool finished_ = false;
};

void convert_fp16_to_fp32(const uint16_t* input, size_t n_values, float* output) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_values; ++i) {
        output[i] = baseline::fp16_to_fp32(input[i]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    auto args = baseline::parse_args(argc, argv);
    if (args.doc_bundle.empty() || args.index_dir.empty()) {
        std::cerr << "Usage: indexer_dessert --doc <bundle> --index_dir <dir> "
                  << "[--param L=32] [--param C=6] [--param n_filter=0]\n";
        return 1;
    }

    int L             = baseline::get_extra_int(args, "L", 32);
    int C             = baseline::get_extra_int(args, "C", 6);
    int n_filter      = baseline::get_extra_int(args, "n_filter", 0);
    int seed          = baseline::get_extra_int(args, "seed", 42);
    int kmeans_iter   = baseline::get_extra_int(args, "kmeans_iter", 20);
    int kmeans_sample = baseline::get_extra_int(args, "kmeans_sample", 500000);
    size_t assignment_batch_tokens = static_cast<size_t>(baseline::get_extra_int(
        args, "assignment_batch_tokens", 262144));

    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    const int dim = doc_bundle.dim();
    const size_t n_docs = doc_bundle.n_docs();
    const size_t n_tokens = doc_bundle.n_tokens();
    std::cerr << "[dessert-indexer] Docs=" << n_docs << " tokens=" << n_tokens << " dim=" << dim << "\n";

    if (C <= 0 || C >= 16) {
        std::cerr << "[dessert-indexer] ERROR: C must be in [1, 15] for uint16 TinyTable buckets.\n";
        return 1;
    }
    if (L <= 0) {
        std::cerr << "[dessert-indexer] ERROR: L must be positive.\n";
        return 1;
    }
    if (assignment_batch_tokens == 0) {
        std::cerr << "[dessert-indexer] ERROR: assignment_batch_tokens must be positive.\n";
        return 1;
    }

    if (n_filter == 0) {
        n_filter = std::min(16384, (int)(std::sqrt((double)n_tokens) * 4));
        n_filter = std::max(256, n_filter);
    }
    const uint32_t range = 1u << C;
    std::cerr << "[dessert-indexer] L=" << L << " C=" << C
              << " range=" << range << " n_filter=" << n_filter << "\n";

    baseline::Timer build_timer;
    build_timer.start();

    // Part A: k-means IVF pre-filter, matching DESSERT's candidate filter.
    baseline::Timer centroid_timer;
    centroid_timer.start();
    std::cerr << "[dessert-indexer] Building IVF pre-filter...\n";
    size_t sample_n = std::min((size_t)kmeans_sample, n_tokens);
    std::vector<float> sample_data(sample_n * dim);
    size_t filled = 0;
    for (size_t di = 0; di < n_docs && filled < sample_n; ++di) {
        auto dview = doc_bundle.get(di);
        const uint16_t* data = dview.as_fp16();
        int ntok = (int)dview.tokens;
        for (int t = 0; t < ntok && filled < sample_n; ++t) {
            for (int d = 0; d < dim; ++d) {
                sample_data[filled * dim + d] = baseline::fp16_to_fp32(data[t * dim + d]);
            }
            ++filled;
        }
    }

    auto filter_centroids = baseline::KMeans::fit(
        sample_data.data(), sample_n, dim, n_filter, kmeans_iter, seed);
    sample_data.clear();
    sample_data.shrink_to_fit();
    centroid_timer.stop();
    double centroid_build_sec = centroid_timer.elapsed_sec();
    std::cerr << "[dessert-indexer] Built k-means centroids in "
              << centroid_build_sec << "s\n";

    std::cerr << "[dessert-indexer] Streaming fp16 tokens in batches of at most "
              << assignment_batch_tokens << " for IVF assignment...\n";
    std::vector<uint32_t> doc_n_tokens(n_docs);
    for (size_t di = 0; di < n_docs; ++di) {
        auto dview = doc_bundle.get(di);
        doc_n_tokens[di] = static_cast<uint32_t>(dview.tokens);
    }

    std::vector<std::vector<int32_t>> ivf_lists(n_filter);
    size_t batch_begin_doc = 0;
    size_t assigned_tokens = 0;
    size_t next_progress_tokens = std::min(n_tokens, static_cast<size_t>(1000000));
    while (batch_begin_doc < n_docs) {
        size_t batch_end_doc = batch_begin_doc;
        size_t batch_n_tokens = 0;
        while (batch_end_doc < n_docs) {
            const size_t ntok = doc_n_tokens[batch_end_doc];
            if (batch_end_doc > batch_begin_doc &&
                batch_n_tokens + ntok > assignment_batch_tokens) {
                break;
            }
            batch_n_tokens += ntok;
            ++batch_end_doc;
        }

        std::vector<float> batch_fp32(batch_n_tokens * static_cast<size_t>(dim));
        std::vector<size_t> batch_doc_offsets(batch_end_doc - batch_begin_doc + 1, 0);
        size_t batch_token_offset = 0;
        for (size_t di = batch_begin_doc; di < batch_end_doc; ++di) {
            batch_doc_offsets[di - batch_begin_doc] = batch_token_offset;
            batch_token_offset += doc_n_tokens[di];
        }
        batch_doc_offsets.back() = batch_token_offset;
        const uint16_t* batch_fp16 =
            doc_bundle.embeddings_fp16() + assigned_tokens * static_cast<size_t>(dim);
        convert_fp16_to_fp32(batch_fp16, batch_n_tokens * static_cast<size_t>(dim),
                             batch_fp32.data());

        auto assignments = baseline::KMeans::assign(
            batch_fp32.data(), batch_n_tokens, dim, filter_centroids.data(), n_filter);
        std::vector<int32_t> unique_centroids;
        for (size_t di = batch_begin_doc; di < batch_end_doc; ++di) {
            const size_t local = di - batch_begin_doc;
            unique_centroids.assign(assignments.begin() + batch_doc_offsets[local],
                                    assignments.begin() + batch_doc_offsets[local + 1]);
            std::sort(unique_centroids.begin(), unique_centroids.end());
            unique_centroids.erase(
                std::unique(unique_centroids.begin(), unique_centroids.end()),
                unique_centroids.end());
            for (int32_t centroid : unique_centroids) {
                ivf_lists[centroid].push_back(static_cast<int32_t>(di));
            }
        }

        assigned_tokens += batch_n_tokens;
        batch_begin_doc = batch_end_doc;
        if (assigned_tokens >= next_progress_tokens || assigned_tokens == n_tokens) {
            std::cerr << "\r[dessert-indexer]   IVF assigned " << assigned_tokens << "/"
                      << n_tokens << " tokens" << std::flush;
            while (next_progress_tokens <= assigned_tokens && next_progress_tokens < n_tokens) {
                next_progress_tokens = std::min(n_tokens, next_progress_tokens + 1000000);
            }
        }
    }
    std::cerr << "\n";

    // Part B: DESSERT sketches. This follows the official TinyTable layout:
    // each document owns compact per-table hash buckets, then query scoring
    // counts collisions and applies the p^(1/C) estimator.
    dessert::SparseRandomProjection srp(dim, C, L, (uint32_t)seed);
    std::vector<uint64_t> doc_table_offsets(n_docs + 1, 0);
    for (size_t di = 0; di < n_docs; ++di) {
        uint64_t ntok = doc_n_tokens[di];
        doc_table_offsets[di + 1] =
            doc_table_offsets[di] + (uint64_t)L * ((uint64_t)range + 1 + ntok);
    }

    std::cerr << "[dessert-indexer] Building per-document TinyTables: "
              << doc_table_offsets.back() << " uint16 entries\n";
    baseline::ensure_dir(args.index_dir);
    const std::string tinytable_path = args.index_dir + "/doc_tinytables.u16.bin";
    MappedU16Output tinytable_output(tinytable_path, doc_table_offsets.back());
    uint16_t* doc_table_data = tinytable_output.data();

    #pragma omp parallel
    {
        std::vector<float> doc_fp32;
        #pragma omp for schedule(dynamic, 256)
        for (size_t di = 0; di < n_docs; ++di) {
            const auto dview = doc_bundle.get(di);
            const uint32_t ntok = doc_n_tokens[di];
            doc_fp32.resize(static_cast<size_t>(ntok) * dim);
            for (size_t i = 0; i < doc_fp32.size(); ++i) {
                doc_fp32[i] = baseline::fp16_to_fp32(dview.as_fp16()[i]);
            }
            auto hashes = dessert::hash_vectors_serial(srp, doc_fp32.data(), (int)ntok, dim);
            auto table = dessert::build_tiny_table(hashes, ntok, (uint32_t)L, range);
            std::copy(table.begin(), table.end(), doc_table_data + doc_table_offsets[di]);
            if (di % 5000 == 0) {
                #pragma omp critical
                std::cerr << "\r[dessert-indexer]   TinyTable doc " << di << "/"
                          << n_docs << std::flush;
            }
        }
    }
    std::cerr << "\n";
    tinytable_output.finish();

    build_timer.stop();
    double build_sec = build_timer.elapsed_sec();

    dessert::DESSERTIndex idx;
    idx.dim = dim;
    idx.L = L;
    idx.C = C;
    idx.srp_sample_size = srp.sample_size;
    idx.n_filter = n_filter;
    idx.n_docs = n_docs;
    idx.n_tokens = n_tokens;
    idx.filter_centroids = std::move(filter_centroids);
    idx.ivf_lists = std::move(ivf_lists);
    idx.srp_random_bits = std::move(srp.random_bits);
    idx.srp_hash_indices = std::move(srp.hash_indices);
    idx.doc_table_offsets = std::move(doc_table_offsets);
    idx.doc_n_tokens = std::move(doc_n_tokens);
    idx.save(args.index_dir, build_sec, centroid_build_sec,
             "kmeans", kmeans_iter, kmeans_sample, seed,
             assignment_batch_tokens, false);

    std::cerr << "[dessert-indexer] Done in " << build_sec << "s\n";
    return 0;
}
