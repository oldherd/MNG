#pragma once

#include "common/index_io.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace warp {

constexpr const char* kLayout = "warp_compacted_residual_v1";

inline std::size_t packed_residual_dim(int dim, int nbits) {
    return ((std::size_t)dim * (std::size_t)nbits + 7u) / 8u;
}

struct WARPIndex {
    int dim = 0;
    int nbits = 4;
    int n_centroids = 0;
    std::size_t n_docs = 0;
    std::size_t n_tokens = 0;
    std::size_t packed_dim = 0;
    double build_time_sec = 0.0;
    std::string input_doc_bundle;

    std::vector<float> centroids;       // n_centroids * dim
    std::vector<float> bucket_cutoffs;  // 2^nbits - 1
    std::vector<float> bucket_weights;  // 2^nbits
    std::vector<int64_t> sizes;         // n_centroids token counts
    std::vector<int64_t> offsets;       // n_centroids + 1 compacted offsets
    std::vector<int32_t> pids;          // n_tokens, compacted by centroid
    std::vector<uint8_t> residuals;     // n_tokens * packed_dim, compacted by centroid

    void validate() const {
        if (dim <= 0 || nbits <= 0 || n_centroids <= 0) {
            throw std::runtime_error("invalid WARP dimensions in index");
        }
        if (packed_dim != packed_residual_dim(dim, nbits)) {
            throw std::runtime_error("invalid WARP packed_dim");
        }
        if (centroids.size() != (std::size_t)n_centroids * (std::size_t)dim) {
            throw std::runtime_error("WARP centroids size mismatch");
        }
        if (bucket_weights.size() != (1u << nbits)) {
            throw std::runtime_error("WARP bucket_weights size mismatch");
        }
        if (bucket_cutoffs.size() != ((1u << nbits) - 1u)) {
            throw std::runtime_error("WARP bucket_cutoffs size mismatch");
        }
        if (sizes.size() != (std::size_t)n_centroids || offsets.size() != (std::size_t)n_centroids + 1u) {
            throw std::runtime_error("WARP compacted offsets size mismatch");
        }
        if (pids.size() != n_tokens || residuals.size() != n_tokens * packed_dim) {
            throw std::runtime_error("WARP compacted payload size mismatch");
        }
    }

    void save(const std::string& dir,
              const std::vector<std::pair<std::string, std::string>>& extra_meta = {}) const {
        validate();
        baseline::ensure_dir(dir);
        baseline::write_vec_bin(dir + "/centroids.fp32.bin", centroids);
        baseline::write_vec_bin(dir + "/bucket_cutoffs.fp32.bin", bucket_cutoffs);
        baseline::write_vec_bin(dir + "/bucket_weights.fp32.bin", bucket_weights);
        baseline::write_vec_bin(dir + "/sizes.i64.bin", sizes);
        baseline::write_vec_bin(dir + "/offsets.i64.bin", offsets);
        baseline::write_vec_bin(dir + "/pids.i32.bin", pids);
        baseline::write_vec_bin(dir + "/residuals.u8.bin", residuals);

        std::vector<std::pair<std::string, std::string>> meta = {
            {"method", "warp"},
            {"layout", kLayout},
            {"compression", "compacted_bbit_residual"},
            {"metric", "ip"},
            {"dim", std::to_string(dim)},
            {"nbits", std::to_string(nbits)},
            {"n_centroids", std::to_string(n_centroids)},
            {"n_docs", std::to_string(n_docs)},
            {"n_tokens", std::to_string(n_tokens)},
            {"packed_dim", std::to_string(packed_dim)},
            {"build_time_sec", std::to_string(build_time_sec)},
            {"input_doc_bundle", input_doc_bundle},
        };
        meta.insert(meta.end(), extra_meta.begin(), extra_meta.end());
        baseline::write_meta(dir, meta);
    }

    static WARPIndex load(const std::string& dir) {
        auto meta = baseline::read_meta(dir);
        if (meta["layout"] != kLayout) {
            throw std::runtime_error("WARP index layout mismatch: expected " +
                                     std::string(kLayout) + ", got " + meta["layout"]);
        }

        WARPIndex idx;
        idx.dim = std::stoi(meta.at("dim"));
        idx.nbits = std::stoi(meta.at("nbits"));
        idx.n_centroids = std::stoi(meta.at("n_centroids"));
        idx.n_docs = std::stoull(meta.at("n_docs"));
        idx.n_tokens = std::stoull(meta.at("n_tokens"));
        idx.packed_dim = std::stoull(meta.at("packed_dim"));
        idx.build_time_sec = std::stod(meta.at("build_time_sec"));
        auto it = meta.find("input_doc_bundle");
        if (it != meta.end()) idx.input_doc_bundle = it->second;

        idx.centroids = baseline::read_vec_bin<float>(dir + "/centroids.fp32.bin");
        idx.bucket_cutoffs = baseline::read_vec_bin<float>(dir + "/bucket_cutoffs.fp32.bin");
        idx.bucket_weights = baseline::read_vec_bin<float>(dir + "/bucket_weights.fp32.bin");
        idx.sizes = baseline::read_vec_bin<int64_t>(dir + "/sizes.i64.bin");
        idx.offsets = baseline::read_vec_bin<int64_t>(dir + "/offsets.i64.bin");
        idx.pids = baseline::read_vec_bin<int32_t>(dir + "/pids.i32.bin");
        idx.residuals = baseline::read_vec_bin<uint8_t>(dir + "/residuals.u8.bin");
        idx.validate();
        return idx;
    }
};

inline std::size_t logical_index_bytes(const std::string& dir) {
    return baseline::directory_file_size(dir);
}

}  // namespace warp
