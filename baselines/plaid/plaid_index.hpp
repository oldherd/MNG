#pragma once

#include "common/index_io.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plaid {

constexpr const char* kLayout = "plaid_colbertv2_residual_v1";

inline std::size_t packed_residual_dim(int dim, int nbits) {
    return ((std::size_t)dim * (std::size_t)nbits + 7u) / 8u;
}

struct PLAIDIndex {
    int dim = 0;
    int nbits = 2;
    int n_centroids = 0;
    std::size_t n_docs = 0;
    std::size_t n_tokens = 0;
    std::size_t packed_dim = 0;
    double build_time_sec = 0.0;
    std::string input_doc_bundle;

    std::vector<float> centroids;                 // n_centroids * dim
    std::vector<float> bucket_cutoffs;            // 2^nbits - 1 global residual cutoffs
    std::vector<float> bucket_weights;            // 2^nbits global residual reconstruction weights
    std::vector<int32_t> doclens;                 // n_docs
    std::vector<int64_t> offsets;                 // n_docs + 1 token offsets
    std::vector<int32_t> codes;                   // n_tokens centroid ids
    std::vector<uint8_t> residuals;               // n_tokens * packed_dim packed residual bucket ids
    std::vector<std::vector<int32_t>> ivf_pid_lists; // centroid -> unique passage ids

    void validate() const {
        if (dim <= 0 || nbits <= 0 || n_centroids <= 0) {
            throw std::runtime_error("invalid PLAID dimensions in index");
        }
        if (packed_dim != packed_residual_dim(dim, nbits)) {
            throw std::runtime_error("invalid PLAID packed_dim");
        }
        if (centroids.size() != (std::size_t)n_centroids * (std::size_t)dim) {
            throw std::runtime_error("PLAID centroids size mismatch");
        }
        if (bucket_weights.size() != (1u << nbits)) {
            throw std::runtime_error("PLAID bucket_weights size mismatch");
        }
        if (bucket_cutoffs.size() != ((1u << nbits) - 1u)) {
            throw std::runtime_error("PLAID bucket_cutoffs size mismatch");
        }
        if (doclens.size() != n_docs || offsets.size() != n_docs + 1) {
            throw std::runtime_error("PLAID doclens/offsets size mismatch");
        }
        if (codes.size() != n_tokens) {
            throw std::runtime_error("PLAID codes size mismatch");
        }
        if (residuals.size() != n_tokens * packed_dim) {
            throw std::runtime_error("PLAID residuals size mismatch");
        }
        if (ivf_pid_lists.size() != (std::size_t)n_centroids) {
            throw std::runtime_error("PLAID ivf list count mismatch");
        }
    }

    void save(const std::string& dir,
              const std::vector<std::pair<std::string, std::string>>& extra_meta = {}) const {
        validate();
        baseline::ensure_dir(dir);
        baseline::write_vec_bin(dir + "/centroids.fp32.bin", centroids);
        baseline::write_vec_bin(dir + "/bucket_cutoffs.fp32.bin", bucket_cutoffs);
        baseline::write_vec_bin(dir + "/bucket_weights.fp32.bin", bucket_weights);
        baseline::write_vec_bin(dir + "/doclens.i32.bin", doclens);
        baseline::write_vec_bin(dir + "/offsets.i64.bin", offsets);
        baseline::write_vec_bin(dir + "/codes.i32.bin", codes);
        baseline::write_vec_bin(dir + "/residuals.u8.bin", residuals);
        baseline::write_varlists(dir + "/ivf_pid_lists.bin", ivf_pid_lists);

        std::vector<std::pair<std::string, std::string>> meta = {
            {"method", "plaid"},
            {"layout", kLayout},
            {"compression", "colbertv2_residual"},
            {"final_scoring", "residual_decompressed_maxsim"},
            {"exact_refine", "false"},
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

    static PLAIDIndex load(const std::string& dir) {
        auto meta = baseline::read_meta(dir);
        if (meta["layout"] != kLayout) {
            throw std::runtime_error("PLAID index layout mismatch: expected " +
                                     std::string(kLayout) + ", got " + meta["layout"]);
        }

        PLAIDIndex idx;
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
        idx.doclens = baseline::read_vec_bin<int32_t>(dir + "/doclens.i32.bin");
        idx.offsets = baseline::read_vec_bin<int64_t>(dir + "/offsets.i64.bin");
        idx.codes = baseline::read_vec_bin<int32_t>(dir + "/codes.i32.bin");
        idx.residuals = baseline::read_vec_bin<uint8_t>(dir + "/residuals.u8.bin");
        idx.ivf_pid_lists = baseline::read_varlists<int32_t>(dir + "/ivf_pid_lists.bin");
        idx.validate();
        return idx;
    }
};

inline std::size_t logical_index_bytes(const std::string& dir) {
    return baseline::directory_file_size(dir);
}

}  // namespace plaid
