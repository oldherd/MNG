#pragma once

#include "common/ann_hnswlib.hpp"
#include "common/index_io.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace igp {

constexpr const char* kLayout = "igp_pg_residual_sq_v1";

inline std::size_t packed_residual_dim(int dim, int nbits) {
    return ((std::size_t)dim * (std::size_t)nbits + 7u) / 8u;
}

struct IGPIndex {
    int dim = 0;
    int nbits = 2;
    int n_centroids = 0;
    int pg_m = 64;
    int ef_construction = 200;
    std::size_t n_docs = 0;
    std::size_t n_tokens = 0;
    std::size_t packed_dim = 0;
    double build_time_sec = 0.0;
    std::string input_doc_bundle;

    std::vector<float> centroids;              // n_centroids * dim
    std::vector<float> bucket_cutoffs;         // 2^nbits - 1
    std::vector<float> bucket_weights;         // 2^nbits
    std::vector<int32_t> doclens;              // n_docs
    std::vector<int64_t> offsets;              // n_docs + 1
    std::vector<int32_t> assignments;          // n_tokens, token order
    std::vector<uint8_t> residuals;            // n_tokens * packed_dim, token order
    std::vector<std::vector<int32_t>> centroid_to_pids; // unique pids per centroid

    void validate() const {
        if (dim <= 0 || nbits <= 0 || n_centroids <= 0) {
            throw std::runtime_error("invalid IGP dimensions in index");
        }
        if (packed_dim != packed_residual_dim(dim, nbits)) {
            throw std::runtime_error("invalid IGP packed_dim");
        }
        if (centroids.size() != (std::size_t)n_centroids * (std::size_t)dim) {
            throw std::runtime_error("IGP centroids size mismatch");
        }
        if (bucket_cutoffs.size() != ((1u << nbits) - 1u)) {
            throw std::runtime_error("IGP bucket_cutoffs size mismatch");
        }
        if (bucket_weights.size() != (1u << nbits)) {
            throw std::runtime_error("IGP bucket_weights size mismatch");
        }
        if (doclens.size() != n_docs || offsets.size() != n_docs + 1u) {
            throw std::runtime_error("IGP doclens/offsets size mismatch");
        }
        if (assignments.size() != n_tokens || residuals.size() != n_tokens * packed_dim) {
            throw std::runtime_error("IGP token payload size mismatch");
        }
        if (centroid_to_pids.size() != (std::size_t)n_centroids) {
            throw std::runtime_error("IGP centroid_to_pids size mismatch");
        }
    }

    void save(const std::string& dir,
              baseline::HnswlibIndex& pg,
              const std::vector<std::pair<std::string, std::string>>& extra_meta = {}) const {
        validate();
        baseline::ensure_dir(dir);
        baseline::write_vec_bin(dir + "/centroids.fp32.bin", centroids);
        baseline::write_vec_bin(dir + "/bucket_cutoffs.fp32.bin", bucket_cutoffs);
        baseline::write_vec_bin(dir + "/bucket_weights.fp32.bin", bucket_weights);
        baseline::write_vec_bin(dir + "/doclens.i32.bin", doclens);
        baseline::write_vec_bin(dir + "/offsets.i64.bin", offsets);
        baseline::write_vec_bin(dir + "/assignments.i32.bin", assignments);
        baseline::write_vec_bin(dir + "/residuals.u8.bin", residuals);
        baseline::write_varlists(dir + "/centroid_to_pids.bin", centroid_to_pids);
        pg.save(dir + "/pg.hnsw.bin");

        std::size_t total_centroid_pid_entries = 0;
        for (const auto& list : centroid_to_pids) total_centroid_pid_entries += list.size();

        std::vector<std::pair<std::string, std::string>> meta = {
            {"method", "igp"},
            {"layout", kLayout},
            {"compression", "residual_scalar_quantization"},
            {"candidate_generation", "proximity_graph_centroid_probe"},
            {"final_scoring", "residual_decompressed_maxsim"},
            {"metric", "ip"},
            {"dim", std::to_string(dim)},
            {"nbits", std::to_string(nbits)},
            {"n_centroids", std::to_string(n_centroids)},
            {"n_docs", std::to_string(n_docs)},
            {"n_tokens", std::to_string(n_tokens)},
            {"packed_dim", std::to_string(packed_dim)},
            {"pg_m", std::to_string(pg_m)},
            {"ef_construction", std::to_string(ef_construction)},
            {"centroid_pid_entries", std::to_string(total_centroid_pid_entries)},
            {"avg_docs_per_centroid", std::to_string(
                (double)total_centroid_pid_entries / (double)std::max(1, n_centroids))},
            {"build_time_sec", std::to_string(build_time_sec)},
            {"input_doc_bundle", input_doc_bundle},
        };
        meta.insert(meta.end(), extra_meta.begin(), extra_meta.end());
        baseline::write_meta(dir, meta);
    }

    static IGPIndex load(const std::string& dir) {
        auto meta = baseline::read_meta(dir);
        if (meta["layout"] != kLayout) {
            throw std::runtime_error("IGP index layout mismatch: expected " +
                                     std::string(kLayout) + ", got " + meta["layout"]);
        }

        IGPIndex idx;
        idx.dim = std::stoi(meta.at("dim"));
        idx.nbits = std::stoi(meta.at("nbits"));
        idx.n_centroids = std::stoi(meta.at("n_centroids"));
        idx.n_docs = std::stoull(meta.at("n_docs"));
        idx.n_tokens = std::stoull(meta.at("n_tokens"));
        idx.packed_dim = std::stoull(meta.at("packed_dim"));
        idx.pg_m = std::stoi(meta.at("pg_m"));
        idx.ef_construction = std::stoi(meta.at("ef_construction"));
        idx.build_time_sec = std::stod(meta.at("build_time_sec"));
        auto it = meta.find("input_doc_bundle");
        if (it != meta.end()) idx.input_doc_bundle = it->second;

        idx.centroids = baseline::read_vec_bin<float>(dir + "/centroids.fp32.bin");
        idx.bucket_cutoffs = baseline::read_vec_bin<float>(dir + "/bucket_cutoffs.fp32.bin");
        idx.bucket_weights = baseline::read_vec_bin<float>(dir + "/bucket_weights.fp32.bin");
        idx.doclens = baseline::read_vec_bin<int32_t>(dir + "/doclens.i32.bin");
        idx.offsets = baseline::read_vec_bin<int64_t>(dir + "/offsets.i64.bin");
        idx.assignments = baseline::read_vec_bin<int32_t>(dir + "/assignments.i32.bin");
        idx.residuals = baseline::read_vec_bin<uint8_t>(dir + "/residuals.u8.bin");
        idx.centroid_to_pids = baseline::read_varlists<int32_t>(dir + "/centroid_to_pids.bin");
        idx.validate();
        return idx;
    }
};

inline std::size_t logical_index_bytes(const std::string& dir) {
    return baseline::directory_file_size(dir);
}

inline std::unique_ptr<baseline::HnswlibIndex> load_pg(const std::string& dir, int dim) {
    return baseline::HnswlibIndex::load_from_file(dir + "/pg.hnsw.bin", dim);
}

}  // namespace igp
