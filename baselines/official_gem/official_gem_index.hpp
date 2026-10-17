#pragma once

#include "common/index_io.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace official_gem {

constexpr const char* kLayout = "gem_official_hnsw_adapter_v1";
constexpr const char* kHnswFile = "hnsw_0.bin";

struct OfficialGEMIndex {
    int dim = 0;
    int n_cquant = 0;
    int n_cindex = 0;
    int fixed_ctop = 3;
    int M = 24;
    int ef_construction = 80;
    int nprobe = 4;
    std::size_t n_docs = 0;
    std::size_t n_tokens = 0;
    double build_time_sec = 0.0;
    std::string input_doc_bundle;

    std::vector<float> cquant_centroids;       // n_cquant * dim
    std::vector<float> cindex_centroids;       // n_cindex * dim
    std::vector<int32_t> cquant_to_cindex;     // n_cquant
    std::vector<int32_t> doclens;              // n_docs
    std::vector<int64_t> offsets;              // n_docs + 1
    std::vector<int32_t> token_codes;          // n_tokens, Cquant ids
    std::vector<std::vector<int32_t>> cluster_to_docs;  // Cindex -> doc ids

    std::string hnsw_path(const std::string& dir) const {
        return dir + "/" + kHnswFile;
    }

    void validate(bool require_hnsw = true, const std::string& dir = "") const {
        if (dim <= 0 || n_cquant <= 0 || n_cindex <= 0 || fixed_ctop <= 0 ||
            M <= 0 || ef_construction <= 0 || nprobe <= 0) {
            throw std::runtime_error("invalid official GEM dimensions or parameters");
        }
        if (cquant_centroids.size() != (std::size_t)n_cquant * (std::size_t)dim) {
            throw std::runtime_error("official GEM Cquant centroid size mismatch");
        }
        if (cindex_centroids.size() != (std::size_t)n_cindex * (std::size_t)dim) {
            throw std::runtime_error("official GEM Cindex centroid size mismatch");
        }
        if (cquant_to_cindex.size() != (std::size_t)n_cquant) {
            throw std::runtime_error("official GEM Cquant-to-Cindex size mismatch");
        }
        if (doclens.size() != n_docs || offsets.size() != n_docs + 1u) {
            throw std::runtime_error("official GEM doclens/offsets size mismatch");
        }
        if (token_codes.size() != n_tokens) {
            throw std::runtime_error("official GEM token code size mismatch");
        }
        if (cluster_to_docs.size() != (std::size_t)n_cindex) {
            throw std::runtime_error("official GEM cluster list count mismatch");
        }
        if (require_hnsw && !dir.empty() && baseline::file_size(hnsw_path(dir)) == 0) {
            throw std::runtime_error("official GEM HNSW file missing: " + hnsw_path(dir));
        }
    }

    void save_metadata(const std::string& dir,
                       const std::vector<std::pair<std::string, std::string>>& extra_meta = {}) const {
        validate(false);
        baseline::write_vec_bin(dir + "/cquant_centroids.fp32.bin", cquant_centroids);
        baseline::write_vec_bin(dir + "/cindex_centroids.fp32.bin", cindex_centroids);
        baseline::write_vec_bin(dir + "/cquant_to_cindex.i32.bin", cquant_to_cindex);
        baseline::write_vec_bin(dir + "/doclens.i32.bin", doclens);
        baseline::write_vec_bin(dir + "/offsets.i64.bin", offsets);
        baseline::write_vec_bin(dir + "/token_codes.i32.bin", token_codes);
        baseline::write_varlists(dir + "/cluster_to_docs.bin", cluster_to_docs);

        std::vector<std::pair<std::string, std::string>> meta = {
            {"method", "gem_official_hnsw_adapter"},
            {"formal_baseline", "false"},
            {"layout", kLayout},
            {"variant", "Official GEM modified hnswlib graph/search path with mvbundle adapter"},
            {"official_upstream_component", "sigmod26gem/hnswlib"},
            {"mvbundle_adapter", "true"},
            {"graph_build_metric", "official_cluster_emd"},
            {"search_metric", "official_qch_cluster_search"},
            {"final_scoring", "raw_mvbundle_exact_maxsim"},
            {"metric", "inner_product"},
            {"shared_doc_store", "mvbundle"},
            {"embedding_source", "shared_mvbundle_embeddings"},
            {"dim", std::to_string(dim)},
            {"n_cquant", std::to_string(n_cquant)},
            {"n_cindex", std::to_string(n_cindex)},
            {"fixed_ctop", std::to_string(fixed_ctop)},
            {"M", std::to_string(M)},
            {"ef_construction", std::to_string(ef_construction)},
            {"nprobe", std::to_string(nprobe)},
            {"n_docs", std::to_string(n_docs)},
            {"n_tokens", std::to_string(n_tokens)},
            {"build_time_sec", std::to_string(build_time_sec)},
            {"logical_file_bytes", std::to_string(baseline::directory_file_size(dir))},
            {"input_doc_bundle", input_doc_bundle},
            {"audit_note", "Adapter uses the official GEM hnswlib graph/search implementation, but data loading and centroid training are wired to this project's mvbundle harness; do not mark as a completed formal paper run until launched through a formal profile script."},
        };
        meta.insert(meta.end(), extra_meta.begin(), extra_meta.end());
        baseline::write_meta(dir, meta);

        for (int rewrite = 0; rewrite < 4; ++rewrite) {
            const std::size_t logical_bytes = baseline::directory_file_size(dir);
            bool changed = false;
            for (auto& kv : meta) {
                if (kv.first == "logical_file_bytes" && kv.second != std::to_string(logical_bytes)) {
                    kv.second = std::to_string(logical_bytes);
                    changed = true;
                    break;
                }
            }
            if (!changed) break;
            baseline::write_meta(dir, meta);
        }
    }

    static OfficialGEMIndex load(const std::string& dir) {
        auto meta = baseline::read_meta(dir);
        if (meta["layout"] != kLayout) {
            throw std::runtime_error("official GEM index layout mismatch: expected " +
                                     std::string(kLayout) + ", got " + meta["layout"]);
        }

        OfficialGEMIndex idx;
        idx.dim = std::stoi(meta.at("dim"));
        idx.n_cquant = std::stoi(meta.at("n_cquant"));
        idx.n_cindex = std::stoi(meta.at("n_cindex"));
        idx.fixed_ctop = std::stoi(meta.at("fixed_ctop"));
        idx.M = std::stoi(meta.at("M"));
        idx.ef_construction = std::stoi(meta.at("ef_construction"));
        idx.nprobe = std::stoi(meta.at("nprobe"));
        idx.n_docs = std::stoull(meta.at("n_docs"));
        idx.n_tokens = std::stoull(meta.at("n_tokens"));
        idx.build_time_sec = std::stod(meta.at("build_time_sec"));
        auto it = meta.find("input_doc_bundle");
        if (it != meta.end()) idx.input_doc_bundle = it->second;

        idx.cquant_centroids = baseline::read_vec_bin<float>(dir + "/cquant_centroids.fp32.bin");
        idx.cindex_centroids = baseline::read_vec_bin<float>(dir + "/cindex_centroids.fp32.bin");
        idx.cquant_to_cindex = baseline::read_vec_bin<int32_t>(dir + "/cquant_to_cindex.i32.bin");
        idx.doclens = baseline::read_vec_bin<int32_t>(dir + "/doclens.i32.bin");
        idx.offsets = baseline::read_vec_bin<int64_t>(dir + "/offsets.i64.bin");
        idx.token_codes = baseline::read_vec_bin<int32_t>(dir + "/token_codes.i32.bin");
        idx.cluster_to_docs = baseline::read_varlists<int32_t>(dir + "/cluster_to_docs.bin");
        idx.validate(true, dir);
        return idx;
    }
};

inline std::size_t logical_index_bytes(const std::string& dir) {
    return baseline::directory_file_size(dir);
}

}  // namespace official_gem
