#pragma once
#include "common/index_io.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace dessert {

struct DESSERTIndex {
    int dim;
    int L;              // number of hash tables
    int C;              // bits per hash
    int srp_sample_size;
    int n_filter;       // IVF pre-filter centroids
    size_t n_docs;
    size_t n_tokens;

    std::vector<float> filter_centroids;                // (n_filter × dim)
    std::vector<std::vector<int32_t>> ivf_lists;        // centroid → doc_ids
    std::vector<int16_t> srp_random_bits;               // (L*C × sample_size)
    std::vector<uint32_t> srp_hash_indices;             // (L*C × sample_size)
    std::vector<uint64_t> doc_table_offsets;            // (n_docs+1) offsets into doc_table_data
    std::vector<uint32_t> doc_n_tokens;                 // tokens per doc
    std::vector<uint16_t> doc_table_data;               // concatenated TinyTable payloads

    void save(const std::string& dir, double build_sec,
              double centroid_build_sec,
              const std::string& centroid_method,
              int kmeans_iter,
              int kmeans_sample,
              int seed,
              size_t assignment_batch_tokens = 0,
              bool write_tinytables = true) const {
        baseline::ensure_dir(dir);
        baseline::write_vec_bin(dir + "/filter_centroids.fp32.bin", filter_centroids);
        baseline::write_varlists(dir + "/ivf_lists.bin", ivf_lists);
        baseline::write_vec_bin(dir + "/srp_random_bits.i16.bin", srp_random_bits);
        baseline::write_vec_bin(dir + "/srp_hash_indices.u32.bin", srp_hash_indices);
        baseline::write_vec_bin(dir + "/doc_table_offsets.u64.bin", doc_table_offsets);
        baseline::write_vec_bin(dir + "/doc_n_tokens.u32.bin", doc_n_tokens);
        if (write_tinytables) {
            baseline::write_vec_bin(dir + "/doc_tinytables.u16.bin", doc_table_data);
        }

        baseline::write_meta(dir, {
            {"method", "dessert"},
            {"layout", "tinytables"},
            {"dim", std::to_string(dim)},
            {"L", std::to_string(L)},
            {"C", std::to_string(C)},
            {"srp_sample_size", std::to_string(srp_sample_size)},
            {"n_filter", std::to_string(n_filter)},
            {"n_docs", std::to_string(n_docs)},
            {"n_tokens", std::to_string(n_tokens)},
            {"build_time_sec", std::to_string(build_sec)},
            {"centroid_method", centroid_method},
            {"centroid_build_time_sec", std::to_string(centroid_build_sec)},
            {"kmeans_iter", std::to_string(kmeans_iter)},
            {"kmeans_sample", std::to_string(kmeans_sample)},
            {"sample_seed", std::to_string(seed)},
            {"source_token_dtype", "float16"},
            {"build_memory_policy", write_tinytables ? "materialized_fp32" : "streamed_fp16_source"},
            {"assignment_batch_tokens", std::to_string(assignment_batch_tokens)},
        });
    }

    static DESSERTIndex load(const std::string& dir) {
        auto meta = baseline::read_meta(dir);
        DESSERTIndex idx;
        idx.dim = std::stoi(meta.at("dim"));
        idx.L = std::stoi(meta.at("L"));
        idx.C = std::stoi(meta.at("C"));
        idx.srp_sample_size = std::stoi(meta.at("srp_sample_size"));
        idx.n_filter = std::stoi(meta.at("n_filter"));
        idx.n_docs = std::stoull(meta.at("n_docs"));
        idx.n_tokens = std::stoull(meta.at("n_tokens"));

        idx.filter_centroids = baseline::read_vec_bin<float>(dir + "/filter_centroids.fp32.bin");
        idx.ivf_lists = baseline::read_varlists<int32_t>(dir + "/ivf_lists.bin");
        idx.srp_random_bits = baseline::read_vec_bin<int16_t>(dir + "/srp_random_bits.i16.bin");
        idx.srp_hash_indices = baseline::read_vec_bin<uint32_t>(dir + "/srp_hash_indices.u32.bin");
        idx.doc_table_offsets = baseline::read_vec_bin<uint64_t>(dir + "/doc_table_offsets.u64.bin");
        idx.doc_n_tokens = baseline::read_vec_bin<uint32_t>(dir + "/doc_n_tokens.u32.bin");
        idx.doc_table_data = baseline::read_vec_bin<uint16_t>(dir + "/doc_tinytables.u16.bin");
        return idx;
    }
};

}  // namespace dessert
