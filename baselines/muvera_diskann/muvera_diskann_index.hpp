#pragma once

#include "common/index_io.hpp"
#include "utils/graph.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace muvera_diskann {

struct FDERepState {
    std::vector<std::vector<float>> simhash_planes;
    std::vector<float> projection;
};

template <typename T>
std::vector<T> read_prefix(const std::string& path, size_t count) {
    const size_t bytes = count * sizeof(T);
    if (baseline::file_size(path) < bytes) {
        throw std::runtime_error("muvera_diskann: file is shorter than expected: " + path);
    }
    std::vector<T> out(count);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("muvera_diskann: cannot open " + path);
    input.read(reinterpret_cast<char*>(out.data()), bytes);
    if (!input) throw std::runtime_error("muvera_diskann: failed reading " + path);
    return out;
}

struct Index {
    int dim = 0;
    int B = 0;
    int R = 0;
    int d_proj = 0;
    int d_fde = 0;
    int ksim = 0;
    int pq_group_size = 0;
    int pq_ksub = 0;
    int pq_m = 0;
    int graph_R = 0;
    int graph_L = 0;
    int graph_num_passes = 0;
    float graph_alpha = 0.0f;
    int seed = 0;
    size_t n_docs = 0;
    double graph_build_sec = 0.0;
    std::string build_scope;
    std::vector<FDERepState> rep_states;
    std::vector<float> pq_centroids;
    std::vector<uint8_t> pq_codes;
    std::unique_ptr<parlayANN::Graph<uint32_t>> graph;

    bool has_pq_256_8() const {
        return pq_group_size == 8 && pq_ksub == 256 && pq_m > 0 &&
               d_fde == pq_m * pq_group_size &&
               pq_centroids.size() ==
                   static_cast<size_t>(pq_m) * pq_ksub * pq_group_size &&
               pq_codes.size() == n_docs * static_cast<size_t>(pq_m);
    }

    static Index load(const std::string& dir) {
        auto meta = baseline::read_meta(dir);
        if (meta.at("method") != "muvera_diskann") {
            throw std::runtime_error("muvera_diskann: index method mismatch in " + dir);
        }

        Index idx;
        idx.dim = std::stoi(meta.at("dim"));
        idx.B = std::stoi(meta.at("B"));
        idx.R = std::stoi(meta.at("R"));
        idx.d_proj = std::stoi(meta.at("d_proj"));
        idx.d_fde = std::stoi(meta.at("d_fde"));
        idx.ksim = std::stoi(meta.at("ksim"));
        idx.pq_group_size = std::stoi(meta.at("pq_group_size"));
        idx.pq_ksub = std::stoi(meta.at("pq_ksub"));
        idx.pq_m = std::stoi(meta.at("pq_m"));
        idx.graph_R = std::stoi(meta.at("diskann_R"));
        idx.graph_L = std::stoi(meta.at("diskann_L"));
        idx.graph_alpha = std::stof(meta.at("diskann_alpha"));
        idx.graph_num_passes = std::stoi(meta.at("diskann_num_passes"));
        idx.seed = std::stoi(meta.at("seed"));
        idx.n_docs = std::stoull(meta.at("n_docs"));
        idx.graph_build_sec = std::stod(meta.at("graph_build_time_sec"));
        idx.build_scope = meta.at("index_build_scope");

        idx.pq_centroids = baseline::read_vec_bin<float>(dir + "/pq_centroids.fp32.bin");
        idx.pq_codes = read_prefix<uint8_t>(
            dir + "/pq_codes.u8.bin", idx.n_docs * static_cast<size_t>(idx.pq_m));
        if (!idx.has_pq_256_8()) {
            throw std::runtime_error("muvera_diskann: PQ-256-8 state missing or malformed");
        }

        auto planes = baseline::read_vec_bin<float>(dir + "/simhash_planes.fp32.bin");
        auto projections = baseline::read_vec_bin<float>(dir + "/projections.fp32.bin");
        const size_t expected_planes = static_cast<size_t>(idx.R) * idx.ksim * idx.dim;
        const size_t expected_projections =
            static_cast<size_t>(idx.R) * idx.dim * idx.d_proj;
        if (planes.size() != expected_planes || projections.size() != expected_projections) {
            throw std::runtime_error("muvera_diskann: FDE projection state size mismatch");
        }

        idx.rep_states.resize(idx.R);
        size_t plane_offset = 0;
        size_t projection_offset = 0;
        for (int r = 0; r < idx.R; ++r) {
            idx.rep_states[r].simhash_planes.resize(idx.ksim);
            for (int h = 0; h < idx.ksim; ++h) {
                idx.rep_states[r].simhash_planes[h].assign(
                    planes.begin() + plane_offset,
                    planes.begin() + plane_offset + idx.dim);
                plane_offset += idx.dim;
            }
            idx.rep_states[r].projection.assign(
                projections.begin() + projection_offset,
                projections.begin() + projection_offset +
                    static_cast<size_t>(idx.dim) * idx.d_proj);
            projection_offset += static_cast<size_t>(idx.dim) * idx.d_proj;
        }

        std::vector<char> graph_path(dir.begin(), dir.end());
        const std::string suffix = "/vamana.graph";
        graph_path.insert(graph_path.end(), suffix.begin(), suffix.end());
        graph_path.push_back('\0');
        idx.graph = std::make_unique<parlayANN::Graph<uint32_t>>(graph_path.data());
        if (idx.graph->size() != idx.n_docs || idx.graph->max_degree() != idx.graph_R) {
            throw std::runtime_error("muvera_diskann: graph shape mismatch");
        }
        return idx;
    }
};

}  // namespace muvera_diskann
