#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

namespace dessert {

struct SparseRandomProjection {
    int dim = 0;
    int hashes_per_table = 0;
    int num_tables = 0;
    int sample_size = 0;
    std::vector<int16_t> random_bits;
    std::vector<uint32_t> hash_indices;

    SparseRandomProjection() = default;

    SparseRandomProjection(int dim_, int hashes_per_table_, int num_tables_, uint32_t seed)
        : dim(dim_),
          hashes_per_table(hashes_per_table_),
          num_tables(num_tables_),
          sample_size((int)std::ceil(dim_ * 0.3)) {
        if (dim <= 0 || hashes_per_table <= 0 || num_tables <= 0 || hashes_per_table >= 32) {
            throw std::invalid_argument("invalid SRP parameters");
        }

        const int total_srps = hashes_per_table * num_tables;
        random_bits.resize((size_t)total_srps * sample_size);
        hash_indices.resize((size_t)total_srps * sample_size);

        std::vector<uint32_t> dims(dim);
        std::iota(dims.begin(), dims.end(), 0);

        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        for (int i = 0; i < total_srps; ++i) {
            std::shuffle(dims.begin(), dims.end(), gen);
            auto* idx = hash_indices.data() + (size_t)i * sample_size;
            auto* bits = random_bits.data() + (size_t)i * sample_size;
            for (int j = 0; j < sample_size; ++j) {
                idx[j] = dims[j];
                bits[j] = (int16_t)(sign_dist(gen) * 2 - 1);
            }
            std::sort(idx, idx + sample_size);
        }
    }

    void hash_one(const float* values, uint32_t* output) const {
        std::memset(output, 0, (size_t)num_tables * sizeof(uint32_t));
        for (int table = 0; table < num_tables; ++table) {
            for (int srp = 0; srp < hashes_per_table; ++srp) {
                double s = 0.0;
                const size_t base = ((size_t)table * hashes_per_table + srp) * sample_size;
                for (int part = 0; part < sample_size; ++part) {
                    s += (double)random_bits[base + part] * values[hash_indices[base + part]];
                }
                if (s > 0.0) output[table] += (1u << srp);
            }
        }
    }
};

inline std::vector<uint32_t> hash_vectors(const SparseRandomProjection& srp,
                                          const float* vectors,
                                          int n_vectors,
                                          int dim) {
    std::vector<uint32_t> hashes((size_t)n_vectors * srp.num_tables);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_vectors; ++i) {
        srp.hash_one(vectors + (size_t)i * dim, hashes.data() + (size_t)i * srp.num_tables);
    }
    return hashes;
}

inline std::vector<uint32_t> hash_vectors_serial(const SparseRandomProjection& srp,
                                                 const float* vectors,
                                                 int n_vectors,
                                                 int dim) {
    std::vector<uint32_t> hashes((size_t)n_vectors * srp.num_tables);
    for (int i = 0; i < n_vectors; ++i) {
        srp.hash_one(vectors + (size_t)i * dim, hashes.data() + (size_t)i * srp.num_tables);
    }
    return hashes;
}

inline std::vector<uint16_t> build_tiny_table(const std::vector<uint32_t>& hashes,
                                              uint32_t num_elements,
                                              uint32_t num_tables,
                                              uint32_t range) {
    if (num_elements == 0) return {};
    if (num_elements > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("DESSERT tiny table requires uint16_t-sized documents");
    }

    const uint64_t table_start = (uint64_t)num_tables * (range + 1);
    std::vector<uint16_t> index(table_start + (uint64_t)num_tables * num_elements);

    for (uint32_t table = 0; table < num_tables; ++table) {
        std::vector<std::vector<uint16_t>> buckets(range);
        for (uint32_t vec_id = 0; vec_id < num_elements; ++vec_id) {
            uint32_t h = hashes[(uint64_t)vec_id * num_tables + table];
            buckets.at(h).push_back((uint16_t)vec_id);
        }

        const uint64_t offset_start = (uint64_t)table * (range + 1);
        index[offset_start] = 0;
        for (uint32_t bucket = 1; bucket <= range; ++bucket) {
            uint32_t next = (uint32_t)index[offset_start + bucket - 1] +
                            (uint32_t)buckets[bucket - 1].size();
            index[offset_start + bucket] = (uint16_t)next;
        }

        uint64_t current = table_start + (uint64_t)table * num_elements;
        for (uint32_t bucket = 0; bucket < range; ++bucket) {
            for (uint16_t item : buckets[bucket]) {
                index[current++] = item;
            }
        }
    }

    return index;
}

inline float score_tiny_table(const uint16_t* table_data,
                              uint32_t num_doc_tokens,
                              const std::vector<uint32_t>& query_hashes,
                              uint32_t num_query_tokens,
                              uint32_t num_tables,
                              uint32_t range,
                              std::vector<uint32_t>& count_buffer,
                              const std::vector<float>& collision_count_to_sim) {
    if (num_doc_tokens == 0 || num_query_tokens == 0) return 0.0f;
    if (count_buffer.size() < num_doc_tokens) count_buffer.resize(num_doc_tokens);

    const uint64_t table_start = (uint64_t)num_tables * (range + 1);
    float sum = 0.0f;
    for (uint32_t qv = 0; qv < num_query_tokens; ++qv) {
        std::fill(count_buffer.begin(), count_buffer.begin() + num_doc_tokens, 0);
        for (uint32_t table = 0; table < num_tables; ++table) {
            uint32_t h = query_hashes[(uint64_t)qv * num_tables + table];
            uint64_t offset_start = (uint64_t)table * (range + 1);
            uint16_t start = table_data[offset_start + h];
            uint16_t end = table_data[offset_start + h + 1];
            uint64_t table_offset = table_start + (uint64_t)table * num_doc_tokens;
            for (uint32_t off = start; off < end; ++off) {
                count_buffer[table_data[table_offset + off]]++;
            }
        }

        uint32_t max_count = 0;
        for (uint32_t i = 0; i < num_doc_tokens; ++i) {
            if (count_buffer[i] > max_count) max_count = count_buffer[i];
        }
        sum += collision_count_to_sim[max_count];
    }

    return sum / (float)num_query_tokens;
}

inline std::vector<float> make_collision_lookup(uint32_t num_tables, uint32_t hashes_per_table) {
    std::vector<float> lookup(num_tables + 1, 0.0f);
    for (uint32_t count = 1; count <= num_tables; ++count) {
        float p = (float)count / (float)num_tables;
        lookup[count] = std::exp(std::log(p) / (float)hashes_per_table);
    }
    return lookup;
}

}  // namespace dessert
