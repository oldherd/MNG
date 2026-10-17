#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace baseline {

// Abstract single-vector ANN index interface.
// Allows swapping out hnswlib for other implementations later.
class ANNIndex {
public:
    virtual ~ANNIndex() = default;

    // Build index from n vectors of dimension dim (fp32, row-major).
    virtual void build(const float* data, size_t n, int dim) = 0;

    // Search for k nearest neighbors of a single query vector.
    // labels: output array of size k (neighbor IDs, 0-based)
    // distances: output array of size k (distances/similarities)
    virtual void search(const float* query, size_t k,
                       int64_t* labels, float* distances) const = 0;

    // Set search-time ef parameter (ef_search for HNSW).
    virtual void set_ef_search(int ef) = 0;

    // Approximate memory usage in bytes.
    virtual size_t memory_bytes() const = 0;

    // Number of indexed vectors.
    virtual size_t size() const = 0;
};

// Factory function — implemented in ann_hnswlib.hpp
std::unique_ptr<ANNIndex> make_hnsw_ip(int dim, int M, int ef_construction);

}  // namespace baseline
