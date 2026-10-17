#pragma once
#include "ann_interface.hpp"
#include "baselines/hnswlib/hnswlib.h"
#include <stdexcept>

namespace baseline {

class HnswlibIndex : public ANNIndex {
public:
    HnswlibIndex(int dim, int M, int ef_construction)
        : dim_(dim), M_(M), ef_construction_(ef_construction) {}

    void build(const float* data, size_t n, int dim) override {
        if (dim != dim_) throw std::runtime_error("dim mismatch");
        space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), n, M_, ef_construction_);
        if (n > 0) index_->addPoint(data, 0);
        #pragma omp parallel for schedule(dynamic, 1000)
        for (size_t i = 1; i < n; ++i) {
            index_->addPoint(data + i * dim, i);
        }
        n_ = n;
    }

    void search(const float* query, size_t k,
               int64_t* labels, float* distances) const override {
        // Initialize all slots to sentinel values
        for (size_t j = 0; j < k; ++j) {
            labels[j] = -1;
            distances[j] = -1e30f;
        }
        auto result = index_->searchKnn(query, k);
        // result is a max-heap of (distance, label) — for IP, distance = 1-IP
        size_t n_results = result.size();
        size_t i = n_results;
        while (!result.empty()) {
            --i;
            labels[i]    = result.top().second;
            distances[i] = 1.0f - result.top().first;
            result.pop();
        }
    }

    void set_ef_search(int ef) override {
        if (index_) index_->setEf(ef);
    }

    size_t memory_bytes() const override {
        if (!index_) return 0;
        // Approximate: data + graph links
        size_t data_bytes = n_ * dim_ * sizeof(float);
        size_t graph_bytes = n_ * M_ * 2 * sizeof(int);
        return data_bytes + graph_bytes;
    }

    size_t size() const override { return n_; }

    void save(const std::string& path) const {
        if (index_) index_->saveIndex(path);
    }

    static std::unique_ptr<HnswlibIndex> load_from_file(const std::string& path, int dim) {
        auto idx = std::make_unique<HnswlibIndex>(dim, 0, 0);
        idx->space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
        idx->index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            idx->space_.get(), path);
        idx->dim_ = dim;
        idx->n_ = idx->index_->cur_element_count;
        idx->M_ = idx->index_->M_;
        idx->ef_construction_ = idx->index_->ef_construction_;
        return idx;
    }

private:
    int dim_, M_, ef_construction_;
    size_t n_ = 0;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

// Factory implementation
inline std::unique_ptr<ANNIndex> make_hnsw_ip(int dim, int M, int ef_construction) {
    return std::make_unique<HnswlibIndex>(dim, M, ef_construction);
}

// Load from file
inline std::unique_ptr<ANNIndex> load_hnsw_ip(const std::string& path, int dim) {
    return HnswlibIndex::load_from_file(path, dim);
}

}  // namespace baseline
