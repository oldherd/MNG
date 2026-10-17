#pragma once
#include <vector>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <numeric>

namespace baseline {

// K-means clustering (Lloyd's algorithm) with OpenMP parallelism.
class KMeans {
public:
    static std::vector<float> fit(const float* data, size_t n, int dim, int k,
                                  int max_iter = 25, int seed = 42) {
        std::vector<float> centroids(k * dim);
        std::vector<int32_t> assignments(n, -1);

        init_kmeanspp(data, n, dim, k, centroids.data(), seed);

        for (int iter = 0; iter < max_iter; ++iter) {
            bool changed = false;
            #pragma omp parallel for schedule(static) reduction(||:changed)
            for (size_t i = 0; i < n; ++i) {
                int best = nearest_centroid(data + i * dim, centroids.data(), k, dim);
                if (best != assignments[i]) {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if (!changed) break;

            std::vector<double> sums((size_t)k * dim, 0.0);
            std::vector<int> counts(k, 0);

            #pragma omp parallel
            {
                std::vector<double> local_sums((size_t)k * dim, 0.0);
                std::vector<int> local_counts(k, 0);
                #pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    int c = assignments[i];
                    local_counts[c]++;
                    for (int d = 0; d < dim; ++d)
                        local_sums[(size_t)c * dim + d] += data[i * dim + d];
                }
                #pragma omp critical
                {
                    for (int c = 0; c < k; ++c) {
                        counts[c] += local_counts[c];
                        for (int d = 0; d < dim; ++d)
                            sums[(size_t)c * dim + d] += local_sums[(size_t)c * dim + d];
                    }
                }
            }

            for (int c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    for (int d = 0; d < dim; ++d)
                        centroids[(size_t)c * dim + d] = (float)(sums[(size_t)c * dim + d] / counts[c]);
                }
            }
        }

        return centroids;
    }

    static std::vector<int32_t> assign(const float* data, size_t n, int dim,
                                       const float* centroids, int k) {
        std::vector<int32_t> assignments(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            assignments[i] = nearest_centroid(data + i * dim, centroids, k, dim);
        }
        return assignments;
    }

    static int nearest_centroid(const float* vec, const float* centroids, int k, int dim) {
        int best = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < k; ++c) {
            float score = 0.0f;
            for (int d = 0; d < dim; ++d)
                score += vec[d] * centroids[(size_t)c * dim + d];
            if (score > best_score) {
                best_score = score;
                best = c;
            }
        }
        return best;
    }

    static std::vector<int> topk_centroids(const float* vec, const float* centroids,
                                           int k, int dim, int nprobe) {
        std::vector<std::pair<float, int>> scores(k);
        for (int c = 0; c < k; ++c) {
            float score = 0.0f;
            for (int d = 0; d < dim; ++d)
                score += vec[d] * centroids[(size_t)c * dim + d];
            scores[c] = {score, c};
        }
        std::partial_sort(scores.begin(), scores.begin() + std::min(nprobe, k),
                         scores.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<int> result(std::min(nprobe, k));
        for (int i = 0; i < (int)result.size(); ++i)
            result[i] = scores[i].second;
        return result;
    }

private:
    static void init_kmeanspp(const float* data, size_t n, int dim, int k,
                              float* centroids, int seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> uni(0, n - 1);

        size_t first = uni(rng);
        std::memcpy(centroids, data + first * dim, dim * sizeof(float));

        std::vector<float> min_dist(n, std::numeric_limits<float>::infinity());

        for (int c = 1; c < k; ++c) {
            const float* prev = centroids + (size_t)(c-1) * dim;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                float dist = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = data[i * dim + d] - prev[d];
                    dist += diff * diff;
                }
                if (dist < min_dist[i]) min_dist[i] = dist;
            }

            float total = 0.0f;
            for (size_t i = 0; i < n; ++i) total += min_dist[i];
            std::uniform_real_distribution<float> unif(0.0f, total);
            float target = unif(rng);
            float cumul = 0.0f;
            size_t chosen = 0;
            for (size_t i = 0; i < n; ++i) {
                cumul += min_dist[i];
                if (cumul >= target) { chosen = i; break; }
            }
            std::memcpy(centroids + (size_t)c * dim, data + chosen * dim, dim * sizeof(float));
        }
    }
};

}  // namespace baseline
