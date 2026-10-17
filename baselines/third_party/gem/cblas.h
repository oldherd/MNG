#pragma once

#include <cstdlib>

typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_LAYOUT;

typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112
} CBLAS_TRANSPOSE;

#ifdef GEM_USE_EXTERNAL_CBLAS

extern "C" void cblas_sgemm(CBLAS_LAYOUT layout,
                            CBLAS_TRANSPOSE trans_a,
                            CBLAS_TRANSPOSE trans_b,
                            int m,
                            int n,
                            int k,
                            float alpha,
                            const float* a,
                            int lda,
                            const float* b,
                            int ldb,
                            float beta,
                            float* c,
                            int ldc);

#else

static inline float cblas_get_rowmajor(const float* a,
                                       int row,
                                       int col,
                                       int lda,
                                       CBLAS_TRANSPOSE trans) {
    if (trans == CblasNoTrans) return a[(std::size_t)row * (std::size_t)lda + (std::size_t)col];
    if (trans == CblasTrans) return a[(std::size_t)col * (std::size_t)lda + (std::size_t)row];
    std::abort();
}

static inline void cblas_sgemm(CBLAS_LAYOUT layout,
                               CBLAS_TRANSPOSE trans_a,
                               CBLAS_TRANSPOSE trans_b,
                               int m,
                               int n,
                               int k,
                               float alpha,
                               const float* a,
                               int lda,
                               const float* b,
                               int ldb,
                               float beta,
                               float* c,
                               int ldc) {
    if (layout != CblasRowMajor) std::abort();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += cblas_get_rowmajor(a, i, kk, lda, trans_a) *
                       cblas_get_rowmajor(b, kk, j, ldb, trans_b);
            }
            c[(std::size_t)i * (std::size_t)ldc + (std::size_t)j] =
                alpha * sum + beta * c[(std::size_t)i * (std::size_t)ldc + (std::size_t)j];
        }
    }
}

#endif
