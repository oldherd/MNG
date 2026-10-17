// gt_inspect.cpp — CLI smoke test for mv::MVGroundTruth.
//
// Build:
//   make
//
// Usage:
//   ./gt_inspect <gt_dir> [--show N]
//
// Cross-validates with `python mvbundle.py --gt <gt_dir>`: same fields and
// numeric values should appear in both outputs.

#include "mvbundle.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

// Same fp16→fp32 conversion as mvbundle_inspect.cpp (so single-source not worth
// it for these two short CLIs).
float fp16_to_fp32(std::uint16_t h) {
    const std::uint32_t s = (h >> 15) & 0x1u;
    const std::uint32_t e = (h >> 10) & 0x1fu;
    const std::uint32_t f =  h        & 0x3ffu;
    std::uint32_t bits;
    if (e == 0) {
        if (f == 0) {
            bits = s << 31;
        } else {
            std::uint32_t mant = f, exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            bits = (s << 31) | ((127u - 15u + exp) << 23) | (mant << 13);
        }
    } else if (e == 31) {
        bits = (s << 31) | (0xffu << 23) | (f << 13);
    } else {
        bits = (s << 31) | ((e - 15u + 127u) << 23) | (f << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

float score_at(const mv::MVGroundTruth& gt, std::size_t q, std::int32_t doc_id) {
    const void* row = gt.scores_row(q);
    if (gt.score_dtype() == mv::DType::Float32) {
        return static_cast<const float*>(row)[doc_id];
    } else {
        return fp16_to_fp32(static_cast<const std::uint16_t*>(row)[doc_id]);
    }
}

float topk_score(const mv::MVGroundTruth& gt, std::size_t q, std::size_t i) {
    const void* row = gt.topk_scores_row(q);
    if (gt.score_dtype() == mv::DType::Float32) {
        return static_cast<const float*>(row)[i];
    } else {
        return fp16_to_fp32(static_cast<const std::uint16_t*>(row)[i]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <gt_dir> [--show N]\n";
        return 1;
    }
    int show = 3;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--show") == 0 && i + 1 < argc) {
            show = std::atoi(argv[++i]);
        }
    }

    try {
        auto gt = mv::MVGroundTruth::load(argv[1]);
        const auto& m = gt.meta();
        std::cout << "MVGroundTruth(" << gt.n_queries() << " q × "
                  << gt.n_docs() << " d, top_k=" << gt.top_k()
                  << ", dtype=" << mv::dtype_name(gt.score_dtype())
                  << ", path=" << gt.path().filename().string() << ")\n";
        std::cout << "  dim=" << m.dim << " q_maxlen=" << m.q_maxlen
                  << " sort_order=" << m.sort_order
                  << " elapsed_sec=" << m.elapsed_sec << "\n";

        const std::size_t n = gt.n_queries();
        const int show_n = std::min<int>(show, static_cast<int>(n));
        for (int qi = 0; qi < show_n; ++qi) {
            const auto* idx = gt.topk_indices_row(qi);
            std::cout << "  q[" << qi << "]: top-3 = [";
            for (int i = 0; i < std::min(3, gt.top_k()); ++i) {
                if (i) std::cout << ", ";
                std::cout << "(" << idx[i] << ", " << topk_score(gt, qi, i) << ")";
            }
            std::cout << "]\n";
        }

        // consistency: scores[q, topk_indices[q,0]] should equal topk_scores[q,0]
        const std::size_t q0 = 0;
        const std::int32_t top_doc = gt.topk_indices_row(q0)[0];
        const float s_full = score_at(gt, q0, top_doc);
        const float s_topk = topk_score(gt, q0, 0);
        std::cout << "  consistency check (q=0, rank 0): full[q, gt_id] = "
                  << s_full << "  vs topk_scores = " << s_topk
                  << "  diff = " << std::abs(s_full - s_topk) << "\n";

        // recall sanity: against itself = 1.0
        std::vector<std::int32_t> self_topk(
            gt.topk_indices_row(q0),
            gt.topk_indices_row(q0) + std::min(10, gt.top_k()));
        float r = gt.recall_at_k(self_topk.data(), q0, self_topk.size());
        std::cout << "  recall_at_k(gt[0,:10], q=0, k=10) = " << r
                  << " (expect 1.0)\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
