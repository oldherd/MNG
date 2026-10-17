// mvbundle_inspect.cpp — CLI smoke test for mv::MVBundle.
//
// Build:
//   g++ -std=c++17 -O3 -Wall mvbundle.cpp mvbundle_inspect.cpp -o mvbundle_inspect
//
// Usage:
//   ./mvbundle_inspect <bundle_dir> [--show N]
//
// Prints the same kind of summary as `python mvbundle.py <path> --show N`
// so the two implementations can be cross-checked.

#include "mvbundle.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

// IEEE 754 half precision (little-endian uint16_t) → float.
float fp16_to_fp32(std::uint16_t h) {
    const std::uint32_t s = (h >> 15) & 0x1u;
    const std::uint32_t e = (h >> 10) & 0x1fu;
    const std::uint32_t f =  h        & 0x3ffu;
    std::uint32_t bits;
    if (e == 0) {
        if (f == 0) {
            bits = s << 31;
        } else {
            // subnormal → normalize
            std::uint32_t mant = f;
            std::uint32_t exp = 1;
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

float l2_norm_first_token(const mv::DocView& v) {
    double sum = 0.0;
    const int d = v.dim;
    if (v.dtype == mv::DType::Float16) {
        const auto* p = v.as_fp16();
        for (int i = 0; i < d; ++i) {
            double x = fp16_to_fp32(p[i]);
            sum += x * x;
        }
    } else {
        const auto* p = v.as_fp32();
        for (int i = 0; i < d; ++i) {
            double x = p[i];
            sum += x * x;
        }
    }
    return static_cast<float>(std::sqrt(sum));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <bundle_dir> [--show N]\n";
        return 1;
    }
    int show = 3;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--show") == 0 && i + 1 < argc) {
            show = std::atoi(argv[++i]);
        }
    }

    try {
        auto b = mv::MVBundle::load(argv[1]);
        const auto& m = b.meta();
        std::cout << "MVBundle(" << (b.is_query() ? "query" : "doc")
                  << ", " << b.n_docs() << " rows, " << b.n_tokens() << " tokens, "
                  << "dim=" << b.dim() << ", dtype=" << mv::dtype_name(b.dtype())
                  << ", path=" << b.path().filename().string() << ")\n";
        std::cout << "  doc_maxlen=" << m.doc_maxlen
                  << " query_maxlen=" << m.query_maxlen
                  << " keep_padding=" << (m.keep_padding ? "true" : "false") << "\n";
        std::cout << "  encoder_ckpt=" << m.encoder_ckpt << "\n";

        const auto* dl = b.doclens();
        std::int32_t mn = dl[0], mx = dl[0];
        long long sumdl = 0;
        for (std::size_t i = 0; i < b.n_docs(); ++i) {
            mn = std::min(mn, dl[i]);
            mx = std::max(mx, dl[i]);
            sumdl += dl[i];
        }
        const double mean = static_cast<double>(sumdl) / static_cast<double>(b.n_docs());
        std::cout << "  doclens: min=" << mn << " max=" << mx
                  << " mean=" << mean << " sum=" << sumdl << "\n";

        const auto* did = b.docids();
        std::cout << "  docids[:5]=";
        for (std::size_t i = 0; i < std::min<std::size_t>(5, b.n_docs()); ++i) {
            std::cout << did[i] << " ";
        }
        std::cout << "\n  docids[-5:]=";
        const std::size_t n = b.n_docs();
        const std::size_t start = (n > 5) ? n - 5 : 0;
        for (std::size_t i = start; i < n; ++i) std::cout << did[i] << " ";
        std::cout << "\n";

        for (int r = 0; r < show && static_cast<std::size_t>(r) < b.n_docs(); ++r) {
            auto v = b.get(r);
            std::cout << "  row " << r << ": id=" << did[r]
                      << " len=" << v.tokens
                      << " dtype=" << mv::dtype_name(v.dtype)
                      << " first_tok_L2=" << l2_norm_first_token(v) << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
