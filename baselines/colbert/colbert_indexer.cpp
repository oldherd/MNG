// ColBERT vector-search indexer: FAISS IVFPQ over shared document token vectors.

#include "mvbundle.hpp"
#include "common/bench_utils.hpp"
#include "common/fp16_utils.hpp"
#include "common/timer.hpp"
#include "colbert_index.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

void fp16_token_to_fp32(const uint16_t* src, int dim, float* dst) {
    for (int d = 0; d < dim; ++d) {
        dst[d] = baseline::fp16_to_fp32(src[d]);
    }
}

std::vector<float> sample_tokens(const mv::MVBundle& doc_bundle,
                                 std::size_t sample_n,
                                 int seed) {
    const int dim = doc_bundle.dim();
    const std::size_t n_tokens = doc_bundle.n_tokens();
    sample_n = std::min(sample_n, n_tokens);

    std::vector<float> sample(sample_n * (std::size_t)dim);
    const uint16_t* all = doc_bundle.embeddings_fp16();

    std::mt19937_64 rng((uint64_t)seed);
    std::uniform_int_distribution<std::size_t> dist(0, n_tokens - 1);
    for (std::size_t i = 0; i < sample_n; ++i) {
        std::size_t token_id = dist(rng);
        fp16_token_to_fp32(all + token_id * (std::size_t)dim, dim,
                           sample.data() + i * (std::size_t)dim);
    }
    return sample;
}

faiss::MetricType parse_metric(const std::string& metric) {
    if (metric == "ip" || metric == "inner_product") return faiss::METRIC_INNER_PRODUCT;
    if (metric == "l2") return faiss::METRIC_L2;
    throw std::runtime_error("unsupported metric=" + metric + " (use ip or l2)");
}

}  // namespace

int main(int argc, char** argv) {
    auto args = baseline::parse_args(argc, argv);
    if (args.doc_bundle.empty() || args.index_dir.empty()) {
        std::cerr << "Usage: indexer_colbert --doc <bundle> --index_dir <dir> "
                  << "[--param nlist=0] [--param pq_m=16] [--param nbits=8] "
                  << "[--param train_samples=1000000] [--param add_batch=100000] "
                  << "[--param seed=42] [--param metric=ip]\n";
        return 1;
    }

    auto doc_bundle = mv::MVBundle::load(args.doc_bundle);
    const int dim = doc_bundle.dim();
    const std::size_t n_docs = doc_bundle.n_docs();
    const std::size_t n_tokens = doc_bundle.n_tokens();

    int nlist = baseline::get_extra_int(args, "nlist", 0);
    int pq_m = baseline::get_extra_int(args, "pq_m", 16);
    int nbits = baseline::get_extra_int(args, "nbits", 8);
    int train_samples = baseline::get_extra_int(args, "train_samples", 1000000);
    int add_batch = baseline::get_extra_int(args, "add_batch", 100000);
    int seed = baseline::get_extra_int(args, "seed", 42);
    std::string metric = "ip";
    for (const auto& kv : args.extra) {
        if (kv.first == "metric") metric = kv.second;
    }

    if (nlist == 0) {
        nlist = (int)std::floor(2.0 * std::sqrt((double)n_tokens));
        nlist = std::max(256, std::min(65536, nlist));
    }
    if (dim % pq_m != 0) {
        std::cerr << "[colbert-indexer] dim=" << dim << " must be divisible by pq_m=" << pq_m << "\n";
        return 1;
    }

    std::cerr << "[colbert-indexer] docs=" << n_docs << " tokens=" << n_tokens
              << " dim=" << dim << " nlist=" << nlist << " pq_m=" << pq_m
              << " nbits=" << nbits << " metric=" << metric << "\n";

    baseline::Timer build_timer;
    build_timer.start();

    std::cerr << "[colbert-indexer] sampling " << std::min<std::size_t>(train_samples, n_tokens)
              << " tokens for IVFPQ training (seed=" << seed << ")...\n";
    std::vector<float> train = sample_tokens(doc_bundle, (std::size_t)train_samples, seed);

    faiss::MetricType metric_type = parse_metric(metric);
    std::unique_ptr<faiss::Index> quantizer;
    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        quantizer = std::make_unique<faiss::IndexFlatIP>(dim);
    } else {
        quantizer = std::make_unique<faiss::IndexFlatL2>(dim);
    }

    faiss::IndexIVFPQ index(quantizer.get(), dim, nlist, pq_m, nbits, metric_type);
    index.verbose = true;
    index.own_fields = false;

    std::cerr << "[colbert-indexer] training FAISS IVFPQ...\n";
    index.train((faiss::idx_t)(train.size() / (std::size_t)dim), train.data());
    train.clear();
    train.shrink_to_fit();

    std::vector<float> batch((std::size_t)add_batch * dim);
    std::vector<faiss::idx_t> ids(add_batch);
    const uint16_t* all = doc_bundle.embeddings_fp16();
    for (std::size_t start = 0; start < n_tokens; start += (std::size_t)add_batch) {
        std::size_t count = std::min<std::size_t>((std::size_t)add_batch, n_tokens - start);
        for (std::size_t i = 0; i < count; ++i) {
            fp16_token_to_fp32(all + (start + i) * (std::size_t)dim, dim,
                               batch.data() + i * (std::size_t)dim);
            ids[i] = (faiss::idx_t)(start + i);
        }
        index.add_with_ids((faiss::idx_t)count, batch.data(), ids.data());
        if (start == 0 || ((start / (std::size_t)add_batch) % 20 == 0)) {
            std::cerr << "[colbert-indexer] added " << (start + count) << "/" << n_tokens << " tokens\n";
        }
    }

    build_timer.stop();
    double build_sec = build_timer.elapsed_sec();

    baseline::ensure_dir(args.index_dir);
    colbert::write_faiss_index(args.index_dir, index);
    baseline::write_meta(args.index_dir, {
        {"method", "colbert"},
        {"layout", colbert::kLayout},
        {"ann_backend", "faiss"},
        {"faiss_index", "IndexIVFPQ"},
        {"metric", metric},
        {"dim", std::to_string(dim)},
        {"nlist", std::to_string(nlist)},
        {"pq_m", std::to_string(pq_m)},
        {"nbits", std::to_string(nbits)},
        {"train_samples", std::to_string(std::min<std::size_t>((std::size_t)train_samples, n_tokens))},
        {"sample_seed", std::to_string(seed)},
        {"n_docs", std::to_string(n_docs)},
        {"n_tokens", std::to_string(n_tokens)},
        {"build_time_sec", std::to_string(build_sec)},
        {"input_doc_bundle", args.doc_bundle},
    });

    std::cerr << "[colbert-indexer] done in " << build_sec << "s; logical bytes="
              << colbert::logical_index_bytes(args.index_dir) << "\n";
    return 0;
}
