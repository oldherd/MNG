#include "common/bench_utils.hpp"
#include "common/index_io.hpp"
#include "common/timer.hpp"
#include "muvera_diskann_fde.hpp"
#include "utils/stats.h"
#include "vamana/index.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void link_artifact(const fs::path& source_dir, const fs::path& output_dir,
                   const std::string& name) {
    const fs::path source = fs::canonical(source_dir / name);
    const fs::path destination = output_dir / name;
    if (fs::exists(destination) || fs::is_symlink(destination)) fs::remove(destination);
    fs::create_symlink(source, destination);
}

}  // namespace

int main(int argc, char** argv) {
    auto args = baseline::parse_args(argc, argv);
    const std::string source_index =
        baseline::get_extra_string(args, "source_index", "");
    if (args.index_dir.empty() || source_index.empty()) {
        std::cerr << "Usage: indexer_muvera_diskann --index_dir <dir> "
                  << "--param source_index=<cached-fde-index> "
                  << "[--param diskann_R=200] [--param diskann_L=600] "
                  << "[--param diskann_alpha=1.0] [--param diskann_num_passes=1] "
                  << "[--param index_threads=96] [--param max_docs=0]\n";
        return 1;
    }

    const int graph_R = baseline::get_extra_int(args, "diskann_R", 200);
    const int graph_L = baseline::get_extra_int(args, "diskann_L", 600);
    const float graph_alpha = baseline::get_extra_float(args, "diskann_alpha", 1.0f);
    const int graph_num_passes =
        baseline::get_extra_int(args, "diskann_num_passes", 1);
    const int index_threads = baseline::get_extra_int(args, "index_threads", 96);
    const int max_docs = baseline::get_extra_int(args, "max_docs", 0);
    if (graph_R <= 0 || graph_L < graph_R || graph_alpha <= 0.0f ||
        graph_num_passes <= 0 || index_threads <= 0 || index_threads > 96 ||
        max_docs < 0) {
        std::cerr << "[muvera-diskann-indexer] ERROR: require R>0, L>=R, alpha>0, "
                  << "num_passes>0, index_threads in [1,96], and max_docs>=0.\n";
        return 1;
    }

    const std::string threads = std::to_string(index_threads);
    ::setenv("PARLAY_NUM_THREADS", threads.c_str(), 1);
    ::setenv("OMP_NUM_THREADS", threads.c_str(), 1);
    ::setenv("OMP_DYNAMIC", "FALSE", 1);

    auto source_meta = baseline::read_meta(source_index);
    const std::string source_method = source_meta.at("method");
    if (source_method != "muvera_fde_hnsw" && source_method != "muvera_fde_cache") {
        std::cerr << "[muvera-diskann-indexer] ERROR: source must be an audited "
                  << "muvera_fde_hnsw or muvera_fde_cache artifact.\n";
        return 1;
    }

    const int dim = std::stoi(source_meta.at("dim"));
    const int B = std::stoi(source_meta.at("B"));
    const int repetitions = std::stoi(source_meta.at("R"));
    const int d_proj = std::stoi(source_meta.at("d_proj"));
    const int d_fde = std::stoi(source_meta.at("d_fde"));
    const int ksim = std::stoi(source_meta.at("ksim"));
    const int pq_group_size = std::stoi(source_meta.at("pq_group_size"));
    const int pq_ksub = std::stoi(source_meta.at("pq_ksub"));
    const int pq_m = std::stoi(source_meta.at("pq_m"));
    const int seed = std::stoi(source_meta.at("seed"));
    const size_t source_n_docs = std::stoull(source_meta.at("n_docs"));
    const size_t n_docs = max_docs > 0
        ? std::min(source_n_docs, static_cast<size_t>(max_docs))
        : source_n_docs;
    if (d_fde != repetitions * B * d_proj || d_fde != pq_m * pq_group_size ||
        pq_group_size != 8 || pq_ksub != 256 || n_docs == 0) {
        std::cerr << "[muvera-diskann-indexer] ERROR: source FDE/PQ metadata mismatch.\n";
        return 1;
    }

    std::cerr << "[muvera-diskann-indexer] Parlay workers: "
              << parlay::num_workers() << "\n";
    if (parlay::num_workers() != static_cast<size_t>(index_threads)) {
        std::cerr << "[muvera-diskann-indexer] ERROR: Parlay worker count mismatch.\n";
        return 1;
    }
    std::cerr << "[muvera-diskann-indexer] Loading " << n_docs << " x " << d_fde
              << " cached FP32 FDEs from " << source_index << "\n";
    auto fdes = muvera_diskann::read_prefix<float>(
        source_index + "/fdes.fp32.bin", n_docs * static_cast<size_t>(d_fde));
    muvera_diskann::RawFDEPointRange points(fdes.data(), n_docs, d_fde);

    parlayANN::BuildParams build_params(
        graph_R, graph_L, static_cast<double>(graph_alpha), graph_num_passes, false);
    parlayANN::Graph<uint32_t> graph(graph_R, n_docs);
    parlayANN::stats<uint32_t> build_stats(n_docs);
    parlayANN::knn_index<muvera_diskann::RawFDEPointRange,
                        muvera_diskann::RawFDEPointRange,
                        uint32_t>
        builder(build_params);

    baseline::Timer graph_timer;
    graph_timer.start();
    builder.build_index(graph, points, points, build_stats);
    graph_timer.stop();
    const double graph_build_sec = graph_timer.elapsed_sec();

    const fs::path output_dir(args.index_dir);
    fs::create_directories(output_dir);
    std::string graph_path = (output_dir / "vamana.graph").string();
    std::vector<char> graph_path_buffer(graph_path.begin(), graph_path.end());
    graph_path_buffer.push_back('\0');
    graph.save(graph_path_buffer.data());

    for (const std::string& artifact : {
             "fdes.fp32.bin", "pq_centroids.fp32.bin", "pq_codes.u8.bin",
             "simhash_planes.fp32.bin", "projections.fp32.bin"}) {
        link_artifact(source_index, output_dir, artifact);
    }

    const bool full_dataset = n_docs == source_n_docs;
    baseline::write_meta(args.index_dir, {
        {"method", "muvera_diskann"},
        {"formal_baseline", "false"},
        {"formal_candidate", full_dataset ? "true" : "false"},
        {"validation_status", "pending_equivalence_validation"},
        {"ann_backend", "parlayann_vamana_diskann"},
        {"parlayann_commit", "a48c6a34e5f959190992aa531cc18454cfcc3dd3"},
        {"metric", "maximum_inner_product"},
        {"index_build_scope", "diskann_graph_from_cached_fde"},
        {"layout", "fde_pq256x8_vamana"},
        {"source_index", fs::canonical(source_index).string()},
        {"source_fde_build_time_sec", source_meta.at("build_time_sec")},
        {"graph_build_time_sec", std::to_string(graph_build_sec)},
        {"build_time_sec", std::to_string(graph_build_sec)},
        {"index_threads", std::to_string(index_threads)},
        {"diskann_R", std::to_string(graph_R)},
        {"diskann_L", std::to_string(graph_L)},
        {"diskann_alpha", std::to_string(graph_alpha)},
        {"diskann_num_passes", std::to_string(graph_num_passes)},
        {"pq_256_8", "true"},
        {"pq_asymmetric_graph_search", "true"},
        {"pq_group_size", std::to_string(pq_group_size)},
        {"pq_ksub", std::to_string(pq_ksub)},
        {"pq_m", std::to_string(pq_m)},
        {"nbits", "8"},
        {"shared_doc_store", "mvbundle"},
        {"dim", std::to_string(dim)},
        {"B", std::to_string(B)},
        {"R", std::to_string(repetitions)},
        {"d_proj", std::to_string(d_proj)},
        {"d_fde", std::to_string(d_fde)},
        {"ksim", std::to_string(ksim)},
        {"seed", std::to_string(seed)},
        {"n_docs", std::to_string(n_docs)},
        {"source_n_docs", std::to_string(source_n_docs)},
    });

    std::cerr << "[muvera-diskann-indexer] Graph built in " << graph_build_sec
              << "s; formal status remains pending validation.\n";
    return 0;
}
