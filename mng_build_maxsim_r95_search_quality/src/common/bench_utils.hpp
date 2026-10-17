#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <limits>
#include <getopt.h>

namespace baseline {

inline std::string rankings_stem(const std::string& results_path) {
    constexpr const char* suffix = ".json";
    if (results_path.size() >= std::strlen(suffix) &&
        results_path.compare(results_path.size() - std::strlen(suffix),
                             std::strlen(suffix), suffix) == 0) {
        return results_path.substr(0, results_path.size() - std::strlen(suffix));
    }
    return results_path;
}

inline std::string rankings_data_path(const std::string& results_path) {
    return rankings_stem(results_path) + ".rankings.i32.bin";
}

inline std::string rankings_meta_path(const std::string& results_path) {
    return rankings_stem(results_path) + ".rankings.meta.json";
}

inline std::string rankings_scores_path(const std::string& results_path) {
    return rankings_stem(results_path) + ".rankings.scores.fp32.bin";
}

inline void write_rankings_metadata(const std::string& results_path,
                                    size_t n_queries,
                                    size_t width,
                                    bool has_scores,
                                    const std::string& score_semantics = "") {
    const std::string meta_path = rankings_meta_path(results_path);
    const std::string meta_tmp = meta_path + ".tmp";
    std::ofstream meta(meta_tmp, std::ios::trunc);
    if (!meta) throw std::runtime_error("cannot open ranking metadata: " + meta_tmp);
    meta << "{\n"
         << "  \"format\": \""
         << (has_scores ? "rankings_i32_f32_v1" : "rankings_i32_v1") << "\",\n"
         << "  \"doc_ids_dtype\": \"int32\",\n";
    if (has_scores) {
        meta << "  \"scores_dtype\": \"float32\",\n";
    }
    meta << "  \"shape\": [" << n_queries << ", " << width << "],\n"
         << "  \"doc_id_padding_value\": -1,\n";
    if (has_scores) {
        meta << "  \"score_padding_value\": \"nan\",\n";
    }
    meta << "  \"order\": \"method_final_rank\",\n"
         << "  \"doc_ids_path\": \"" << rankings_data_path(results_path) << "\",\n";
    if (has_scores) {
        meta << "  \"scores_path\": \"" << rankings_scores_path(results_path) << "\",\n"
             << "  \"score_semantics\": \"" << score_semantics << "\",\n";
    } else {
        meta << "  \"score_policy\": \"deferred_not_computed\",\n";
    }
    meta << "  \"result_json\": \"" << results_path << "\"\n"
         << "}\n";
    meta.close();
    if (!meta) throw std::runtime_error("failed writing ranking metadata: " + meta_tmp);
    if (std::rename(meta_tmp.c_str(), meta_path.c_str()) != 0) {
        throw std::runtime_error("cannot publish ranking metadata: " + meta_path);
    }
}

// Persist final ranked document IDs before evaluation so metrics can be recomputed offline.
inline void write_rankings_i32(const std::string& results_path,
                               const std::vector<std::vector<int32_t>>& rankings,
                               size_t width) {
    if (results_path.empty()) return;

    const std::string data_path = rankings_data_path(results_path);
    const std::string data_tmp = data_path + ".tmp";
    std::ofstream data(data_tmp, std::ios::binary | std::ios::trunc);
    if (!data) throw std::runtime_error("cannot open ranking artifact: " + data_tmp);

    std::vector<int32_t> row(width, -1);
    for (const auto& ranking : rankings) {
        std::fill(row.begin(), row.end(), -1);
        const size_t count = std::min(width, ranking.size());
        std::copy_n(ranking.begin(), count, row.begin());
        data.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(width * sizeof(int32_t)));
    }
    data.close();
    if (!data) throw std::runtime_error("failed writing ranking artifact: " + data_tmp);
    if (std::rename(data_tmp.c_str(), data_path.c_str()) != 0) {
        throw std::runtime_error("cannot publish ranking artifact: " + data_path);
    }

    write_rankings_metadata(results_path, rankings.size(), width, false);
}

inline void write_ranking_scores_f32(const std::string& results_path,
                                     const std::vector<std::vector<float>>& scores,
                                     size_t width,
                                     const std::string& score_semantics) {
    if (results_path.empty()) return;

    const std::string scores_path = rankings_scores_path(results_path);
    const std::string scores_tmp = scores_path + ".tmp";
    std::ofstream data(scores_tmp, std::ios::binary | std::ios::trunc);
    if (!data) throw std::runtime_error("cannot open ranking scores: " + scores_tmp);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> row(width, nan);
    for (const auto& query_scores : scores) {
        std::fill(row.begin(), row.end(), nan);
        const size_t count = std::min(width, query_scores.size());
        std::copy_n(query_scores.begin(), count, row.begin());
        data.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(width * sizeof(float)));
    }
    data.close();
    if (!data) throw std::runtime_error("failed writing ranking scores: " + scores_tmp);
    if (std::rename(scores_tmp.c_str(), scores_path.c_str()) != 0) {
        throw std::runtime_error("cannot publish ranking scores: " + scores_path);
    }
    write_rankings_metadata(results_path, scores.size(), width, true, score_semantics);
}

struct BenchArgs {
    std::string doc_bundle;
    std::string query_bundle;
    std::string gt_dir;
    std::string index_dir;
    int k = 10;
    std::string output;
    std::vector<std::pair<std::string, std::string>> extra;
};

inline int get_extra_int(const BenchArgs& args, const std::string& key, int default_val) {
    for (auto& [k, v] : args.extra) {
        if (k == key) return std::atoi(v.c_str());
    }
    return default_val;
}

inline float get_extra_float(const BenchArgs& args, const std::string& key, float default_val) {
    for (auto& [k, v] : args.extra) {
        if (k == key) return std::atof(v.c_str());
    }
    return default_val;
}

inline std::string get_extra_string(const BenchArgs& args,
                                    const std::string& key,
                                    const std::string& default_val) {
    for (auto& [k, v] : args.extra) {
        if (k == key) return v;
    }
    return default_val;
}

inline BenchArgs parse_args(int argc, char** argv) {
    BenchArgs args;
    static struct option long_opts[] = {
        {"doc",       required_argument, nullptr, 'd'},
        {"query",     required_argument, nullptr, 'q'},
        {"gt",        required_argument, nullptr, 'g'},
        {"k",         required_argument, nullptr, 'k'},
        {"output",    required_argument, nullptr, 'o'},
        {"param",     required_argument, nullptr, 'p'},
        {"index_dir", required_argument, nullptr, 'i'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:q:g:k:o:p:i:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'd': args.doc_bundle = optarg; break;
            case 'q': args.query_bundle = optarg; break;
            case 'g': args.gt_dir = optarg; break;
            case 'k': args.k = std::atoi(optarg); break;
            case 'o': args.output = optarg; break;
            case 'i': args.index_dir = optarg; break;
            case 'p': {
                std::string s = optarg;
                auto eq = s.find('=');
                if (eq != std::string::npos) {
                    args.extra.emplace_back(s.substr(0, eq), s.substr(eq + 1));
                }
                break;
            }
        }
    }

    return args;
}

// Full results JSON including all metrics
inline void write_results_json(const std::string& path,
                               const std::string& method,
                               const std::string& dataset,
                               int k,
                               const std::vector<std::pair<std::string, std::string>>& params,
                               float recall_10, float recall_100, float recall_1000,
                               float mrr_10, float success_5, float success_100,
                               double build_sec, double load_sec,
                               double search_sec, double per_query_ms,
                               size_t index_bytes) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"method\": \"" << method << "\",\n";
    f << "  \"dataset\": \"" << dataset << "\",\n";
    f << "  \"k\": " << k << ",\n";
    f << "  \"params\": {";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i > 0) f << ", ";
        f << "\"" << params[i].first << "\": \"" << params[i].second << "\"";
    }
    f << "},\n";
    f << "  \"metrics\": {\n";
    f << "    \"recall@10\": " << recall_10 << ",\n";
    f << "    \"recall@100\": " << recall_100 << ",\n";
    f << "    \"recall@1000\": " << recall_1000 << ",\n";
    f << "    \"mrr@10\": " << mrr_10 << ",\n";
    f << "    \"success@5\": " << success_5 << ",\n";
    f << "    \"success@100\": " << success_100 << "\n";
    f << "  },\n";
    f << "  \"timing\": {\n";
    f << "    \"build_sec\": " << build_sec << ",\n";
    f << "    \"load_sec\": " << load_sec << ",\n";
    f << "    \"search_total_sec\": " << search_sec << ",\n";
    f << "    \"search_per_query_ms\": " << per_query_ms << "\n";
    f << "  },\n";
    const char* search_threads = std::getenv("MV_SEARCH_THREADS");
    f << "  \"runtime\": {\n";
    f << "    \"search_threads\": " << (search_threads ? search_threads : "1") << "\n";
    f << "  },\n";
    f << "  \"rankings\": {\n";
    f << "    \"doc_ids_path\": \"" << rankings_data_path(path) << "\",\n";
    const std::string scores_path = rankings_scores_path(path);
    std::ifstream scores(scores_path, std::ios::binary);
    if (scores.good()) {
        f << "    \"scores_path\": \"" << scores_path << "\",\n";
    } else {
        f << "    \"score_policy\": \"deferred_not_computed\",\n";
    }
    f << "    \"metadata_path\": \"" << rankings_meta_path(path) << "\"\n";
    f << "  },\n";
    f << "  \"index_size_bytes\": " << index_bytes << "\n";
    f << "}\n";
}

}  // namespace baseline
