#include "common/ranking_artifact.hpp"

namespace baseline {

inline void write_rankings_ids_only(
        const std::string& results_path,
        const std::vector<std::vector<int32_t>>& rankings,
        size_t width,
        const std::string&,
        const std::string&) {
    write_rankings_i32(results_path, rankings, width);
}

}  // namespace baseline

#define write_rankings_with_exact_maxsim write_rankings_ids_only
#include "mng_build_maxsim_r95_searcher_impl.cpp"
