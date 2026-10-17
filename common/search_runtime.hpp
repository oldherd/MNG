#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace baseline {

inline void set_env_if_possible(const char* key, const char* value) {
#if defined(_WIN32)
    (void)key;
    (void)value;
#else
    ::setenv(key, value, 1);
#endif
}

inline void enforce_single_thread_searcher(const char* method) {
    set_env_if_possible("MV_SEARCH_THREADS", "1");
    set_env_if_possible("OMP_NUM_THREADS", "1");
    set_env_if_possible("OMP_DYNAMIC", "FALSE");
    set_env_if_possible("OPENBLAS_NUM_THREADS", "1");
    set_env_if_possible("MKL_NUM_THREADS", "1");
    set_env_if_possible("BLIS_NUM_THREADS", "1");
    set_env_if_possible("VECLIB_MAXIMUM_THREADS", "1");
    set_env_if_possible("NUMEXPR_NUM_THREADS", "1");
    set_env_if_possible("PARLAY_NUM_THREADS", "1");

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif

    std::ostringstream msg;
    msg << "[" << method << "] runtime: single-thread search enforced";
#ifdef _OPENMP
    msg << " (OpenMP max_threads=" << omp_get_max_threads() << ")";
#endif
    std::cerr << msg.str() << std::endl;
}

}  // namespace baseline
