#pragma once

#include "common/index_io.hpp"

#include <faiss/Index.h>
#include <faiss/index_io.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace colbert {

constexpr const char* kFaissIndexFile = "faiss.ivfpq.index";
constexpr const char* kLayout = "faiss_ivfpq_token_index";

inline std::string index_file(const std::string& dir) {
    return dir + "/" + kFaissIndexFile;
}

inline void write_faiss_index(const std::string& dir, const faiss::Index& index) {
    faiss::write_index(&index, index_file(dir).c_str());
}

inline std::unique_ptr<faiss::Index> read_faiss_index(const std::string& dir) {
    faiss::Index* raw = faiss::read_index(index_file(dir).c_str());
    if (!raw) {
        throw std::runtime_error("failed to read FAISS index from " + index_file(dir));
    }
    return std::unique_ptr<faiss::Index>(raw);
}

inline std::size_t logical_index_bytes(const std::string& dir) {
    return baseline::directory_file_size(dir);
}

}  // namespace colbert
