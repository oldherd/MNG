#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sys/stat.h>
#include <dirent.h>

namespace baseline {

inline void ensure_dir(const std::string& dir) {
    mkdir(dir.c_str(), 0755);
}

inline void write_bin(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_bin: cannot open " + path);
    f.write(reinterpret_cast<const char*>(data), bytes);
    if (!f) throw std::runtime_error("write_bin: write failed " + path);
}

inline void read_bin(const std::string& path, void* data, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_bin: cannot open " + path);
    f.read(reinterpret_cast<char*>(data), bytes);
    if (!f) throw std::runtime_error("read_bin: read failed " + path);
}

inline std::vector<char> read_bin_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("read_bin_vec: cannot open " + path);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

inline size_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return (size_t)st.st_size;
}

inline size_t directory_file_size(const std::string& dir) {
    DIR* handle = opendir(dir.c_str());
    if (!handle) return 0;

    size_t total = 0;
    struct dirent* entry;
    while ((entry = readdir(handle)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        std::string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0) continue;

        if (S_ISREG(st.st_mode)) {
            total += (size_t)st.st_size;
        } else if (S_ISDIR(st.st_mode)) {
            total += directory_file_size(path);
        }
    }

    closedir(handle);
    return total;
}

// Minimal JSON writer for string key-value pairs (no external deps)
inline void write_meta(const std::string& dir,
                       const std::vector<std::pair<std::string, std::string>>& kv) {
    std::string path = dir + "/meta.json";
    std::ofstream f(path);
    if (!f) throw std::runtime_error("write_meta: cannot open " + path);
    f << "{\n";
    for (size_t i = 0; i < kv.size(); ++i) {
        f << "  \"" << kv[i].first << "\": \"" << kv[i].second << "\"";
        if (i + 1 < kv.size()) f << ",";
        f << "\n";
    }
    f << "}\n";
}

// Minimal JSON reader (string values only, no nested objects)
inline std::unordered_map<std::string, std::string> read_meta(const std::string& dir) {
    std::string path = dir + "/meta.json";
    std::ifstream f(path);
    if (!f) throw std::runtime_error("read_meta: cannot open " + path);
    std::unordered_map<std::string, std::string> result;
    std::string line;
    while (std::getline(f, line)) {
        // Find key: "key": "value"
        auto kstart = line.find('"');
        if (kstart == std::string::npos) continue;
        auto kend = line.find('"', kstart + 1);
        if (kend == std::string::npos) continue;
        auto vstart = line.find('"', kend + 1);
        if (vstart == std::string::npos) continue;
        // Find matching closing quote (handle no escapes — our values are simple)
        auto vend = line.find('"', vstart + 1);
        if (vend == std::string::npos) continue;
        std::string key = line.substr(kstart + 1, kend - kstart - 1);
        std::string val = line.substr(vstart + 1, vend - vstart - 1);
        result[key] = val;
    }
    return result;
}

// Write a vector<T> as raw binary
template<typename T>
void write_vec_bin(const std::string& path, const std::vector<T>& vec) {
    write_bin(path, vec.data(), vec.size() * sizeof(T));
}

// Read raw binary into vector<T> (infers count from file size)
template<typename T>
std::vector<T> read_vec_bin(const std::string& path) {
    size_t sz = file_size(path);
    if (sz % sizeof(T) != 0) {
        throw std::runtime_error("read_vec_bin: file size not aligned to element size: " + path);
    }
    std::vector<T> vec(sz / sizeof(T));
    read_bin(path, vec.data(), sz);
    return vec;
}

// Write variable-length lists (e.g., IVF inverted lists, adjacency lists)
// Format: [n_lists:uint32][offset_0:uint64]...[offset_n:uint64][data...]
// Each list's data is contiguous T elements; offsets[i] = start of list i in element count
template<typename T>
void write_varlists(const std::string& path, const std::vector<std::vector<T>>& lists) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_varlists: cannot open " + path);
    uint32_t n = (uint32_t)lists.size();
    f.write(reinterpret_cast<const char*>(&n), 4);
    // Write offsets (n+1 uint64 values)
    std::vector<uint64_t> offsets(n + 1);
    offsets[0] = 0;
    for (uint32_t i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + lists[i].size();
    }
    f.write(reinterpret_cast<const char*>(offsets.data()), (n + 1) * sizeof(uint64_t));
    // Write all data contiguously
    for (auto& lst : lists) {
        if (!lst.empty()) {
            f.write(reinterpret_cast<const char*>(lst.data()), lst.size() * sizeof(T));
        }
    }
}

template<typename T>
std::vector<std::vector<T>> read_varlists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_varlists: cannot open " + path);
    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), 4);
    std::vector<uint64_t> offsets(n + 1);
    f.read(reinterpret_cast<char*>(offsets.data()), (n + 1) * sizeof(uint64_t));
    std::vector<std::vector<T>> result(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t count = offsets[i + 1] - offsets[i];
        result[i].resize(count);
        f.read(reinterpret_cast<char*>(result[i].data()), count * sizeof(T));
    }
    return result;
}

}  // namespace baseline
