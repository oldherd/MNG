#pragma once
#include <chrono>

namespace baseline {

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop()  { end_   = std::chrono::high_resolution_clock::now(); }

    double elapsed_sec() const {
        return std::chrono::duration<double>(end_ - start_).count();
    }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

}  // namespace baseline
