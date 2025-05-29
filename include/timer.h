#pragma once
#include <cuda_runtime.h>
#include <chrono>

// CPU 타이머 (std::chrono)
class CpuTimer {
public:
    void start();
    void stop();
    double elapsed_ms() const;
private:
    std::chrono::high_resolution_clock::time_point t_start, t_end;
};

// GPU 타이머 (cudaEvent)
class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();
    void start();
    void stop();
    float elapsed_ms() const;
private:
    cudaEvent_t start_event, stop_event;
};
