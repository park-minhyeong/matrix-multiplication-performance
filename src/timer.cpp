#include "../include/timer.h"

// ================= CPU Timer =================
void CpuTimer::start() {
    t_start = std::chrono::high_resolution_clock::now();
}
void CpuTimer::stop() {
    t_end = std::chrono::high_resolution_clock::now();
}
double CpuTimer::elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

// ================= GPU Timer =================
GpuTimer::GpuTimer() {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
}
GpuTimer::~GpuTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}
void GpuTimer::start() {
    cudaEventRecord(start_event, 0);
}
void GpuTimer::stop() {
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
}
float GpuTimer::elapsed_ms() const {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_event, stop_event);
    return ms;
}
