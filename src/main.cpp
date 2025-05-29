#include "../include/matrix.h"
#include "../include/timer.h"
#include "../include/config.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>

// 각 구현 함수 선언
void serial_matmul(const Matrix& A, const Matrix& B, Matrix& C);
void naive_cuda_matmul(const Matrix& A, const Matrix& B, Matrix& C);
void tiled_cuda_matmul(const Matrix& A, const Matrix& B, Matrix& C, int tile_size);

int main() {
    std::cout << "Matrix Multiplication Performance Comparison (Serial vs. Naive vs. Tiled)\n";
    std::cout << "Matrix sizes:";
    for (int i = 0; i < NUM_SIZES; ++i) std::cout << " " << MATRIX_SIZES[i];
    std::cout << "\nRepeats: " << NUM_REPEATS << "\n\n";

    for (int size_idx = 0; size_idx < NUM_SIZES; ++size_idx) {
        int N = MATRIX_SIZES[size_idx];
        std::cout << "[Matrix size " << N << "x" << N << "]\n";
        // 행렬 생성 및 초기화
        Matrix A = create_matrix(N, true);
        Matrix B = create_matrix(N, true);
        Matrix C_serial = create_matrix(N);
        Matrix C_naive = create_matrix(N);
        Matrix C_tiled = create_matrix(N);
        // 1. Serial (CPU)
        double cpu_time = 0.0;
        for (int rep = 0; rep < NUM_REPEATS; ++rep) {
            CpuTimer timer;
            timer.start();
            serial_matmul(A, B, C_serial);
            timer.stop();
            cpu_time += timer.elapsed_ms();
        }
        cpu_time /= NUM_REPEATS;
        // 2. Naive CUDA
        float naive_time = 0.0f;
        for (int rep = 0; rep < NUM_REPEATS; ++rep) {
            GpuTimer timer;
            timer.start();
            naive_cuda_matmul(A, B, C_naive);
            timer.stop();
            naive_time += timer.elapsed_ms();
        }
        naive_time /= NUM_REPEATS;
        // 3. Tiled CUDA
        // Tiled CUDA: 여러 타일 크기에 대해 반복 실험
        for (int t = 0; t < NUM_TILE_SIZES; ++t) {
            int tile_size = TILE_SIZES[t];
            float tiled_time = 0.0f;
            for (int rep = 0; rep < NUM_REPEATS; ++rep) {
                GpuTimer timer;
                timer.start();
                tiled_cuda_matmul(A, B, C_tiled, tile_size);
                timer.stop();
                tiled_time += timer.elapsed_ms();
            }
            tiled_time /= NUM_REPEATS;
            bool tiled_ok = compare_matrix(C_serial, C_tiled, EPSILON);
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Tiled (tile_size=" << tile_size << "): " << tiled_time << " ms [" << (tiled_ok ? "OK" : "FAIL") << "]\n";
        }
        std::cout << std::endl;
        free_matrix(A);
        free_matrix(B);
        free_matrix(C_serial);
        free_matrix(C_naive);
        free_matrix(C_tiled);
    }
    return 0;
}
