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

// [실험 파라미터 안내]
// matrix size, tile size, 반복 횟수 등은 모두 config.h에서 쉽게 변경할 수 있습니다.
// TILE_SIZES, MATRIX_SIZES, NUM_REPEATS 등 직접 수정 가능합니다.
int main() {
    std::cout << "Matrix Multiplication Performance Comparison (Serial vs. Naive vs. Tiled)\n";
    std::cout << "Matrix sizes:";
    for (int i = 0; i < NUM_SIZES; ++i) std::cout << " " << MATRIX_SIZES[i];
    std::cout << "\nTile sizes:";
    for (int t = 0; t < NUM_TILE_SIZES; ++t) std::cout << " " << TILE_SIZES[t];
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
        std::vector<float> tiled_times(NUM_TILE_SIZES);
        std::vector<bool> tiled_oks(NUM_TILE_SIZES);
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
            tiled_times[t] = tiled_time;
            tiled_oks[t] = compare_matrix(C_serial, C_tiled, EPSILON);
        }
        // 4. 결과 출력 (CPU, Naive, Tiled 모두)
        bool naive_ok = compare_matrix(C_serial, C_naive, EPSILON);
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  CPU:   " << cpu_time << " ms\n";
        std::cout << "  Naive: " << naive_time << " ms [" << (naive_ok ? "OK" : "FAIL") << "]\n";
        for (int t = 0; t < NUM_TILE_SIZES; ++t) {
            std::cout << "  Tiled (tile_size=" << TILE_SIZES[t] << "): " << tiled_times[t]
                      << " ms [" << (tiled_oks[t] ? "OK" : "FAIL") << "]\n";
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
