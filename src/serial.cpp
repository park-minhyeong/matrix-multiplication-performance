#include "../include/matrix.h"

// Serial (CPU) matrix multiplication
void serial_matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    int N = A.N;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A.data[row * N + k] * B.data[k * N + col];
            }
            C.data[row * N + col] = sum;
        }
    }
}
