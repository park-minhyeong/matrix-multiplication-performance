#include "../include/matrix.h"
#include <cuda_runtime.h>

// Naive CUDA kernel (global memory only)
__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Host wrapper for naive CUDA kernel
void naive_cuda_matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    int N = A.N;
    size_t bytes = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A.data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data, bytes, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    naive_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaMemcpy(C.data, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
