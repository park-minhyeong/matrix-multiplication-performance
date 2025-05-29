#include "../include/matrix.h"
#include "../include/config.h"
#include <cuda_runtime.h>

// Tiled CUDA kernel (shared memory)
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int N, int tile_size) {
    extern __shared__ float tile[];
    float* As = tile;
    float* Bs = tile + tile_size * tile_size;
    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + tile_size - 1) / tile_size; ++t) {
        // Load tiles with boundary check
        int tiledRow = row;
        int tiledCol = t * tile_size + threadIdx.x;
        int tiledRowB = t * tile_size + threadIdx.y;
        int tiledColB = col;
        As[threadIdx.y * tile_size + threadIdx.x] = (tiledRow < N && tiledCol < N) ? A[tiledRow * N + tiledCol] : 0.0f;
        Bs[threadIdx.y * tile_size + threadIdx.x] = (tiledRowB < N && tiledColB < N) ? B[tiledRowB * N + tiledColB] : 0.0f;
        __syncthreads();
        for (int k = 0; k < tile_size; ++k)
            sum += As[threadIdx.y * tile_size + k] * Bs[k * tile_size + threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Host wrapper for tiled CUDA kernel
void tiled_cuda_matmul(const Matrix& A, const Matrix& B, Matrix& C, int tile_size) {
    int N = A.N;
    size_t bytes = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A.data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data, bytes, cudaMemcpyHostToDevice);
    dim3 block(tile_size, tile_size);
    dim3 grid((N + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);
    size_t shared_mem = 2 * tile_size * tile_size * sizeof(float);
    tiled_matmul_kernel<<<grid, block, shared_mem>>>(d_A, d_B, d_C, N, tile_size);
    cudaMemcpy(C.data, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
