#include "../include/matrix.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <cstring>

Matrix create_matrix(int N, bool random_init) {
    Matrix m;
    m.N = N;
    m.data = new float[N * N];
    if (random_init) {
        std::mt19937 gen(42); // 고정 시드 
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < N * N; ++i)
            m.data[i] = dist(gen);
    } else {
        std::memset(m.data, 0, N * N * sizeof(float));
    }
    return m;
}

void free_matrix(Matrix& m) {
    delete[] m.data; 
    m.data = nullptr;
    m.N = 0;
}

bool compare_matrix(const Matrix& a, const Matrix& b, float epsilon) {
    if (a.N != b.N) return false;
    for (int i = 0; i < a.N * a.N; ++i) {
        if (std::abs(a.data[i] - b.data[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void copy_matrix(const Matrix& src, Matrix& dst) {
    if (src.N != dst.N) return;
    std::memcpy(dst.data, src.data, src.N * src.N * sizeof(float));
}

void print_matrix(const Matrix& m) {
    for (int i = 0; i < m.N; ++i) {
        for (int j = 0; j < m.N; ++j) {
            std::cout << m.data[i * m.N + j] << " ";
        }
        std::cout << std::endl;
    }
}
