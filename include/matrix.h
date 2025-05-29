#pragma once
#include <vector>

// 행렬 데이터 구조체 (1D float vector, N*N)
struct Matrix {
    int N;
    float* data;
};

// 행렬 생성/해제
Matrix create_matrix(int N, bool random_init = false);
void free_matrix(Matrix& m);

// 행렬 비교 (CPU vs GPU, epsilon 허용)
bool compare_matrix(const Matrix& a, const Matrix& b, float epsilon);

// 행렬 복사
void copy_matrix(const Matrix& src, Matrix& dst);

// 행렬 출력 (디버깅용)
void print_matrix(const Matrix& m);
