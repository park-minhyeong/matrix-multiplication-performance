#pragma once

// 실험 파라미터 정의
constexpr int NUM_SIZES = 5;
constexpr int MATRIX_SIZES[NUM_SIZES] = { 64, 128, 256, 512, 1024 }; // 예시
constexpr int NUM_REPEATS = 20; // 각 실험 반복 횟수
constexpr int NUM_TILE_SIZES = 3;
constexpr int TILE_SIZES[NUM_TILE_SIZES] = {8, 16, 32}; // 여러 타일 크기 실험
constexpr float EPSILON = 1e-3f;
