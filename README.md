# GPU Matrix Multiplication 
Serial vs. Naive CUDA vs. Tiled CUDA

## Project Structure

```
gpu-computing/
├── include/
│   ├── matrix.h         
│   ├── timer.h          
│   └── config.h         
├── src/
│   ├── matrix.cpp    
│   ├── timer.cpp        
│   ├── serial.cpp       
│   ├── cuda_naive.cu    
│   ├── cuda_tiled.cu    
│   └── main.cpp         
├── Makefile             
└── README.md        
```

## Implementation
You can run in WSL2/Ubuntu with CUDA.
// build
```sh
make
```
// run
```sh
./matmul
```

## Features
- Serial(CPU), Naive CUDA, Tiled CUDA(Shared Memory) 행렬곱 구현
- 다양한 행렬 크기(64~1024)에서 20회 반복, 평균 실행시간 측정
- CPU/GPU 결과 자동 검증 (epsilon 오차)
- 타이머 유틸(chrono, cudaEvent)로 정확한 타이밍
- 관심사 분리 및 모듈화

## Example Output
```
Matrix Multiplication Performance Comparison (Serial vs. Naive vs. Tiled)
Matrix sizes: 64 128 256 512 1024
Repeats: 20

[Matrix size 64x64]
  CPU:   0.52 ms
  Naive: 0.19 ms [OK]
  Tiled: 0.11 ms [OK]
...
```