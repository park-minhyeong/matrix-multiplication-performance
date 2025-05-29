# GPU Matrix Multiplication Makefile

NVCC = nvcc
CXX = g++
INCLUDES = -I./include
SRC_DIR = ./src
OBJS = $(SRC_DIR)/matrix.o $(SRC_DIR)/timer.o $(SRC_DIR)/serial.o $(SRC_DIR)/cuda_naive.o $(SRC_DIR)/cuda_tiled.o
TARGET = matmul

all: $(TARGET)

$(SRC_DIR)/matrix.o: $(SRC_DIR)/matrix.cpp ./include/matrix.h
	$(CXX) -c $< -o $@ $(INCLUDES)

$(SRC_DIR)/timer.o: $(SRC_DIR)/timer.cpp ./include/timer.h
	$(CXX) -c $< -o $@ $(INCLUDES)

$(SRC_DIR)/serial.o: $(SRC_DIR)/serial.cpp ./include/matrix.h
	$(CXX) -c $< -o $@ $(INCLUDES)

$(SRC_DIR)/cuda_naive.o: $(SRC_DIR)/cuda_naive.cu ./include/matrix.h
	$(NVCC) -c $< -o $@ $(INCLUDES) -O3

$(SRC_DIR)/cuda_tiled.o: $(SRC_DIR)/cuda_tiled.cu ./include/matrix.h ./include/config.h
	$(NVCC) -c $< -o $@ $(INCLUDES) -O3

$(TARGET): $(SRC_DIR)/main.cpp $(OBJS)
	$(NVCC) $^ -o $@ $(INCLUDES) -O3

clean: 
	rm -f $(SRC_DIR)/*.o $(TARGET)
