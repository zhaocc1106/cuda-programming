//
// Created by zhaocc on 2021/6/8.
//

#include <iostream>

#include "cuda_start.h"

#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
  double* elements;
} Matrix;

/**
 * The cuda kernel function of matrix inner product.
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
__global__ void MatInnerProdKernel(Matrix a, Matrix b, Matrix c) {
  /* Each thread computes one element of matrxi c. */
  double c_element = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y; // The row num of element.
  int col = blockIdx.x * blockDim.x + threadIdx.x; // The col num of element.
  for (int i = 0; i < a.width; i++) {
    c_element += a.elements[row * a.width + i] * b.elements[i * b.width + col];
  }
  c.elements[row * c.width + col] = c_element;
}

/**
 * Matrix inner product in gpu. Matrix dimensions are assumed to be multiples of BLOCK_SIZE
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
void MatInnerProdInGpu(const Matrix& a, const Matrix& b, Matrix& c) {
  /* Load mat a and b to gpu memory. */
  Matrix g_a = {a.width, a.height, nullptr};
  size_t size = a.width * a.height * sizeof(double);
  CHECK(cudaMalloc(&g_a.elements, size));
  CHECK(cudaMemcpy(g_a.elements, a.elements, size, cudaMemcpyHostToDevice));
  Matrix g_b = {b.width, b.height, nullptr};
  size = b.width * b.height * sizeof(double);
  CHECK(cudaMalloc(&g_b.elements, size));
  CHECK(cudaMemcpy(g_b.elements, b.elements, size, cudaMemcpyHostToDevice));

  /* Alloc mat c in gpu memory. */
  Matrix g_c = {b.width, a.height, nullptr};
  size = g_c.width * g_c.height * sizeof(double);
  CHECK(cudaMalloc(&g_c.elements, size));

  /* Invoke kernel function. */
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dim_grid(g_c.width / dim_block.x, g_c.height / dim_block.y);
  MatInnerProdKernel<<<dim_grid, dim_block>>>(g_a, g_b, g_c);

  /* Copy mat c from gpu memory into cpu memory. */
  c.width = g_c.width;
  c.height = g_c.height;
  size = c.width * c.height * sizeof(double);
  if (c.elements == nullptr) {
    c.elements = (double*) malloc(size);
  }
  CHECK(cudaMemcpy(c.elements, g_c.elements, size, cudaMemcpyDeviceToHost));

  /* Free gpu memory. */
  CHECK(cudaFree(g_a.elements));
  CHECK(cudaFree(g_b.elements));
  CHECK(cudaFree(g_c.elements));
}

/**
 * Matrix inner product in cpu.
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
void MatInnerProdInCpu(Matrix a, Matrix b, Matrix c) {
  c.width = b.width;
  c.height = a.height;
  size_t size = c.width * c.height * sizeof(double);
  if (c.elements == nullptr) {
    c.elements = (double*) malloc(size);
  }

  for (int row = 0; row < a.height; row++) {
    for (int col = 0; col < b.width; col++) {
      double c_element = 0;
      for (int i = 0; i < a.width; i++) {
        c_element += a.elements[row * a.width + i] * b.elements[i * b.width + col];
      }
      c.elements[row * c.width + col] = c_element;
    }
  }
}

/* Initialize data by random. */
void InitialData(double* p, size_t size) {
  time_t t;
  srand((unsigned) time(&t));
  for (int i = 0; i < size; i++) {
    p[i] = (double) (rand() & 0xffff) / 1000.0f;
  }
}

// Check calc result between cpu and gpu.
void CheckResult(double* cpu_ref, double* gpu_ref, const int size) {
  double epsilon = 1.0E-8; // 错误容忍度
  for (int i = 0; i < size; i++) {
    if (abs(cpu_ref[i] - gpu_ref[i]) > epsilon) {
      printf("Results don\'t match!\n");
      printf("%f(cpu_ref[%d] )!= %f(gpu_ref[%d])\n", cpu_ref[i], i, gpu_ref[i], i);
      return;
    }
  }
  printf("Check result success!\n");
}

int main(int argc, char** argv) {
  printf("starting...\n");
  initDevice(0);

  /* 假设矩阵的dim size为BLOCK_SIZE的整数倍，如果矩阵比较小，gpu计算速度并没有cpu快 */
  Matrix a = {12800, 6400, nullptr};
  Matrix b = {6400, 25600, nullptr};
  size_t size_a = a.width * a.height * sizeof(double);
  size_t size_b = b.width * b.height * sizeof(double);
  a.elements = (double*) malloc(size_a);
  b.elements = (double*) malloc(size_b);

  InitialData(a.elements, a.width * a.height);
  InitialData(b.elements, b.width * b.height);

  Matrix c1 = {0, 0, nullptr};
  Matrix c2 = {0, 0, nullptr};

  double gpu_start = cpuSecond(); // Mark GPU start time
  MatInnerProdInGpu(a, b, c1);
  CHECK(cudaDeviceSynchronize());
  double gpu_time = cpuSecond() - gpu_start;
  printf("GPU Execution Time: %f sec\n", gpu_time);

  double cpu_start = cpuSecond(); // Mark CPU start time
  MatInnerProdInCpu(a, b, c2);
  double cpu_time = cpuSecond() - cpu_start;
  printf("CPU Execution Time: %f sec\n", cpu_time);

  CheckResult(c1.elements, c2.elements, c1.width * c1.height);

  if (c1.elements) {
    free(c1.elements);
  }
  if (c2.elements) {
    free(c2.elements);
  }

  cudaDeviceReset();
  return 0;
}
