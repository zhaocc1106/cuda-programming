//
// Created by zhaocc on 2021/6/8.
//

#include <iostream>
#include <sstream>

#include "cuda_start.h"

#define BLOCK_SIZE 16

/*-----------------------------------Mat inner product in gpu without shared memory.-----------------------------------*/

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

/*-----------------------------------Mat inner product in gpu with shared memory.-----------------------------------*/
// SubMatrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
  int width;
  int height;
  int stride;
  double* elements;
} SubMatrix;

// Get a sub matrix element
__device__ double GetElement(const SubMatrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}

// Set a sub matrix element
__device__ void SetElement(SubMatrix A, int row, int col,
                           double value) {
  A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ SubMatrix GetSubMatrix(Matrix A, int row, int col) {
  SubMatrix sub;
  sub.width = BLOCK_SIZE;
  sub.height = BLOCK_SIZE;
  sub.stride = A.width;
  sub.elements = &A.elements[A.width * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return sub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatInnerProdKernelWithSharedMem(Matrix A, Matrix B, Matrix C) {
  // Block row and column
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each thread block computes one sub-matrix Csub of C
  SubMatrix c_sub = GetSubMatrix(C, block_row, block_col);

  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  double c_value = 0;

  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

    // Get sub-matrix Asub of A
    SubMatrix a_sub = GetSubMatrix(A, block_row, m);

    // Get sub-matrix Bsub of B
    SubMatrix b_sub = GetSubMatrix(B, m, block_col);

    // Shared memory used to store Asub and Bsub respectively
    __shared__ double a_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double b_s[BLOCK_SIZE][BLOCK_SIZE];

    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    a_s[row][col] = GetElement(a_sub, row, col);
    b_s[row][col] = GetElement(b_sub, row, col);

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      c_value += a_s[row][e] * b_s[e][col];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write Csub to device memory
  // Each thread writes one element
  SetElement(c_sub, row, col, c_value);
}

/**
 * Matrix inner product in gpu. Matrix dimensions are assumed to be multiples of BLOCK_SIZE
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 * @param use_shared_mem: If use shared memory.
 */
void MatInnerProdInGpu(const Matrix& a, const Matrix& b, Matrix& c, bool use_shared_mem) {
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
  if (use_shared_mem) {
    MatInnerProdKernelWithSharedMem<<<dim_grid, dim_block>>>(g_a, g_b, g_c);
  } else {
    MatInnerProdKernel<<<dim_grid, dim_block>>>(g_a, g_b, g_c);
  }

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

/*-----------------------------------Mat inner product in cpu.-----------------------------------*/

/**
 * Matrix inner product in cpu.
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
void MatInnerProdInCpu(const Matrix& a, const Matrix& b, Matrix& c) {
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
  Matrix a = {1280, 640, nullptr};
  Matrix b = {640, 2560, nullptr};
  size_t size_a = a.width * a.height * sizeof(double);
  size_t size_b = b.width * b.height * sizeof(double);
  a.elements = (double*) malloc(size_a);
  b.elements = (double*) malloc(size_b);

  InitialData(a.elements, a.width * a.height);
  InitialData(b.elements, b.width * b.height);

  Matrix c1 = {0, 0, nullptr};
  Matrix c2 = {0, 0, nullptr};
  Matrix c3 = {0, 0, nullptr};

  double gpu_start = cpuSecond(); // Mark GPU start time
  MatInnerProdInGpu(a, b, c1, false);
  CHECK(cudaDeviceSynchronize());
  double gpu_time = cpuSecond() - gpu_start;
  printf("GPU Execution Time: %f sec\n", gpu_time); // GPU Execution Time: 0.061128 sec

  double gpu_use_shared_mem_start = cpuSecond(); // Mark GPU with shared memory start time
  MatInnerProdInGpu(a, b, c2, true);
  CHECK(cudaDeviceSynchronize());
  double gpu_use_shared_mem_time = cpuSecond() - gpu_use_shared_mem_start;
  printf("GPU with shared memory Execution Time: %f sec\n", gpu_use_shared_mem_time); // GPU with shared memory Execution Time: 0.011619 sec

  double cpu_start = cpuSecond(); // Mark CPU start time
  MatInnerProdInCpu(a, b, c3);
  double cpu_time = cpuSecond() - cpu_start;
  printf("CPU Execution Time: %f sec\n", cpu_time); // CPU Execution Time: 1.684192 sec

  CheckResult(c3.elements, c1.elements, c1.width * c1.height);
  CheckResult(c3.elements, c2.elements, c2.width * c2.height);

  if (c1.elements) {
    free(c1.elements);
  }
  if (c2.elements) {
    free(c2.elements);
  }

  cudaDeviceReset();
  return 0;
}
