/**
 * 尝试cuda graph构建，实例化和运行，并和不通过graph直接运行的stream比较耗时。
 * 发现有graph基本上比没有graph要快一些。
 */

#include <iostream>

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
 * @param g_a: Matrix a in gpu.
 * @param b: Matrix b.
 * @param g_b: Matrix b in gpu.
 * @param c: The matrix to save ab.
 * @param g_c: The matrix c in gpu.
 * @param use_shared_mem: If use shared memory.
 * @param stream: The cuda stream.
 */
void MatInnerProdInGpu(const Matrix& a, const Matrix& g_a, const Matrix& b, const Matrix& g_b, const Matrix& c,
                       const Matrix& g_c, bool use_shared_mem, const cudaStream_t& stream) {
  /* Load mat a and b to gpu memory. */
  size_t size = a.width * a.height * sizeof(double);
  CHECK(cudaMemcpyAsync(g_a.elements, a.elements, size, cudaMemcpyHostToDevice, stream));
  size = b.width * b.height * sizeof(double);
  CHECK(cudaMemcpyAsync(g_b.elements, b.elements, size, cudaMemcpyHostToDevice, stream));

  /* Invoke kernel function. */
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dim_grid(g_c.width / dim_block.x, g_c.height / dim_block.y);
  if (use_shared_mem) {
    MatInnerProdKernelWithSharedMem<<<dim_grid, dim_block, 0, stream>>>(g_a, g_b, g_c);
  } else {
    MatInnerProdKernel<<<dim_grid, dim_block, 0, stream>>>(g_a, g_b, g_c);
  }

  /* Copy mat c from gpu memory into cpu memory. */
  size = c.width * c.height * sizeof(double);
  CHECK(cudaMemcpyAsync(c.elements, g_c.elements, size, cudaMemcpyDeviceToHost, stream));
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
  Matrix g_a = {a.width, a.height, nullptr};
  Matrix b = {640, 2560, nullptr};
  Matrix g_b = {b.width, b.height, nullptr};
  Matrix c1 = {b.width, a.height, nullptr}; // C = A * B
  Matrix g_c1 = {b.width, a.height, nullptr};
  Matrix c2 = {b.width, a.height, nullptr};
  Matrix g_c2 = {b.width, a.height, nullptr};

  size_t size_a = a.width * a.height * sizeof(double);
  size_t size_b = b.width * b.height * sizeof(double);
  size_t size_c = c1.width * c1.height * sizeof(double);

  /* CPU and GPU Malloc. */
  a.elements = (double*) malloc(size_a);
  CHECK(cudaMalloc(&g_a.elements, size_a));
  b.elements = (double*) malloc(size_b);
  CHECK(cudaMalloc(&g_b.elements, size_b));
  c1.elements = (double*) malloc(size_c);
  CHECK(cudaMalloc(&g_c1.elements, size_c));
  c2.elements = (double*) malloc(size_c);
  CHECK(cudaMalloc(&g_c2.elements, size_c));

  InitialData(a.elements, a.width * a.height);
  InitialData(b.elements, b.width * b.height);

  cudaEvent_t start, stop; // Used to calc the time cost.

  /* Test cost time of gpu loop op without graph. */
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < 50; i++) {
    MatInnerProdInGpu(a, g_a, b, g_b, c1, g_c1, true, stream);
  }
  CHECK(cudaEventRecord(stop, stream));
  CHECK(cudaEventSynchronize(stop));

  float no_graph_time;
  CHECK(cudaEventElapsedTime(&no_graph_time, start, stop));
  printf("GPU op loop with no graph Execution Time: %f sec\n", (no_graph_time / 1000));

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaStreamDestroy(stream));

  /* Test cost time of gpu loop op with graph. */
  bool graph_created = false;
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  cudaStream_t stream1;
  CHECK(cudaStreamCreate(&stream1));
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < 50; i++) {
    if (!graph_created) {
      CHECK(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
      MatInnerProdInGpu(a, g_a, b, g_b, c2, g_c2, true, stream1);
      CHECK(cudaStreamEndCapture(stream1, &graph));
      CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
      graph_created = true;
      CHECK(cudaEventRecord(start, stream));
    }
    CHECK(cudaGraphLaunch(graph_exec, stream1));
  }
  CHECK(cudaEventRecord(stop, stream));
  CHECK(cudaEventSynchronize(stop));

  float with_graph_time;
  CHECK(cudaEventElapsedTime(&with_graph_time, start, stop));
  printf("GPU op loop with graph Execution Time: %f sec\n", with_graph_time / 1000);
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaStreamDestroy(stream1));

  CheckResult(c1.elements, c2.elements, c1.height * c1.width);

  /* CPU and GPU free. */
  free(a.elements);
  CHECK(cudaFree(g_a.elements));
  free(b.elements);
  CHECK(cudaFree(g_b.elements));
  free(c1.elements);
  CHECK(cudaFree(g_c1.elements));
  free(c2.elements);
  CHECK(cudaFree(g_c2.elements));

  cudaDeviceReset();
  return 0;
}
