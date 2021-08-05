/**
 * 实现cuda版归并排序
 */

#include <iostream>
#include <vector>

#include "cuda_start.h"

#define DATA_SIZE 10000

/**
 * 归并两个数组
 * @param data: 完整数组
 * @param tmp: 临时数组空间用于归并
 * @param sorted_size: 已经排好序的子数组大小
 * @param arr_size: 完整数组大小
 */
__global__ void MergeSortSub(float* data, float* tmp, int sorted_size, int arr_size) {
  int block_id = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x; // 计算当前线程属于第几个block
  int id = block_id * blockDim.x + threadIdx.x; // 根据当前block id计算分给当前线程的待归并的子数组id

  unsigned long index1, index2, end_index1, end_index2, tmp_index;
  index1 = id * 2 * sorted_size; // 待归并的数组1当前位置
  end_index1 = index1 + sorted_size; // 待归并的数组1终止位置
  index2 = end_index1; // 待归并的数组2当前位置
  end_index2 = index2 + sorted_size; // 待归并的数组2终止位置
  tmp_index = id * 2 * sorted_size; // 临时归并空间dev_tmp的index

  if (index1 >= arr_size) return;

  // 调整边界
  if (end_index1 > arr_size) {
    end_index1 = arr_size;
    index2 = end_index2 = arr_size;
  }
  if (index2 > arr_size) {
    index2 = end_index2 = arr_size;
  }
  if (end_index2 > arr_size) {
    end_index2 = arr_size;
  }

  bool done = false;
  while (!done) {
    if ((index1 == end_index1) && (index2 < end_index2)) // 数组1已全部放到tmp中
      tmp[tmp_index++] = data[index2++];
    else if ((index2 == end_index2) && (index1 < end_index1)) // 数组2已全部放到tmp中
      tmp[tmp_index++] = data[index1++];
    else if (data[index1] < data[index2]) // 数组1当前位置数据小于数组2当前位置数据
      tmp[tmp_index++] = data[index1++];
    else
      tmp[tmp_index++] = data[index2++]; // 数组2当前位置数据小于数组1当前位置数据
    if ((index1 == end_index1) && (index2 == end_index2)) // 两个数组全部放到tmp中，归并完毕
      done = true;
  }
}

/**
 * Cuda版归并排序
 */
void MergeSortInGpu(float* data, int size) {
  size_t bytes_size = size * sizeof(float);
  if (bytes_size <= sizeof(float)) {
    return;
  }

  float* dev_data = nullptr; // device上申请的数组保存排序结果
  float* dev_tmp = nullptr; // device上申请的数组保存临时排序结果
  CHECK(cudaMalloc(&dev_data, bytes_size));
  CHECK(cudaMalloc(&dev_tmp, bytes_size));
  CHECK(cudaMemcpy(dev_data, data, bytes_size, cudaMemcpyHostToDevice));

  dim3 blocks(512, 1, 1);
  dim3 grids(128, 1, 1);

  int sorted_size = 1; // 当前已经排好序的子数组长度
  while (sorted_size < size) {
    MergeSortSub<<<grids, blocks>>>(dev_data, dev_tmp, sorted_size, size);
    CHECK(cudaMemcpy(dev_data, dev_tmp, bytes_size, cudaMemcpyDeviceToDevice));
    sorted_size *= 2;
  }

  CHECK(cudaMemcpy(data, dev_data, bytes_size, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(dev_tmp));
  CHECK(cudaFree(dev_data));
  cudaDeviceSynchronize();
}

/* Initialize data by random. */
void InitialData(float* arr, size_t size) {
  time_t t;
  srand((unsigned) time(&t));
  for (int i = 0; i < size; i++) {
    arr[i] = (float) (rand() & 0xffff) / 1000.0f;
  }
}

void DumpData(float* arr, size_t size) {
  for (int i = 0; i < size; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

int main() {
  printf("starting...\n");
  InitDevice(0);

  float data[DATA_SIZE];
  InitialData(data, DATA_SIZE);
  // float data1[DATA_SIZE];
  // std::copy(data, data + DATA_SIZE, data1);

  std::cout << "Before sort: ";
  DumpData(data, DATA_SIZE);

  double gpu_start = CpuSecond();
  MergeSortInGpu(data, DATA_SIZE);
  double gpu_elapsed = CpuSecond() - gpu_start;

  std::cout << "After sort: ";
  DumpData(data, DATA_SIZE);
  std::cout << "Merge sort in gpu used " << gpu_elapsed << " s" << std::endl;

  cudaDeviceReset();
  return 0;
}
