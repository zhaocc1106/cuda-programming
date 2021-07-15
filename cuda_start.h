#ifndef CUDASTART_H
#define CUDASTART_H

#include <ctime>
#include <sys/time.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

double cpuSecond() {
  struct timeval tp{};
  gettimeofday(&tp, nullptr);
  return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);

}

void initDevice(int devNum) {
  int dev = devNum;
  cudaDeviceProp deviceProp{};
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using device %d, name: %s, warpSize: %d, concurrentKernels: %d, totalConstMem: %zu, totalGlobalMem:"
         " %zu,\nmaxBlocksPerMultiProcessor: %d, maxThreadsPerMultiProcessor: %d, maxThreadsPerBlock %d, "
         "globalL1CacheSupported: %d, localL1CacheSupported: %d,\nl2CacheSize: %d, persistingL2CacheMaxSize: %d, "
         "accessPolicyMaxWindowSize: %d, concurrentKernels: %d, asyncEngineCount: %d, unifiedAddressing: %d.\n",
         dev,
         deviceProp.name,
         deviceProp.warpSize,
         deviceProp.concurrentKernels,
         deviceProp.totalConstMem,
         deviceProp.totalGlobalMem,
         deviceProp.maxBlocksPerMultiProcessor,
         deviceProp.maxThreadsPerMultiProcessor,
         deviceProp.maxThreadsPerBlock,
         deviceProp.globalL1CacheSupported,
         deviceProp.localL1CacheSupported,
         deviceProp.l2CacheSize,
         deviceProp.persistingL2CacheMaxSize,
         deviceProp.accessPolicyMaxWindowSize,
         deviceProp.concurrentKernels,
         deviceProp.asyncEngineCount,
         deviceProp.unifiedAddressing
  );
  CHECK(cudaSetDevice(dev));
}

#endif