# cuda-programming
Learning cuda programming. Learning NVIDIA gpu programming.

## 源文件简介
* [matrix_sum.cu](https://github.com/zhaocc1106/cuda-programming/blob/master/matrix_sum.cu): Cuda hello world，实现一个矩阵加法，并比较和cpu实现的耗时，发现矩阵较小时cpu可能快些，但较大后gpu快很多。
* [matrix_inner_product.cu](https://github.com/zhaocc1106/cuda-programming/blob/master/matrix_inner_product.cu): cuda实现矩阵内积算法，并尝试比较使用了共享内存和不使用共享内存两种方式，比较耗时发现使用共享内存延时低很多。
* [async_stream_test.cu](https://github.com/zhaocc1106/cuda-programming/blob/master/async_stream_test.cu): 尝试使用stream方式实现多个stream并行执行，比较和串行执行的耗时，发现耗时会降低很多。
* [cuda_graph_test.cu](https://github.com/zhaocc1106/cuda-programming/blob/master/cuda_graph_test.cu): 尝试cuda graph构建，实例化和运行，并和不通过graph直接运行的stream比较耗时。发现有graph基本上比没有graph要快一些。
* [merge_sort.cu](https://github.com/zhaocc1106/cuda-programming/blob/master/merge_sort.cu): cuda实现merge sort，并比较和cpu版并发merge sort耗时。
