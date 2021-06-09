cmake_minimum_required(VERSION 3.19)

project(cuda_programming CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(matrix_sum matrix_sum.cu cuda_start.h)

set_target_properties(
        matrix_sum
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(matrix_inner_product matrix_inner_product.cu cuda_start.h)

set_target_properties(
        matrix_inner_product
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)