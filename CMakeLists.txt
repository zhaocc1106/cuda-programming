cmake_minimum_required(VERSION 3.19)

project(cuda_programming CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_programming matrix_sum.cu cuda_start.h)

set_target_properties(
        cuda_programming
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)