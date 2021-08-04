cmake_minimum_required(VERSION 3.19)

project(cuda_programming CUDA C CXX)

message(STATUS "CMAKE_C_FLAGS_DEBUG: " ${CMAKE_C_FLAGS_DEBUG})
message(STATUS "CMAKE_C_FLAGS_RELEASE: " ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "CMAKE_C_FLAGS_RELWITHDEBINFO: " ${CMAKE_C_FLAGS_RELWITHDEBINFO})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_C_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: " ${CMAKE_C_FLAGS_RELWITHDEBINFO})

message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

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

add_executable(async_stream_test async_stream_test.cu cuda_start.h)

set_target_properties(
        async_stream_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(cuda_graph_test cuda_graph_test.cu cuda_start.h)

set_target_properties(
        cuda_graph_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)