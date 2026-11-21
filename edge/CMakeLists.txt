cmake_minimum_required(VERSION 3.16)
project(SNEPPX_EdgeRuntime VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)

include_directories(include)

add_library(edge_runtime src/edge_runtime.c)
target_link_libraries(edge_runtime m)
