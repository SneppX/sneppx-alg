# NCCL based distributed communication backend.
# NCCL is loaded dynamically at runtime (see nccl.c), so the build does not
# require NCCL headers or libraries to be present. The source is compiled
# directly into the core kernel library.

target_sources(neural_core_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nccl.c)
target_include_directories(neural_core_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(neural_core_kernel PRIVATE CUDA::cudart)
target_compile_definitions(neural_core_kernel PRIVATE SNEPPX_HAS_CUDA)
