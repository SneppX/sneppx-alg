cmake_minimum_required(VERSION 3.16)

# Core CUDA Kernel Library
file(GLOB CUDA_TENSOR_SRC
    tensor_cuda.cu
    tensor_cuda.h
    attention_cuda.cu
    attention_cuda.h
    autodiff_cuda.cu
    autodiff_cuda.h
    optim_cuda.cu
    optim_cuda.h
    memory_cuda.cu
    memory_cuda.h
    rng_cuda.cu
    rng_cuda.h
    common.cuh
)

add_library(neural_cuda_kernels STATIC ${CUDA_TENSOR_SRC})

target_include_directories(neural_cuda_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/kernel>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/architecture>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/neural_core/security>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/neural_core/kernel>
    $<INSTALL_INTERFACE:include/neural_core/architecture>
    $<INSTALL_INTERFACE:include/neural_core/security>
)

target_link_libraries(neural_cuda_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

target_compile_definitions(neural_cuda_kernels PRIVATE SNEPPX_HAS_CUDA)

# CUDA-specific compile options
set_target_properties(neural_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Enable PTXAS verbose for debugging register usage
target_compile_options(neural_cuda_kernels PRIVATE
    $<$<CXX_COMPILER_ID:NVIDIA>:-Xptxas=-v>
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neural_core/cuda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Install library
install(TARGETS neural_cuda_kernels
    EXPORT SNEPPXTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)