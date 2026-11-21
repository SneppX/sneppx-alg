# HSS algorithm CUDA extensions — compiled directly into the architecture layer.

file(GLOB SNEPPX_HSS_CUDA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
target_sources(neural_architecture_layer PRIVATE ${SNEPPX_HSS_CUDA_SRC})
target_include_directories(neural_architecture_layer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(neural_architecture_layer PRIVATE CUDA::cudart CUDA::cufft)
