file(GLOB BENCH_SRCS *.c)

macro(add_neural_bench bench_src)
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} neural_core_kernel neural_architecture_layer neural_security_c)
    target_include_directories(${bench_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
        ${CMAKE_SOURCE_DIR}/include/neural_core/security
        ${CMAKE_SOURCE_DIR}/tests
    )
    if(SNEPPX_USE_LTO)
        SNEPPX_enable_lto(${bench_name})
    endif()
endmacro()

foreach(bench_src ${BENCH_SRCS})
    add_neural_bench(${bench_src})
endforeach()
