if(SNEPPX_BUILD_TESTS)
    add_executable(sample_basic sample_basic.c)
    target_link_libraries(sample_basic PRIVATE neural_core_kernel neural_architecture_layer neural_security_c)
endif()
