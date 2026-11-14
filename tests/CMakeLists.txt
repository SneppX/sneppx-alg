file(GLOB_RECURSE UNIT_TEST_SRCS ../tests/unit/*.c)
file(GLOB_RECURSE INTEGRATION_TEST_SRCS ../tests/integration/*.c)
file(GLOB_RECURSE SECURITY_TEST_SRCS ../tests/security/*.c)

# Tests excluded because they reference APIs that were never implemented
set(EXCLUDED_TESTS
    test_secure_allocator
    test_firewall_engine
)

# --- GoogleTest discovery -------------------------------------------------
# Prefer an installed GTest; fall back to FetchContent. The C test sources
# are migrated to the GoogleTest framework (see tests/test_gtest.h) and are
# only compiled when SNEPPX_USE_GTEST is ON, so the rest of the build is
# unaffected when GTest cannot be obtained.
if(SNEPPX_USE_GTEST)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE
        )
        if(MSVC)
            set(gtest_force_shared_crt ON CACHE BOOL "Force gtest to use the shared C runtime" FORCE)
        endif()
        FetchContent_MakeAvailable(googletest)
    endif()
    if(GTest_FOUND)
        set(SNEPPX_GTEST_LIBS GTest::gtest_main GTest::gtest)
    else()
        set(SNEPPX_GTEST_LIBS gtest_main gtest)
    endif()
endif()

macro(add_neural_test test_src link_libs)
    get_filename_component(test_name ${test_src} NAME_WE)
    list(FIND EXCLUDED_TESTS ${test_name} _excluded)
    if(NOT _excluded EQUAL -1)
        message(STATUS "Skipping excluded test: ${test_name}")
    else()
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE ${link_libs})
    if(SNEPPX_USE_GTEST)
        # The migrated test sources are C files that now use GoogleTest;
        # compile them as C++ (MSVC needs /TP because cl decides the
        # language from the .c extension).
        set_source_files_properties(${test_src} PROPERTIES LANGUAGE CXX)
        if(MSVC)
            set_source_files_properties(${test_src} PROPERTIES COMPILE_OPTIONS "/TP")
        endif()
        target_link_libraries(${test_name} PRIVATE ${SNEPPX_GTEST_LIBS})
    endif()
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
        ${CMAKE_SOURCE_DIR}/include/neural_core/security
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/lib/internal
        ${CMAKE_SOURCE_DIR}/drivers/cuda
        ${CMAKE_SOURCE_DIR}/drivers/tpu
        ${CMAKE_SOURCE_DIR}/drivers/rocm
        ${CMAKE_SOURCE_DIR}/fs/format
        ${CMAKE_SOURCE_DIR}/security/monitor
        ${CMAKE_SOURCE_DIR}/security/memory
        ${CMAKE_SOURCE_DIR}/kernel
        ${CMAKE_SOURCE_DIR}/security/crypto/c
        ${CMAKE_SOURCE_DIR}/security/network
        ${CMAKE_SOURCE_DIR}/security/updates
        ${CMAKE_SOURCE_DIR}/security/formal
        ${CMAKE_SOURCE_DIR}/security/pentest
        ${CMAKE_SOURCE_DIR}/security/ai
        ${CMAKE_SOURCE_DIR}/security/ui
        ${CMAKE_SOURCE_DIR}/drivers/vulkan
        ${CMAKE_SOURCE_DIR}/drivers/tpu
        ${CMAKE_SOURCE_DIR}/drivers/http
        ${CMAKE_SOURCE_DIR}/drivers/zk
        ${CMAKE_SOURCE_DIR}/drivers/amd
        ${CMAKE_SOURCE_DIR}/drivers/intel
        ${CMAKE_SOURCE_DIR}/drivers/qualcomm
        ${CMAKE_SOURCE_DIR}/drivers/npu
        ${CMAKE_SOURCE_DIR}/drivers/sgx
        ${CMAKE_SOURCE_DIR}/drivers/shim
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endmacro()

if(SNEPPX_USE_GTEST)
    foreach(test_src ${UNIT_TEST_SRCS})
        add_neural_test(${test_src} "neural_core_kernel;neural_architecture_layer")
    endforeach()

    foreach(test_src ${INTEGRATION_TEST_SRCS})
        add_neural_test(${test_src} "neural_core_kernel;neural_architecture_layer")
    endforeach()

    # test_vizmon exercises the separate sneppx_vizmon dashboard library.
    if(TARGET sneppx_vizmon)
        target_link_libraries(test_vizmon PRIVATE sneppx_vizmon)
        if(MSVC AND SNEPPX_BUILD_VIZMON)
            add_custom_command(TARGET test_vizmon POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:sneppx_vizmon>
                $<TARGET_FILE_DIR:test_vizmon>
            )
        endif()
    endif()

    foreach(test_src ${SECURITY_TEST_SRCS})
        add_neural_test(${test_src} "neural_security_c;neural_core_kernel;neural_architecture_layer")
    endforeach()
else()
    message(WARNING "SNEPPX_USE_GTEST=OFF: C test executables are not built. "
                    "Set SNEPPX_USE_GTEST=ON to build the GoogleTest suite.")
endif()

# The firewall policy engine has a standalone C test so it runs even when
# the optional GoogleTest migration suite is disabled.
if(TARGET neural_security_c)
    add_executable(test_firewall_engine_native ../tests/security/test_firewall_engine.c)
    target_link_libraries(test_firewall_engine_native PRIVATE neural_security_c)
    target_include_directories(test_firewall_engine_native PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(test_firewall_engine_native PRIVATE c_std_11)
    add_test(NAME firewall_engine_native COMMAND test_firewall_engine_native)
endif()

file(GLOB_RECURSE SECURITY_CPP_TEST_SRCS ../tests/security/cpp/*.cpp)

macro(add_neural_cpp_test test_src link_libs)
    get_filename_component(test_name ${test_src} NAME_WE)
    list(FIND EXCLUDED_TESTS ${test_name} _excluded)
    if(NOT _excluded EQUAL -1)
        message(STATUS "Skipping excluded test: ${test_name}")
    else()
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} ${link_libs})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
        ${CMAKE_SOURCE_DIR}/include/neural_core/security
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/security/obfuscation
        ${CMAKE_SOURCE_DIR}/security/cpp/include
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endmacro()

foreach(test_src ${SECURITY_CPP_TEST_SRCS})
    add_neural_cpp_test(${test_src} "neural_security_cpp;neural_security_c;neural_core_kernel;neural_architecture_layer")
endforeach()

# Python tests are exercised by the dedicated `pytest tests/python` step in
# CI (which sets up PYTHONPATH + the required pure-Python deps correctly).
# They are intentionally NOT registered as ctest entries here: the previous
# ctest registration pointed PYTHONPATH at a non-existent build dir and ran
# the files with plain `python` instead of pytest, so every python test
# failed with ModuleNotFoundError before any assertion could run.
