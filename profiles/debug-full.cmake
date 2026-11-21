# Debug-full build profile
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
set(SNEPPX_BUILD_TESTS ON CACHE BOOL "Build tests")
set(SNEPPX_BUILD_BENCHMARKS ON CACHE BOOL "Build benchmarks")
set(SNEPPX_USE_ASAN ON CACHE BOOL "Enable ASAN")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer -fstack-protector-strong" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer -fstack-protector-strong" CACHE STRING "")
endif()
