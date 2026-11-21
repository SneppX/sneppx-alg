# Release-optimized build profile
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(SNEPPX_BUILD_TESTS OFF CACHE BOOL "Build tests")
set(SNEPPX_BUILD_BENCHMARKS ON CACHE BOOL "Build benchmarks")
set(SNEPPX_USE_LTO ON CACHE BOOL "Enable LTO")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -ffast-math -funroll-loops" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -ffast-math -funroll-loops" CACHE STRING "")
elseif(MSVC)
    set(CMAKE_C_FLAGS_RELEASE "/O2 /GS- /GL /Gy /DNDEBUG" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GS- /GL /Gy /DNDEBUG" CACHE STRING "")
endif()
