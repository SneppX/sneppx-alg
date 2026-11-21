# CMake module: Code coverage
#
# Adds coverage build type and coverage report targets.
#
# Requires GCC or Clang with gcov/lcov/genhtml on Linux.
#
# Usage:
#   include(CodeCoverage)
#   append_coverage_compiler_flags()
#   setup_coverage_target(my_test_target)

include(CMakeDependentOption)

option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)

if(NOT ENABLE_COVERAGE)
    return()
endif()

if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    message(WARNING "Code coverage requires GCC or Clang. Disabling.")
    set(ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
    return()
endif()

set(CMAKE_C_FLAGS_COVERAGE "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
    CACHE STRING "Flags for coverage build")
set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
    CACHE STRING "Flags for coverage build")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage"
    CACHE STRING "Linker flags for coverage build")

mark_as_advanced(CMAKE_C_FLAGS_COVERAGE CMAKE_CXX_FLAGS_COVERAGE CMAKE_EXE_LINKER_FLAGS_COVERAGE)

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "" FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;RelWithDebInfo;Coverage")

find_program(LCOV lcov)
find_program(GENHTML genhtml)

function(setup_coverage_target TARGET)
    if(NOT LCOV OR NOT GENHTML)
        message(WARNING "lcov or genhtml not found. Coverage report target not created.")
        return()
    endif()
    add_custom_target(coverage-${TARGET}
        COMMAND ${CMAKE_COMMAND} -E echo "=== Running ${TARGET} ==="
        COMMAND $<TARGET_FILE:${TARGET}>
        COMMAND ${LCOV} --directory ${CMAKE_BINARY_DIR} --capture --output-file coverage_${TARGET}.info
        COMMAND ${LCOV} --remove coverage_${TARGET}.info "/usr/*" "${CMAKE_SOURCE_DIR}/tests/*" --output-file coverage_${TARGET}_filtered.info
        COMMAND ${GENHTML} coverage_${TARGET}_filtered.info --output-directory coverage_report_${TARGET}
        COMMAND ${CMAKE_COMMAND} -E echo "=== Report: coverage_report_${TARGET}/index.html ==="
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating coverage report for ${TARGET}"
    )
endfunction()
