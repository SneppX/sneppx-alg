# CMake module: Static analyzers
#
# Integrates clang-tidy, cppcheck, and include-what-you-use.
#
# Usage:
#   include(StaticAnalyzers)
#   enable_clang_tidy(target OPTIONS)
#   enable_cppcheck(target OPTIONS)

find_program(CLANG_TIDY clang-tidy)
find_program(CPPCHECK cppcheck)
find_program(IWYU include-what-you-use)

option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)
option(ENABLE_CPPCHECK "Enable cppcheck static analysis" OFF)
option(ENABLE_IWYU "Enable include-what-you-use" OFF)

function(enable_clang_tidy TARGET)
    if(NOT CLANG_TIDY OR NOT ENABLE_CLANG_TIDY)
        return()
    endif()
    set(CLANG_TIDY_OPTIONS ${ARGN})
    if(NOT CLANG_TIDY_OPTIONS)
        set(CLANG_TIDY_OPTIONS "-checks=*,-clang-analyzer-*,-llvmlibc-*,-modernize-use-trailing-return-type" "-warnings-as-errors=*")
    endif()
    set_target_properties(${TARGET} PROPERTIES C_CLANG_TIDY "${CLANG_TIDY};${CLANG_TIDY_OPTIONS}")
endfunction()

function(enable_cppcheck TARGET)
    if(NOT CPPCHECK OR NOT ENABLE_CPPCHECK)
        return()
    endif()
    set(CPPCHECK_OPTIONS ${ARGN})
    if(NOT CPPCHECK_OPTIONS)
        set(CPPCHECK_OPTIONS "--enable=warning,performance,portability" "--suppress=missingIncludeSystem" "--language=c" "--std=c11")
    endif()
    set_target_properties(${TARGET} PROPERTIES C_CPPCHECK "${CPPCHECK};${CPPCHECK_OPTIONS}")
endfunction()

function(enable_iwyu TARGET)
    if(NOT IWYU OR NOT ENABLE_IWYU)
        return()
    endif()
    set_target_properties(${TARGET} PROPERTIES C_INCLUDE_WHAT_YOU_USE "${IWYU}")
endfunction()
