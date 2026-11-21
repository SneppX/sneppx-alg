# CompilerOptions.cmake
# Shared compiler option helpers for the SneppX_ALG project.

include_guard(GLOBAL)

# Enable LTO (Link-Time Optimization) based on compiler
macro(SNEPPX_enable_lto target)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE -flto)
        target_link_options(${target} PRIVATE -flto)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /GL)
        set_target_properties(${target} PROPERTIES
            STATIC_LIBRARY_FLAGS /LTCG
            INTERFACE_LINK_LIBRARIES ""
        )
        if(NOT ${target} MATCHES ".*_static")
            target_link_options(${target} PRIVATE /LTCG)
        endif()
    endif()
endmacro()

# Enable architecture-specific tuning
macro(SNEPPX_enable_arch_tuning target)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target} PRIVATE -march=native -maes)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
            target_compile_options(${target} PRIVATE -march=armv8-a+crypto)
        endif()
    endif()
endmacro()
