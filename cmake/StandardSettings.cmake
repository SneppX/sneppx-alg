# CMake module: Standard project settings
#
# Centralizes common CMake configuration for all targets.
#
# Usage:
#   include(StandardSettings)
#   apply_standard_settings(my_target)

function(apply_standard_settings TARGET)
    # C standard
    set_target_properties(${TARGET} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )

    # Warnings
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${TARGET} PRIVATE
            -Wall -Wextra -Wpedantic
            -Wshadow -Wformat=2 -Wconversion
            -Wstrict-prototypes -Wmissing-prototypes
            $<$<CONFIG:Debug>:-g -O0>
            $<$<CONFIG:Release>:-O3 -DNDEBUG>
        )
    elseif(MSVC)
        target_compile_options(${TARGET} PRIVATE
            /W4 /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS
            $<$<CONFIG:Debug>:/Zi /RTC1>
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
        )
    endif()

    # Position independent code
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Visibility
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${TARGET} PRIVATE -fvisibility=hidden)
    endif()
endfunction()
