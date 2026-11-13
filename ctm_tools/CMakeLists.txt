cmake_minimum_required(VERSION 3.16)
project(ctm_tools LANGUAGES NONE)

# The historical C implementations are incomplete prototypes: several
# platforms return placeholder values instead of performing a real scan.
# Do not build or distribute those executables until the C implementations
# have a shared, tested cross-platform filesystem walker.
message(STATUS "CTM: the supported implementation is the Python standard-library scanner.")
message(STATUS "CTM: legacy C prototypes are intentionally not exposed as build targets.")

add_custom_target(ctm-help
    COMMAND ${CMAKE_COMMAND} -E echo "Run: python ${CMAKE_CURRENT_SOURCE_DIR}/py/ctm.py ."
    VERBATIM
)
