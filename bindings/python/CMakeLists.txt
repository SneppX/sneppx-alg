# =============================================================================
# SNEPPX Algo – Python Bindings (pybind11)
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# Locate pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Locate Python 3
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development NumPy)

# Build Python extension module (links pre-built static libs instead of
# recompiling all C sources)
pybind11_add_module(_SNEPPX_c MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp"
)

target_include_directories(_SNEPPX_c PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/security
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel/autodiff
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel/optimizer
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel/train
    ${CMAKE_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS}
)

target_compile_definitions(_SNEPPX_c PRIVATE
    $<$<CONFIG:Release>:SNEPPX_S1_SECURE_MEMORY>
    _CRT_SECURE_NO_WARNINGS
)

# Link pre-built SNEPPX static libraries from the parent build
target_link_libraries(_SNEPPX_c PRIVATE
    neural_architecture_layer
    neural_core_kernel
    neural_security_c
    pybind11::headers
    Python::Python
)

if(WIN32)
    target_compile_definitions(_SNEPPX_c PRIVATE
        SNEPPX_CRYPTO_SECURE
        NOMINMAX
    )
endif()

# Platform libraries
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(_SNEPPX_c PRIVATE rt pthread)
elseif(APPLE)
    target_link_libraries(_SNEPPX_c PRIVATE pthread)
endif()

# Install the compiled module next to the Python package
install(TARGETS _SNEPPX_c
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/SneppX_ALG"
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/SneppX_ALG"
)

# Copy the .pyd next to the package source for development imports
add_custom_command(TARGET _SNEPPX_c POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:_SNEPPX_c>"
        "${CMAKE_CURRENT_SOURCE_DIR}/SneppX_ALG/"
    COMMENT "Copying compiled extension to SneppX_ALG/ package directory"
)
