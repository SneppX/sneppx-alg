# Model Zoo - Config, Registry, Weights, Model Card, Model Factory
# Phase 1: Model Config, Registry, Weights, Model Card, Model Factory

set(MODEL_ZOO_SOURCES
    model_config.c
    registry.c
    weights.c
    model_card.c
)

# Create static libraries
add_library(neural_model_config STATIC model_config.c)
add_library(neural_model_registry STATIC registry.c)
add_library(neural_model_weights STATIC weights.c)
add_library(neural_model_card STATIC model_card.c)
add_library(neural_model_factory STATIC model_factory.cpp)

# Common include directories
target_include_directories(neural_model_config PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(neural_model_registry PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(neural_model_weights PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(neural_model_card PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(neural_model_factory PUBLIC ${CMAKE_SOURCE_DIR}/include)

# C standard (MSVC uses /std:c11; GCC/Clang already get C11 via CMAKE_C_STANDARD)
if(MSVC)
    target_compile_options(neural_model_config PRIVATE /std:c11)
    target_compile_options(neural_model_registry PRIVATE /std:c11)
    target_compile_options(neural_model_weights PRIVATE /std:c11)
    target_compile_options(neural_model_card PRIVATE /std:c11)
endif()

# Link dependencies
target_link_libraries(neural_model_config PUBLIC neural_core_kernel)
target_link_libraries(neural_model_registry PUBLIC neural_core_kernel neural_model_config)
target_link_libraries(neural_model_weights PUBLIC neural_core_kernel neural_model_config)
target_link_libraries(neural_model_card PUBLIC neural_core_kernel neural_model_config neural_model_registry)
target_link_libraries(neural_model_factory PUBLIC neural_model_config neural_model_registry neural_model_weights neural_model_card)

# Tests
if(BUILD_TESTING)
    add_executable(test_model_config tests/test_model_config.c)
    target_link_libraries(test_model_config PRIVATE neural_model_config)
    target_include_directories(test_model_config PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_model_config COMMAND test_model_config)

    add_executable(test_model_registry tests/test_model_registry.c)
    target_link_libraries(test_model_registry PRIVATE neural_model_registry neural_model_config)
    target_include_directories(test_model_registry PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_model_registry COMMAND test_model_registry)

    add_executable(test_model_weights tests/test_model_weights.c)
    target_link_libraries(test_model_weights PRIVATE neural_model_weights neural_model_config neural_model_registry)
    target_include_directories(test_model_weights PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_model_weights COMMAND test_model_weights)

    add_executable(test_model_card tests/test_model_card.c)
    target_link_libraries(test_model_card PRIVATE neural_model_card neural_model_config neural_model_registry)
    target_include_directories(test_model_card PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_model_card COMMAND test_model_card)

    add_executable(test_model_factory tests/test_model_factory.cpp)
    target_link_libraries(test_model_factory PRIVATE neural_model_factory neural_model_config neural_model_registry neural_model_weights neural_model_card)
    target_include_directories(test_model_factory PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_model_factory COMMAND test_model_factory)

    add_executable(test_integration tests/test_integration.c)
    target_link_libraries(test_integration PRIVATE neural_model_config neural_model_registry neural_model_weights neural_model_card)
    target_include_directories(test_integration PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_integration COMMAND test_integration)
endif()