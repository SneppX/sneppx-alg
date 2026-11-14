# Object pool is a small standalone C subsystem and must be tested even
# when the optional GoogleTest migration suite is disabled.
add_executable(test_object_pool_native
    ../tests/unit/memory/test_object_pool.c
    ../mm/pool/object_pool.c
)
target_include_directories(test_object_pool_native PRIVATE
    ${CMAKE_SOURCE_DIR}/mm/pool
)
target_compile_features(test_object_pool_native PRIVATE c_std_11)
add_test(NAME object_pool_native COMMAND test_object_pool_native)

