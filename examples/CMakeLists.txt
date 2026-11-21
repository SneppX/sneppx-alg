add_executable(hss_demo hss_demo.c)
target_link_libraries(hss_demo neural_core_kernel neural_architecture_layer)
target_include_directories(hss_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
)

add_executable(ser_demo ser_demo.c)
target_link_libraries(ser_demo neural_core_kernel neural_architecture_layer)
target_include_directories(ser_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
)

add_executable(arc_demo arc_demo.c)
target_link_libraries(arc_demo neural_core_kernel neural_architecture_layer)
target_include_directories(arc_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
)

add_executable(npe_demo npe_demo.c)
target_link_libraries(npe_demo neural_core_kernel neural_architecture_layer)
target_include_directories(npe_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
)

add_executable(fm_demo fm_demo.c)
target_link_libraries(fm_demo neural_core_kernel neural_architecture_layer)
target_include_directories(fm_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
)

add_executable(security_stress security_stress.c)
target_link_libraries(security_stress neural_security_c neural_core_kernel neural_architecture_layer)
target_include_directories(security_stress PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/security
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
)

add_executable(obf_demo obf_demo.cpp)
target_link_libraries(obf_demo neural_security_cpp neural_security_c neural_core_kernel neural_architecture_layer)
target_include_directories(obf_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include/neural_core/security
    ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
    ${CMAKE_SOURCE_DIR}/include/neural_core/architecture
)

if(SNEPPX_BUILD_HTTP)
    add_executable(http_server_demo http_server_demo.c)
    target_link_libraries(http_server_demo neural_core_kernel neural_architecture_layer)
    target_include_directories(http_server_demo PRIVATE
        ${CMAKE_SOURCE_DIR}/include/neural_core/kernel
        ${CMAKE_SOURCE_DIR}/net/http
    )
endif()
