cmake_minimum_required(VERSION 3.14)
project(mnn_htp_qurt_simulator_test CXX)

option(MNN_OFFLINE_RPC_HOST_ONLY "Build only the native offline RPC request tool" OFF)
set(MNN_OFFLINE_RPC_MNN_BUILD_DIR "" CACHE PATH "Optional native MNN build for the MatMul graph recorder")

if(MNN_OFFLINE_RPC_HOST_ONLY)
    add_executable(mnn_htp_offline_rpc_host offline_rpc_host.cpp offline_rpc_protocol.h)
    target_compile_features(mnn_htp_offline_rpc_host PRIVATE cxx_std_11)
    target_include_directories(mnn_htp_offline_rpc_host PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../3rd_party/flatbuffers/include)
    if(MNN_OFFLINE_RPC_MNN_BUILD_DIR)
        add_executable(mnn_htp_mnn_matmul_record mnn_matmul_record.cpp)
        add_executable(mnn_htp_linear_attention_record linear_attention_record.cpp)
        add_executable(mnn_htp_mnn_variable_dump mnn_variable_dump.cpp)
        target_compile_features(mnn_htp_mnn_matmul_record PRIVATE cxx_std_11)
        target_compile_features(mnn_htp_linear_attention_record PRIVATE cxx_std_11)
        target_compile_features(mnn_htp_mnn_variable_dump PRIVATE cxx_std_11)
        target_include_directories(mnn_htp_mnn_matmul_record PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../include)
        target_include_directories(mnn_htp_linear_attention_record PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../source
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../schema/current
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../3rd_party/half
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../3rd_party/flatbuffers/include)
        target_include_directories(mnn_htp_mnn_variable_dump PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../3rd_party/flatbuffers/include)
        target_link_libraries(mnn_htp_mnn_matmul_record PRIVATE
            ${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/libMNN.so
            ${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/express/libMNN_Express.so)
        target_link_libraries(mnn_htp_linear_attention_record PRIVATE
            ${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/libMNN.so
            ${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/express/libMNN_Express.so)
        target_link_libraries(mnn_htp_mnn_variable_dump PRIVATE
            ${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/libMNN.so
            ${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/express/libMNN_Express.so)
        set_target_properties(mnn_htp_mnn_matmul_record PROPERTIES
            BUILD_RPATH "${MNN_OFFLINE_RPC_MNN_BUILD_DIR};${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/express")
        set_target_properties(mnn_htp_linear_attention_record PROPERTIES
            BUILD_RPATH "${MNN_OFFLINE_RPC_MNN_BUILD_DIR};${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/express")
        set_target_properties(mnn_htp_mnn_variable_dump PROPERTIES
            BUILD_RPATH "${MNN_OFFLINE_RPC_MNN_BUILD_DIR};${MNN_OFFLINE_RPC_MNN_BUILD_DIR}/express")
    endif()
    return()
endif()

if(NOT DEFINED HEXAGON_SDK_ROOT)
    message(FATAL_ERROR "HEXAGON_SDK_ROOT is required")
endif()
if(NOT DEFINED MNN_HTPOPS_SKEL OR NOT EXISTS "${MNN_HTPOPS_SKEL}")
    message(FATAL_ERROR "MNN_HTPOPS_SKEL must point to a mock libMNN_htpops_skel.so")
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

add_library(mnn_htp_qurt_runner SHARED skel_smoke_runner.cpp offline_rpc_protocol.h)
target_link_libraries(mnn_htp_qurt_runner ${MNN_HTPOPS_SKEL})
target_compile_options(mnn_htp_qurt_runner PRIVATE -O2)
set_target_properties(mnn_htp_qurt_runner PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# QuRT's runtime loader searches the simulator working directory for DT_NEEDED
# libraries. Keep the skel SONAME and the runner in that same directory.
configure_file(${MNN_HTPOPS_SKEL}
    ${CMAKE_CURRENT_BINARY_DIR}/libMNN_htpops_skel.so COPYONLY)

set(HEXAGON_EXEC_SIM_OPTIONS --mhmx=3)
runMainOnHexagonSim(mnn_htp_qurt_runner)

# runMainOnHexagonSim stages C++ runtimes under ship/, while QuRT's loader
# searches the simulator working directory for this runner's dependencies.
configure_file(${CMAKE_CURRENT_BINARY_DIR}/ship/libc++.so.1.0
    ${CMAKE_CURRENT_BINARY_DIR}/libc++.so.1 COPYONLY)
configure_file(${CMAKE_CURRENT_BINARY_DIR}/ship/libc++abi.so.1.0
    ${CMAKE_CURRENT_BINARY_DIR}/libc++abi.so.1 COPYONLY)
