cmake_minimum_required(VERSION 3.14)
project(mnn_htp_qurt_simulator_test CXX)

if(NOT DEFINED HEXAGON_SDK_ROOT)
    message(FATAL_ERROR "HEXAGON_SDK_ROOT is required")
endif()
if(NOT DEFINED MNN_HTPOPS_SKEL OR NOT EXISTS "${MNN_HTPOPS_SKEL}")
    message(FATAL_ERROR "MNN_HTPOPS_SKEL must point to a mock libMNN_htpops_skel.so")
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

add_library(mnn_htp_qurt_runner SHARED skel_smoke_runner.cpp)
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
