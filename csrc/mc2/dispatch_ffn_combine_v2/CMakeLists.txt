cmake_minimum_required(VERSION 3.16)
project(dispatch_ffn_combine_v2)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
set(CMAKE_C_LINK_DEPENDS_USE_LINKER FALSE)
set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
set(CATLASS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/catlass/include)
set(SOC_VERSION "ascend910_93" CACHE STRING "Ascend SoC version")
set(DISPATCH_FFN_COMBINE_REQUESTED_SOC_VERSION "${SOC_VERSION}")
set(DISPATCH_FFN_COMBINE_TILING_KEY "1000010" CACHE STRING "Standalone dispatch_ffn_combine tiling key")

if(SOC_VERSION STREQUAL "ascend910_93")
    set(ASCENDC_SOC_VERSION "ascend910_9391")
elseif(SOC_VERSION STREQUAL "ascend910_9391" OR
       SOC_VERSION STREQUAL "ascend910_9381" OR
       SOC_VERSION STREQUAL "ascend910_9372" OR
       SOC_VERSION STREQUAL "ascend910_9392" OR
       SOC_VERSION STREQUAL "ascend910_9382" OR
       SOC_VERSION STREQUAL "ascend910_9362")
    set(ASCENDC_SOC_VERSION "${SOC_VERSION}")
else()
    message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}")
endif()
set(SOC_VERSION ${ASCENDC_SOC_VERSION} CACHE STRING "AscendC exact SoC version" FORCE)
set(RUN_MODE "npu" CACHE STRING "Run mode")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check CANN installation.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

ascendc_library(dispatch_ffn_combine_v2_kernel SHARED
    op_kernel/dispatch_ffn_combine.cpp
)

ascendc_include_directories(dispatch_ffn_combine_v2_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/unpermute
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/moe_init_routing_quant_v2
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl
)

ascendc_compile_options(dispatch_ffn_combine_v2_kernel PRIVATE
    -I${CATLASS_INCLUDE}
    -UTILING_KEY_VAR
    -DTILING_KEY_VAR=${DISPATCH_FFN_COMBINE_TILING_KEY}
)

ascendc_compile_definitions(dispatch_ffn_combine_v2_kernel PRIVATE
    HCCL_COMM
    CATLASS_ARCH=2201
    DTYPE_W1=int8_t
    DTYPE_OUT=half
)

ascendc_compile_options(dispatch_ffn_combine_v2_kernel PRIVATE
    -Wno-ignored-attributes
    -forward-options-to-host-compiler
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

add_custom_command(TARGET dispatch_ffn_combine_v2_kernel PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/dispatch_ffn_combine_v2_kernel_host_dir/objects
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

add_executable(dispatch_ffn_combine_v2
    main.cpp
    runtime_context.cpp
    tiling_builder.cpp
    data_utils.cpp
)

target_compile_options(dispatch_ffn_combine_v2 PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_compile_definitions(dispatch_ffn_combine_v2 PRIVATE
    DISPATCH_FFN_COMBINE_V2_REQUESTED_SOC_VERSION="${DISPATCH_FFN_COMBINE_REQUESTED_SOC_VERSION}"
    DISPATCH_FFN_COMBINE_V2_COMPILE_SOC_VERSION="${SOC_VERSION}"
)

target_include_directories(dispatch_ffn_combine_v2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/moe_init_routing_quant_v2
    ${CATLASS_INCLUDE}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api
    ${ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
)

target_link_directories(dispatch_ffn_combine_v2 PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/aarch64-linux/lib64
)

target_link_libraries(dispatch_ffn_combine_v2 PRIVATE
    host_intf_pub
    dispatch_ffn_combine_v2_kernel
    ascendcl hccl hccl_fwk hcomm runtime nnopbase tiling_api platform c_sec dl pthread stdc++ m
)
