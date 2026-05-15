import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]
V2_DIR = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_v2"


def read_text(relative_path: str) -> str:
    return (V2_DIR / relative_path).read_text(encoding="utf-8")


def assert_header_member_defs_are_inline(header_text: str) -> None:
    for line in header_text.splitlines():
        if "::" in line and "(" in line and line.strip().endswith("{"):
            if line.lstrip().startswith(("inline ", "__aicore__ inline ", "static inline ")):
                continue
            raise AssertionError(f"non-inline header member definition: {line}")


def test_v2_runtime_and_tiling_sources_exist_with_expected_symbols():
    runtime_h = read_text("runtime_context.hpp")
    runtime_cpp = read_text("runtime_context.cpp")
    tiling_h = read_text("tiling_builder.hpp")
    tiling_cpp = read_text("tiling_builder.cpp")
    kernel_tiling_h = read_text("op_kernel/dispatch_ffn_combine_tiling.h")
    routing_v2_tiling_h = read_text("op_kernel/moe_init_routing_quant_v2/moe_init_routing_v2_tiling.h")
    routing_quant_v2_tiling_h = read_text("op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h")
    data_h = read_text("data_utils.hpp")
    data_cpp = read_text("data_utils.cpp")

    assert "struct StandaloneRankRuntime" in runtime_h
    assert "InitStandaloneRankRuntime" in runtime_h
    assert "HcclAllocComResourceByTiling" in runtime_h
    assert "BuildWindowTable" in runtime_cpp
    assert "window_table_dev" in runtime_cpp

    assert "struct CaseConfig" in tiling_h
    assert "BuildDispatchFFNCombineTiling" in tiling_h
    assert "runtime.window_table_dev" in tiling_cpp
    assert "launchConfig.tilingKey = 1000010" in tiling_cpp

    assert '"hccl/hccl_tilingdata.h"' not in kernel_tiling_h
    assert '"kernel_tiling.h"' in kernel_tiling_h
    assert "Mc2InitTiling mc2InitTiling;" in kernel_tiling_h
    assert "Mc2CcTiling mc2CcTiling;" in kernel_tiling_h
    assert "#include <cmath>" in routing_v2_tiling_h
    assert "#include <algorithm>" in routing_v2_tiling_h
    assert "#include <algorithm>" in routing_quant_v2_tiling_h
    assert "inline void SetGatherTilingDatawithloop(" in routing_quant_v2_tiling_h
    assert_header_member_defs_are_inline(routing_v2_tiling_h)
    assert_header_member_defs_are_inline(routing_quant_v2_tiling_h)

    kernel_cpp = read_text("op_kernel/dispatch_ffn_combine.cpp")
    kernel_h = read_text("op_kernel/dispatch_ffn_combine.h")
    shmem_h = read_text("op_kernel/utils/hccl_shmem.hpp")
    assert "GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingData, tilingData, tilingGM);" not in kernel_cpp
    assert "GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingData, tilingData, tilingGM);" not in kernel_h
    assert "GET_TILING_DATA(tilingData, tilingGM);" not in kernel_h
    assert "#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)" in kernel_cpp
    assert '#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)\n#include "lib/matmul_intf.h"\n#include "dispatch_ffn_combine_tiling.h"\n#include "dispatch_ffn_combine.h"' in kernel_cpp
    assert "#if defined(__CCE_KT_TEST__) || !defined(__CCE_AICORE__)" in kernel_cpp
    assert "dispatch_ffn_combine<<<args.block_dim, nullptr, stream>>>" in kernel_cpp
    assert "reinterpret_cast<__gm__ DispatchFFNCombineTilingData *>(tilingGM)" in kernel_cpp
    assert "const __gm__ DispatchFFNCombineTilingData *tilingData" in kernel_h
    assert "tilingData->dispatchFFNCombineInfo.M" in kernel_h
    assert "reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.symmetricPtr)" in kernel_h
    assert "moeInitRoutingQuantV2TilingData = tilingData->cocTiling.moeInitRoutingQuantV2TilingData;" not in kernel_h
    assert "moeInitRoutingQuantV2TilingData.vbsComputeParamsOp =" not in kernel_h
    assert "moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp =" not in kernel_h
    assert "moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp =" not in kernel_h
    assert "moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp =" not in kernel_h
    assert "moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp =" not in kernel_h
    assert "moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp =" not in kernel_h
    assert "moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.needCoreNum =" in kernel_h
    assert "moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCoreElements =" in kernel_h
    assert "moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp.needCoreNum =" in kernel_h
    assert "moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp.oneLoopMaxElements =" in kernel_h
    assert "moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.needCoreNum =" in kernel_h
    assert "moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.needCoreNum =" in kernel_h
    assert "moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.needCoreNum =" in kernel_h
    assert "return reinterpret_cast<GM_ADDR>(windows[rankId]);" in shmem_h

    assert "RankFileSet" in data_h
    assert "LoadCaseConfig" in data_h
    assert "CompareBf16File" in data_cpp
    assert "Bf16ToFloat" in data_cpp


def test_v2_standalone_entrypoints_and_runner_files_exist():
    main_cpp = read_text("main.cpp")
    comm_mpi_h = read_text("comm_mpi.h")
    run_sh = read_text("run.sh")
    readme = read_text("README.md")
    cmake = read_text("CMakeLists.txt")
    gen_data = read_text("scripts/gen_data.py")

    assert "CommMpiInit" in comm_mpi_h
    assert "MPI_Init" in comm_mpi_h

    assert "launchDispatchFFNCombine" in main_cpp
    assert "ZeroWindowMemory" in main_cpp
    assert "rtSetDevice(rank_id)" in main_cpp
    assert "aclrtSetDevice(rank_id)" in main_cpp
    assert "HcclGetRootInfo" in main_cpp
    assert main_cpp.index("rtSetDevice(rank_id)") < main_cpp.index("HcclGetRootInfo(&root_info)")
    assert main_cpp.index("aclrtSetDevice(rank_id)") < main_cpp.index("HcclGetRootInfo(&root_info)")
    assert "output_rank" in main_cpp

    assert "scripts/gen_data.py" in run_sh
    assert "cmake -S" in run_sh
    assert 'set +e\nset +u\nset +o pipefail\nsource "${ASCEND_HOME_PATH}/set_env.sh"\nset -euo pipefail' in run_sh
    assert 'export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH}"' in run_sh
    assert '"${MPI_RUNNER}" -n "${WORLD_SIZE}"' in run_sh

    assert "dispatch_ffn_combine_v2" in readme
    assert "Known limits" in readme
    assert "torch" in readme and "vllm" in readme

    assert "add_executable(dispatch_ffn_combine_v2" in cmake
    assert "dispatch_ffn_combine_v2_kernel" in cmake
    assert "ascendcl" in cmake
    assert " pthread stdc++ m" in cmake or " m pthread stdc++" in cmake or " stdc++ m" in cmake

    assert "case.json" in gen_data
    assert "expected_out" in gen_data


def test_v2_cmake_delegates_dynamic_kernel_build_to_official_ascendc_pipeline():
    cmake = read_text("CMakeLists.txt")

    assert "dav-c220-mix" not in cmake
    assert "ascendc.cmake" in cmake
    assert "ascendc_library(dispatch_ffn_combine_v2_kernel" in cmake
    assert "ascendc_include_directories(dispatch_ffn_combine_v2_kernel" in cmake
    assert "${CATLASS_INCLUDE}" in cmake
    assert "ascendc_compile_definitions(dispatch_ffn_combine_v2_kernel" in cmake
    assert "HCCL_COMM" in cmake
    assert "CATLASS_ARCH=2201" in cmake
    assert "DTYPE_W1=int8_t" in cmake
    assert "DTYPE_OUT=half" in cmake
    assert "if(NOT CMAKE_BUILD_TYPE)" in cmake
    assert 'set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)' in cmake
    assert "set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)" in cmake
    assert "set(CMAKE_C_LINK_DEPENDS_USE_LINKER FALSE)" in cmake
    assert "set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)" in cmake
    assert "add_custom_command(TARGET dispatch_ffn_combine_v2_kernel PRE_LINK" in cmake
    assert "${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/dispatch_ffn_combine_v2_kernel_host_dir/objects" in cmake
    assert "ascendc_compile_options(dispatch_ffn_combine_v2_kernel PRIVATE\n    -Wno-ignored-attributes\n    -forward-options-to-host-compiler\n    \"SHELL:-include stdint.h\"\n    \"SHELL:-include stddef.h\"\n)" in cmake
