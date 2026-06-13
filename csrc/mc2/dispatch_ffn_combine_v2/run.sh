#!/usr/bin/env bash
set -euo pipefail

WORLD_SIZE=2
SOC=ascend910_93
M=16
K=128
N=128
TOPK=2
EXPERTS=2
MAX_OUTPUT_SIZE=32
SEED=20260515
CASE_MODE=cpu-golden
ATOL=1e-3
RTOL=1e-3
WARMUP_ITERS=${DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS:-3}
MEASURE_ITERS=${DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS:-5}
SKIP_GOLDEN=${DISPATCH_FFN_COMBINE_V2_SKIP_GOLDEN:-0}
STAGE_PROFILE=${DISPATCH_FFN_COMBINE_V2_STAGE_PROFILE:-0}
START_SYNC_DEBUG=${DISPATCH_FFN_COMBINE_V2_START_SYNC_DEBUG:-0}
SKIP_BUILD=${DISPATCH_FFN_COMBINE_V2_SKIP_BUILD:-0}
TRACE=${DISPATCH_FFN_COMBINE_V2_TRACE:-0}

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  if [[ -d /usr/local/Ascend/cann-8.5.0 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
  elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  else
    echo "ASCEND_HOME_PATH is not set and no default CANN path was found" >&2
    exit 1
  fi
fi
MPI_ENV_BIN=${MPI_ENV_BIN:-/home/ntlab/miniconda3/envs/ltr_pto/bin}
MPI_ENV_LIB=${MPI_ENV_LIB:-/home/ntlab/miniconda3/envs/ltr_pto/lib}
MPI_LIB_PATH=${MPI_LIB_PATH:-${MPI_ENV_LIB}/libmpi.so}
MPI_RUNNER=${MPI_RUNNER:-mpirun}

set +e
set +u
set +o pipefail
source "${ASCEND_HOME_PATH}/set_env.sh"
set -euo pipefail
export ASCEND_HOME_PATH
export PATH="${MPI_ENV_BIN}:$PATH"
export LD_LIBRARY_PATH="${MPI_ENV_LIB}:${LD_LIBRARY_PATH:-}"
export MPI_LIB_PATH

while [[ $# -gt 0 ]]; do
  case "$1" in
    --soc) SOC="$2"; shift 2 ;;
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --max-output-size) MAX_OUTPUT_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --case-mode) CASE_MODE="$2"; shift 2 ;;
    --atol) ATOL="$2"; shift 2 ;;
    --rtol) RTOL="$2"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
    --measure-iters) MEASURE_ITERS="$2"; shift 2 ;;
    --skip-golden) SKIP_GOLDEN=1; shift ;;
    --stage-profile|-stage-profile) STAGE_PROFILE=1; shift ;;
    --start-sync-debug) START_SYNC_DEBUG=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --trace) TRACE=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

if [[ "${WORLD_SIZE}" == "2" && -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=0,1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR="${SCRIPT_DIR}/build"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${DISPATCH_FFN_COMBINE_V2_LOG_DIR:-"${OUT_DIR}/ascend_logs/${RUN_STAMP}"}
TRACE_DIR="${LOG_DIR}/host_trace"
mkdir -p "${LOG_DIR}"
mkdir -p "${TRACE_DIR}"

GEN_DATA_EXTRA_ARGS=()
if [[ "${SKIP_GOLDEN}" != "0" ]]; then
  GEN_DATA_EXTRA_ARGS+=(--skip-golden)
  export DISPATCH_FFN_COMBINE_V2_SKIP_ACCURACY=1
fi

export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-${LOG_DIR}}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export HCCL_ENTRY_LOG_ENABLE="${HCCL_ENTRY_LOG_ENABLE:-0}"
export DISPATCH_FFN_COMBINE_V2_TRACE="${TRACE}"
export DISPATCH_FFN_COMBINE_V2_TRACE_FILE_DIR="${TRACE_DIR}"

MIB=$((1024 * 1024))
PACKED_OFFSET_A_BYTES=$((MAX_OUTPUT_SIZE * (K + 32)))
OFFSET_A_WINDOW_BYTES=$((PACKED_OFFSET_A_BYTES * 3))
OFFSET_D_BYTES=$((MAX_OUTPUT_SIZE * K * 2))
OFFSET_D_WINDOW_BYTES=$((((OFFSET_D_BYTES + 3 * MIB + 511) * 3 + 1) / 2))
NEEDED_WINDOW_BYTES="${OFFSET_A_WINDOW_BYTES}"
if [[ "${OFFSET_D_WINDOW_BYTES}" -gt "${NEEDED_WINDOW_BYTES}" ]]; then
  NEEDED_WINDOW_BYTES="${OFFSET_D_WINDOW_BYTES}"
fi
NEEDED_HCCL_BUFFSIZE_MB=$(((NEEDED_WINDOW_BYTES + MIB - 1) / MIB + 64))
CURRENT_HCCL_BUFFSIZE_MB="${HCCL_BUFFSIZE:-200}"
if [[ "${CURRENT_HCCL_BUFFSIZE_MB}" -lt "${NEEDED_HCCL_BUFFSIZE_MB}" ]]; then
  echo "[INFO] Raising HCCL_BUFFSIZE from ${CURRENT_HCCL_BUFFSIZE_MB} to ${NEEDED_HCCL_BUFFSIZE_MB} MB" \
    "for maxOutputSize=${MAX_OUTPUT_SIZE} K=${K}"
  export HCCL_BUFFSIZE="${NEEDED_HCCL_BUFFSIZE_MB}"
fi

echo "ASCEND_PROCESS_LOG_PATH=${ASCEND_PROCESS_LOG_PATH}"
echo "ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL}"
echo "ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT}"
echo "HCCL_ENTRY_LOG_ENABLE=${HCCL_ENTRY_LOG_ENABLE}"
echo "HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}"
echo "SKIP_BUILD=${SKIP_BUILD}"
echo "TRACE=${TRACE}"
echo "START_SYNC_DEBUG=${START_SYNC_DEBUG}"
echo "DISPATCH_FFN_COMBINE_V2_TRACE_FILE_DIR=${DISPATCH_FFN_COMBINE_V2_TRACE_FILE_DIR}"

python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --m "${M}" --k "${K}" --n "${N}" \
  --topk "${TOPK}" --experts "${EXPERTS}" \
  --max-output-size "${MAX_OUTPUT_SIZE}" \
  --seed "${SEED}" \
  --case-mode "${CASE_MODE}" \
  --atol "${ATOL}" \
  --rtol "${RTOL}" \
  "${GEN_DATA_EXTRA_ARGS[@]}"

if [[ "${SKIP_BUILD}" == "0" ]]; then
  rm -rf \
    "${BUILD_DIR}/dispatch_ffn_combine_v2_kernel_host_dir" \
    "${BUILD_DIR}/dispatch_ffn_combine_v2_kernel_host-prefix" \
    "${BUILD_DIR}/CMakeFiles/dispatch_ffn_combine_v2_kernel_host_stub_obj.dir" \
    "${BUILD_DIR}/lib/libdispatch_ffn_combine_v2_kernel.so"

  cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DSOC_VERSION="${SOC}"
  cmake --build "${BUILD_DIR}" --target dispatch_ffn_combine_v2 -j16
else
  if [[ ! -x "${BUILD_DIR}/dispatch_ffn_combine_v2" ]]; then
    echo "skip-build requested but ${BUILD_DIR}/dispatch_ffn_combine_v2 does not exist or is not executable" >&2
    echo "run once without --skip-build to build it first" >&2
    exit 1
  fi
  if [[ ! -f "${BUILD_DIR}/lib/libdispatch_ffn_combine_v2_kernel.so" ]]; then
    echo "skip-build requested but ${BUILD_DIR}/lib/libdispatch_ffn_combine_v2_kernel.so does not exist" >&2
    echo "run once without --skip-build to build it first" >&2
    exit 1
  fi
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH}"
export DISPATCH_FFN_COMBINE_V2_CASE_DIR="${OUT_DIR}"
export DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS="${WARMUP_ITERS}"
export DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS="${MEASURE_ITERS}"
export DISPATCH_FFN_COMBINE_V2_STAGE_PROFILE="${STAGE_PROFILE}"
export DISPATCH_FFN_COMBINE_V2_START_SYNC_DEBUG="${START_SYNC_DEBUG}"
echo "RUN_MPI=${MPI_RUNNER} -n ${WORLD_SIZE} ${BUILD_DIR}/dispatch_ffn_combine_v2"
"${MPI_RUNNER}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_ffn_combine_v2"
