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
ATOL=1e-4
RTOL=1e-3
WARMUP_ITERS=${DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS:-3}
MEASURE_ITERS=${DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS:-5}

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}
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
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR="${SCRIPT_DIR}/build"

rm -rf \
  "${BUILD_DIR}/dispatch_ffn_combine_v2_kernel_host_dir" \
  "${BUILD_DIR}/dispatch_ffn_combine_v2_kernel_host-prefix" \
  "${BUILD_DIR}/CMakeFiles/dispatch_ffn_combine_v2_kernel_host_stub_obj.dir" \
  "${BUILD_DIR}/lib/libdispatch_ffn_combine_v2_kernel.so"

python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --m "${M}" --k "${K}" --n "${N}" \
  --topk "${TOPK}" --experts "${EXPERTS}" \
  --max-output-size "${MAX_OUTPUT_SIZE}" \
  --seed "${SEED}" \
  --case-mode "${CASE_MODE}" \
  --atol "${ATOL}" \
  --rtol "${RTOL}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DSOC_VERSION="${SOC}"
cmake --build "${BUILD_DIR}" --target dispatch_ffn_combine_v2 -j16

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH}"
export DISPATCH_FFN_COMBINE_V2_CASE_DIR="${OUT_DIR}"
export DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS="${WARMUP_ITERS}"
export DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS="${MEASURE_ITERS}"
"${MPI_RUNNER}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_ffn_combine_v2"
