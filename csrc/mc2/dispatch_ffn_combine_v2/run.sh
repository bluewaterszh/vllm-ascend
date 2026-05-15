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
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR="${SCRIPT_DIR}/build"

python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --m "${M}" --k "${K}" --n "${N}" \
  --topk "${TOPK}" --experts "${EXPERTS}" \
  --max-output-size "${MAX_OUTPUT_SIZE}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DSOC_VERSION="${SOC}"
cmake --build "${BUILD_DIR}" --target dispatch_ffn_combine_v2 -j16

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH}"
export DISPATCH_FFN_COMBINE_V2_CASE_DIR="${OUT_DIR}"
"${MPI_RUNNER}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_ffn_combine_v2"
