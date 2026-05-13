#!/usr/bin/env bash
#
# Build the standalone dispatch_ffn_combine .run package.
#
# Environment is intentionally externalized. Export variables in your shell
# before running this script.
#
# Common usage:
#   export CPATH="$PWD/csrc/third_party/catlass/include:${CPATH:-}"
#   bash tools/build_dispatch_ffn_combine_run_package.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CSRC_ROOT="${REPO_ROOT}/csrc"
BUILD_SCRIPT="${CSRC_ROOT}/build.sh"

OPS="${OPS:-dispatch_ffn_combine}"
SOC_VERSION="${SOC_VERSION:-ascend910b}"
BUILD_JOBS="${BUILD_JOBS:-}"
BUILD_OPT_LEVEL="${BUILD_OPT_LEVEL:-3}"
BUILD_EXPERIMENTAL="${BUILD_EXPERIMENTAL:-0}"
BUILD_OOM="${BUILD_OOM:-0}"
PACKAGE_PATH="${PACKAGE_PATH:-${CSRC_ROOT}/build/cann-ops-transformer-custom_linux-aarch64.run}"

if [[ ! -f "${BUILD_SCRIPT}" ]]; then
    echo "[ERROR] build script not found: ${BUILD_SCRIPT}" >&2
    exit 2
fi

if [[ -n "${BUILD_JOBS}" ]]; then
    if ! [[ "${BUILD_JOBS}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] BUILD_JOBS must be an integer, got: ${BUILD_JOBS}" >&2
        exit 2
    fi
fi

case "${BUILD_OPT_LEVEL}" in
    0|1|2|3)
        ;;
    *)
        echo "[ERROR] BUILD_OPT_LEVEL must be one of: 0 1 2 3" >&2
        exit 2
        ;;
esac

build_args=("--pkg" "--ops=${OPS}" "--soc=${SOC_VERSION}" "-O${BUILD_OPT_LEVEL}")
if [[ -n "${BUILD_JOBS}" ]]; then
    build_args+=("-j${BUILD_JOBS}")
fi
if [[ "${BUILD_EXPERIMENTAL}" == "1" ]]; then
    build_args+=("--experimental")
fi
if [[ "${BUILD_OOM}" == "1" ]]; then
    build_args+=("--oom")
fi

echo "[INFO] repo root: ${REPO_ROOT}"
echo "[INFO] csrc root: ${CSRC_ROOT}"
echo "[INFO] build command: bash ${BUILD_SCRIPT} ${build_args[*]}"

cd "${CSRC_ROOT}"
bash "${BUILD_SCRIPT}" "${build_args[@]}"

if [[ ! -f "${PACKAGE_PATH}" ]]; then
    echo "[ERROR] package was not produced at expected path: ${PACKAGE_PATH}" >&2
    exit 2
fi

echo "[INFO] package ready: ${PACKAGE_PATH}"
