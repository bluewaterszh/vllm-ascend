#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
SOC_VERSION="${SOC_VERSION:-ascend910_93}"
BUILD_JOBS="${BUILD_JOBS:-16}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DSOC_VERSION="${SOC_VERSION}"
cmake --build "${BUILD_DIR}" --target dispatch_ffn_combine_v4 -j"${BUILD_JOBS}"
exec "${BUILD_DIR}/dispatch_ffn_combine_v4" "$@"
