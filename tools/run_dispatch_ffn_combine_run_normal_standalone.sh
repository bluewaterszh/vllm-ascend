#!/usr/bin/env bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Minimal standalone runner for dispatch_ffn_combine on the current fork.
# It installs the latest custom-op package into the repo-local custom-op folder
# and only runs the `run_normal()` path in fresh worker processes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ntlab/miniconda3/envs/pto-zy/bin/python}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}"
PACKAGE_PATH="${PACKAGE_PATH:-${REPO_ROOT}/csrc/build/cann-ops-transformer-custom_linux-aarch64.run}"
INSTALL_DIR="${INSTALL_DIR:-${REPO_ROOT}/vllm_ascend/_cann_ops_custom}"
TEST_PATH="${TEST_PATH:-${REPO_ROOT}/tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dispatch_ffn_combine.py}"
WORLD_SIZE="${WORLD_SIZE:-2}"
QUEUE_TIMEOUT_SECONDS="${QUEUE_TIMEOUT_SECONDS:-240}"
JOIN_TIMEOUT_SECONDS="${JOIN_TIMEOUT_SECONDS:-15}"
INSTALL_PACKAGE="${INSTALL_PACKAGE:-1}"
VERBOSE_LOGS="${VERBOSE_LOGS:-1}"
BASE_SEED="${BASE_SEED:-20260512}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] python not found or not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ ! -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    echo "[ERROR] CANN env script not found: ${ASCEND_HOME_PATH}/set_env.sh" >&2
    exit 2
fi

if [[ ! -f "${TEST_PATH}" ]]; then
    echo "[ERROR] test file not found: ${TEST_PATH}" >&2
    exit 2
fi

if [[ "${INSTALL_PACKAGE}" == "1" && ! -f "${PACKAGE_PATH}" ]]; then
    echo "[ERROR] package file not found: ${PACKAGE_PATH}" >&2
    exit 2
fi

export ASCEND_HOME_PATH
export PYTHONPATH="${PYTHONPATH:-}"
export VERBOSE_LOGS
export BASE_SEED
# shellcheck disable=SC1090
set +e
set +u
set +o pipefail
source "${ASCEND_HOME_PATH}/set_env.sh"
source_status=$?
set -euo pipefail
if [[ "${source_status}" -ne 0 ]]; then
    echo "[ERROR] failed to source ${ASCEND_HOME_PATH}/set_env.sh" >&2
    exit "${source_status}"
fi

export PATH="/home/ntlab/miniconda3/envs/pto-zy/bin:/home/ntlab/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="/home/ntlab/miniconda3/envs/pto-zy/lib:/home/ntlab/miniconda3/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${INSTALL_PACKAGE}" == "1" ]]; then
    mkdir -p "${INSTALL_DIR}"
    chmod -R u+w "${INSTALL_DIR}" 2>/dev/null || true
    find "${INSTALL_DIR}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf -- {} +
    chmod +x "${PACKAGE_PATH}" || true
    "${PACKAGE_PATH}" --install-path="${INSTALL_DIR}"
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" - "${TEST_PATH}" "${WORLD_SIZE}" "${QUEUE_TIMEOUT_SECONDS}" "${JOIN_TIMEOUT_SECONDS}" <<'PY'
import importlib.util
import os
import pathlib
import queue
import random
import sys
import time
import traceback

test_path = pathlib.Path(sys.argv[1])
world_size = int(sys.argv[2])
queue_timeout_seconds = int(sys.argv[3])
join_timeout_seconds = int(sys.argv[4])
verbose_logs = os.environ.get("VERBOSE_LOGS", "1") != "0"
base_seed = int(os.environ.get("BASE_SEED", "20260512"))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import vllm.envs as vllm_envs

vllm_envs.VLLM_BATCH_INVARIANT = False

_orig_randint = torch.randint


def compat_randint(*args, **kwargs):
    if len(args) == 4 and isinstance(args[3], torch.dtype):
        low, high, size, dtype = args
        return _orig_randint(low, high, size, dtype=dtype, **kwargs)
    return _orig_randint(*args, **kwargs)


torch.randint = compat_randint

spec = importlib.util.spec_from_file_location("test_dispatch_ffn_combine", test_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

import vllm_ascend
from vllm_ascend import utils as vllm_ascend_utils

if verbose_logs:
    print(f"script_repo_root={pathlib.Path.cwd()}", flush=True)
    print(f"script_test_path={test_path}", flush=True)
    print(f"script_vllm_ascend_file={vllm_ascend.__file__}", flush=True)
    print(f"script_vllm_ascend_utils_file={vllm_ascend_utils.__file__}", flush=True)
    print(f"script_verbose_logs={verbose_logs}", flush=True)

registration_ok = hasattr(torch.ops._C_ascend, "dispatch_ffn_combine")
print(f"registration_ok={registration_ok}")
if not registration_ok:
    print("bottom_line=run_normal standalone failed before operator registration")
    raise SystemExit(1)

_orig_dispatch = torch.ops._C_ascend.dispatch_ffn_combine


def _to_float_cpu(tensor):
    if tensor is None:
        return None
    cpu_tensor = tensor.detach().to("cpu")
    if cpu_tensor.dtype != torch.float32:
        cpu_tensor = cpu_tensor.to(torch.float32)
    return cpu_tensor


def _sample_list(cpu_tensor, limit=8):
    if cpu_tensor is None or cpu_tensor.numel() == 0:
        return []
    return [round(float(value), 6) for value in cpu_tensor.reshape(-1)[:limit].tolist()]


def _tensor_summary(tensor, limit=8):
    cpu_tensor = _to_float_cpu(tensor)
    if cpu_tensor is None:
        return "none"
    numel = int(cpu_tensor.numel())
    if numel == 0:
        return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} numel=0"
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"sum={round(float(cpu_tensor.sum().item()), 6)} "
        f"absmax={round(float(cpu_tensor.abs().max().item()), 6)} "
        f"sample={_sample_list(cpu_tensor, limit)}"
    )


def _tensor_meta(tensor):
    if tensor is None:
        return "none"
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} numel={tensor.numel()}"


def _tensor_list_meta(value):
    if isinstance(value, (list, tuple)):
        length = len(value)
        first_shape = tuple(value[0].shape) if length > 0 and hasattr(value[0], "shape") else None
        return length, first_shape
    if value is None:
        return 0, None
    return 1, tuple(value.shape) if hasattr(value, "shape") else None


def compat_dispatch_ffn_combine(*args, **kwargs):
    if "bias1" in kwargs and isinstance(kwargs["bias1"], torch.Tensor) and kwargs["bias1"].numel() == 0:
        kwargs["bias1"] = None
    if "bias2" in kwargs and isinstance(kwargs["bias2"], torch.Tensor) and kwargs["bias2"].numel() == 0:
        kwargs["bias2"] = None
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "na"
    weight1_items, weight1_first_shape = _tensor_list_meta(kwargs.get("weight1"))
    weight2_items, weight2_first_shape = _tensor_list_meta(kwargs.get("weight2"))
    scale1_items, scale1_first_shape = _tensor_list_meta(kwargs.get("scale1"))
    scale2_items, scale2_first_shape = _tensor_list_meta(kwargs.get("scale2"))
    if verbose_logs:
        print(
            "dispatch_ffn_combine_call_begin "
            f"rank={rank} "
            f"x_shape={tuple(kwargs['x'].shape)} "
            f"expert_idx_shape={tuple(kwargs['expert_idx'].shape)} "
            f"probs_shape={tuple(kwargs['probs'].shape)} "
            f"weight1_items={weight1_items} weight1_first_shape={weight1_first_shape} "
            f"weight2_items={weight2_items} weight2_first_shape={weight2_first_shape} "
            f"scale1_items={scale1_items} scale1_first_shape={scale1_first_shape} "
            f"scale2_items={scale2_items} scale2_first_shape={scale2_first_shape} "
            f"has_x_active_mask={kwargs.get('x_active_mask') is not None} "
            f"group_is_set={bool(kwargs.get('group'))}",
            flush=True,
        )
        print(
            "dispatch_ffn_combine_buffers_declared "
            f"rank={rank} "
            f"out={_tensor_meta(kwargs.get('out'))} "
            f"expert_token_nums={_tensor_meta(kwargs.get('expert_token_nums'))}",
            flush=True,
        )
    result = _orig_dispatch(*args, **kwargs)
    if verbose_logs:
        print(
            "dispatch_ffn_combine_call_end "
            f"rank={rank} "
            "status=returned_without_python_exception",
            flush=True,
        )
        try:
            print(
                "dispatch_ffn_combine_buffers_after "
                f"rank={rank} "
                f"out={_tensor_summary(kwargs.get('out'))} "
                f"expert_token_nums={_tensor_summary(kwargs.get('expert_token_nums'))}",
                flush=True,
            )
        except Exception as exc:
            print(
                "dispatch_ffn_combine_buffers_after_error "
                f"rank={rank} "
                f"error_type={type(exc).__name__} "
                f"error={str(exc).splitlines()[0]}",
                flush=True,
            )
            raise
    return result


torch.ops._C_ascend.dispatch_ffn_combine = compat_dispatch_ffn_combine


def safe_worker(rank, group_world_size, port, q):
    try:
        worker_seed = base_seed + rank
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed"):
            torch.npu.manual_seed(worker_seed)
        if verbose_logs:
            print(f"worker_seed rank={rank} seed={worker_seed}", flush=True)
            print(f"worker_begin rank={rank} world_size={group_world_size} port={port}", flush=True)
        op = module.TestDispatchFFNCombine(rank, group_world_size, port)
        op.generate_hcom()
        if verbose_logs:
            print(f"worker_hcom_ready rank={rank}", flush=True)
        q.put(("stage_ok", rank, "generate_hcom"))
        try:
            if verbose_logs:
                print(f"worker_run_normal_begin rank={rank}", flush=True)
            out = op.run_normal()
            if verbose_logs:
                print(f"worker_run_normal_end rank={rank} result={bool(out)}", flush=True)
            q.put(("result", rank, "run_normal", bool(out)))
        except Exception:
            q.put(("error", rank, "run_normal", traceback.format_exc()))
            return
    except Exception:
        q.put(("error", rank, "worker_setup", traceback.format_exc()))


mp.set_start_method("fork", force=True)
result_queue = mp.Queue()
port = 29501 + random.randint(0, 10000)
processes = []

for rank in range(world_size):
    process = mp.Process(target=safe_worker, args=(rank, world_size, port, result_queue))
    process.start()
    processes.append(process)

events = []
deadline = time.time() + queue_timeout_seconds
while time.time() < deadline:
    if not any(process.is_alive() for process in processes) and result_queue.empty():
        break
    try:
        events.append(result_queue.get(timeout=1))
    except queue.Empty:
        continue

for process in processes:
    process.join(timeout=join_timeout_seconds)

proc_states = []
for process in processes:
    proc_states.append({"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()})
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)

stage_ok_events = [event for event in events if event[0] == "stage_ok"]
result_events = [event for event in events if event[0] == "result"]
error_events = [event for event in events if event[0] == "error"]

run_normal_results = [event for event in result_events if event[2] == "run_normal"]
probe_ok = (
    registration_ok
    and len(run_normal_results) == world_size
    and all(event[3] for event in run_normal_results)
    and not error_events
)

print(f"stage_ok_events={len(stage_ok_events)}")
print(f"result_events={result_events}")
print(f"proc_states={proc_states}")
if error_events:
    print("first_error_stage=" + error_events[0][2])
    print("first_error_begin")
    print(error_events[0][3].rstrip())
    print("first_error_end")
print(
    "bottom_line="
    + (
        "run_normal standalone succeeded"
        if probe_ok
        else "run_normal standalone failed"
    )
)

raise SystemExit(0 if probe_ok else 1)
PY
