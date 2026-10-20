# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared skips and builders for the cudnn.ops tests."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[4]


def _supported():
    if not torch.cuda.is_available():
        return "no CUDA device"
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        return "cudnn.ops SDPA/KDA kernels run on SM100/SM103"
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        return "needs nvidia-cutlass-dsl >= 4.7"
    return None


requires_ops = pytest.mark.skipif(_supported() is not None, reason=str(_supported()))


def cuda_home():
    for env in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"):
        if os.environ.get(env):
            return os.environ[env]
    nvcc = shutil.which("nvcc")
    return str(Path(nvcc).resolve().parents[1]) if nvcc else None


def cudnn_home():
    import nvidia.cudnn

    return os.path.dirname(nvidia.cudnn.__file__)


def build_sample(name, out_dir):
    """Build samples/ops/<name>.cpp with the samples' warning flags; skip when there is no toolchain."""
    cxx, cuda = shutil.which(os.environ.get("CXX", "g++")), cuda_home()
    if cxx is None or cuda is None:
        pytest.skip("needs a C++ compiler and the CUDA toolkit")
    exe = os.path.join(out_dir, name)
    cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-function",
        "-fno-rtti",
        f"-I{REPO / 'include'}",
        f"-I{cuda}/include",
        f"-I{cudnn_home()}/include",
        str(REPO / "samples" / "ops" / f"{name}.cpp"),
        f"-L{cuda}/lib64",
        f"-L{cudnn_home()}/lib",
        "-lcudart",
        "-l:libcudnn.so.9",
        "-ldl",
        "-o",
        exe,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return exe


def run(cmd, text=True, **kwargs):
    env = dict(os.environ, LD_LIBRARY_PATH=f"{cudnn_home()}/lib:{cuda_home()}/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""))
    return subprocess.run(cmd, capture_output=True, text=text, env=env, **kwargs)
