# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel libraries: Python compiles, C++ loads; the samples under samples/ops run from C++ with no Python."""

import json
import struct
import sys

import pytest

from ops_test_utils import build_sample, requires_ops, run

pytestmark = [pytest.mark.L0, requires_ops]


SAMPLES = ("sdpa_forward", "sdpa_backward", "kda_forward", "kda_backward")


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    out = tmp_path_factory.mktemp("ops_samples")
    exes = {name: build_sample(name, str(out)) for name in SAMPLES}
    configs = {name: run([exe, "--config"], check=True).stdout.split() for name, exe in exes.items()}
    (out / "configs.txt").write_text("\n".join(c for cs in configs.values() for c in cs) + "\n")
    proc = run([sys.executable, "-m", "cudnn.ops", str(out / "configs.txt"), "-o", str(out / "kernels.cudnn")])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out, exes, configs


@pytest.mark.parametrize("name", SAMPLES)
def test_cpp_sample_runs_from_a_library_file(samples, name):
    out, exes, _ = samples
    proc = run([exes[name], str(out / "kernels.cudnn")])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "rms error" in proc.stdout


def test_python_config_is_the_cpp_config(samples):
    from cudnn.ops import _library

    _, _, configs = samples
    for config in (c for cs in configs.values() for c in cs):
        assert _library.canonical(config) == config


def test_missing_kernel_names_its_configuration(samples, tmp_path):
    from cudnn.ops import _library

    _, exes, configs = samples
    _library.compile(configs["kda_forward"]).save(tmp_path / "kda_only.cudnn")  # no SDPA kernel in it
    proc = run([exes["sdpa_forward"], str(tmp_path / "kda_only.cudnn")])
    assert proc.returncode != 0
    assert configs["sdpa_forward"][0] in proc.stderr and "cudnn.ops.compile" in proc.stderr


def test_library_for_another_gpu_is_refused(samples, tmp_path):
    out, exes, _ = samples
    data = (out / "kernels.cudnn").read_bytes()
    (size,) = struct.unpack("<Q", data[8:16])
    header = json.loads(data[16 : 16 + size])
    header["target"]["num_sm"] += 1
    raw = json.dumps(header).encode()
    (tmp_path / "other.cudnn").write_bytes(b"CUDNNOPS" + struct.pack("<Q", len(raw)) + raw + data[16 + size :])
    proc = run([exes["sdpa_forward"], str(tmp_path / "other.cudnn")])
    assert proc.returncode != 0 and "kernel library is for" in proc.stderr


def test_registry_miss_is_an_error_not_a_compile():
    """C++ never compiles: a configuration nobody registered fails create() with its name."""
    import torch

    import cudnn
    from cudnn import _compiled_module
    from cudnn.ops import _tensors

    cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    x = torch.zeros(16, 2, 64, dtype=torch.float16, device="cuda")
    beta = torch.zeros(16, 2, dtype=torch.float16, device="cuda")  # an fp16 beta: a configuration no other test uses
    descs = {n: _tensors.describe(t) for n, t in dict(q=x, k=x, v=x, g=x, beta=beta, o=x, cu_seqlens=cu).items()}
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match='"beta":"fp16"'):
        _compiled_module._KdaForward(descs, {})
