# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.ops.kda_backward: the Python op running the C++ KdaBackward."""

import itertools
import sys
from pathlib import Path

import pytest
import torch

from ops_test_utils import requires_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "linear_attention"))
from reference_kda import kda_reference, rms_ratio  # noqa: E402

pytestmark = [pytest.mark.L0, requires_ops]
NAMES = ("dq", "dk", "dv", "dg", "dbeta", "d_initial_state")


def _inputs(lens, h=4, d=128, seed=0):
    torch.manual_seed(seed)
    t = sum(lens)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device="cuda")
    unit = lambda: torch.nn.functional.normalize(torch.randn(t, h, d, device="cuda"), dim=-1).bfloat16()  # noqa: E731
    q, k = unit(), unit()
    v = torch.randn(t, h, d, dtype=torch.bfloat16, device="cuda")
    g = -0.1 * torch.rand(t, h, d, device="cuda")
    beta = torch.rand(t, h, device="cuda")
    s0 = 0.1 * torch.randn(len(lens), h, d, d, device="cuda")
    do = torch.randn(t, h, d, dtype=torch.bfloat16, device="cuda")
    ds_t = 0.1 * torch.randn_like(s0)
    return q, k, v, g, beta, cu, s0, do, ds_t


@pytest.mark.parametrize("opts", [dict(use_qk_l2norm=True), dict(use_qk_l2norm=True, use_beta_sigmoid=True), dict()])
def test_bitwise_equal_to_the_frost_engine(opts):
    """On shapes where the engine's backward also runs the uncut schedule, the op matches it bit for bit."""
    import cudnn
    from cudnn.linear_attention.ops.kda import kimi_delta_attention

    q, k, v, g, beta, cu, s0, do, ds_t = _inputs([64, 64, 64, 64])
    grads = cudnn.ops.kda_backward(q, k, v, g, beta, cu, do, initial_state=s0, d_final_state=ds_t, **opts)
    leaves = [t.clone().requires_grad_() for t in (q, k, v, g, beta, s0)]
    engine = dict(use_qk_l2norm_in_kernel=opts.get("use_qk_l2norm", False), use_beta_sigmoid_in_kernel=opts.get("use_beta_sigmoid", False))
    o, s_t = kimi_delta_attention(*leaves[:5], cu, initial_state=leaves[5], output_final_state=True, batch_invariant=True, **engine)
    torch.autograd.backward([o, s_t], [do, ds_t])
    for name, grad, leaf in zip(NAMES, grads, leaves):
        assert torch.equal(grad, leaf.grad), name


def test_matches_fp64_reference_on_ragged_lengths():
    import cudnn

    q, k, v, g, beta, cu, s0, do, ds_t = _inputs([100, 37, 300, 1], d=64)
    grads = cudnn.ops.kda_backward(q, k, v, g, beta, cu, do, initial_state=s0, d_final_state=ds_t)
    ref = [t.double().clone().requires_grad_() for t in (q, k, v, g, beta, s0)]
    o, s_t = kda_reference(*(t[None] for t in ref[:5]), initial_state=ref[5], cu_seqlens=cu)
    torch.autograd.backward([o[0], s_t], [do.double(), ds_t.double()])
    for name, grad, leaf in zip(NAMES, grads, ref):
        assert rms_ratio(grad, leaf.grad) < 4e-2, name


def test_without_states():
    import cudnn

    q, k, v, g, beta, cu, _, do, _ = _inputs([50, 70])
    *grads, ds0 = cudnn.ops.kda_backward(q, k, v, g, beta, cu, do)
    assert ds0 is None
    ref = [t.double().clone().requires_grad_() for t in (q, k, v, g, beta)]
    o, _ = kda_reference(*(t[None] for t in ref), cu_seqlens=cu)
    o[0].backward(do.double())
    for name, grad, leaf in zip(NAMES, grads, ref):
        assert rms_ratio(grad, leaf.grad) < 4e-2, name
