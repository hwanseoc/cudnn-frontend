# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.ops.kda_forward: the Python op running the C++ KdaForward."""

import itertools
import sys
from pathlib import Path

import pytest
import torch

from ops_test_utils import requires_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "linear_attention"))
from reference_kda import kda_reference, rms_ratio  # noqa: E402

pytestmark = [pytest.mark.L0, requires_ops]


def _inputs(lens, hq, hv, d, dtype=torch.bfloat16, state_dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    ho = max(hq, hv)
    t = sum(lens)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device="cuda")
    # unit-norm keys keep the delta rule bounded whether or not the op normalizes them
    q = torch.nn.functional.normalize(torch.randn(t, hq, d, device="cuda"), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn(t, hq, d, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(t, hv, d, dtype=dtype, device="cuda")
    g = -0.1 * torch.rand(t, ho, d, device="cuda")
    beta = torch.rand(t, ho, device="cuda")
    s0 = (0.1 * torch.randn(len(lens), ho, d, d, device="cuda")).to(state_dtype)
    return q, k, v, g, beta, cu, s0


CASES = {
    "l2norm": dict(use_qk_l2norm=True),
    "beta_sigmoid": dict(use_qk_l2norm=True, use_beta_sigmoid=True, allow_neg_eigval=True),
    "plain": dict(),
}


@pytest.mark.parametrize("case", list(CASES))
def test_bitwise_equal_to_the_frost_engine(case):
    """The op runs the engine's uncut forward, so it matches the engine's batch-invariant path bit for bit."""
    import cudnn
    from cudnn.linear_attention.ops.kda import kimi_delta_attention

    opts = CASES[case]
    q, k, v, g, beta, cu, s0 = _inputs([100, 37, 300, 1], 4, 4, 128)
    o, s1 = cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, **opts)
    engine = dict(
        use_qk_l2norm_in_kernel=opts.get("use_qk_l2norm", False),
        use_beta_sigmoid_in_kernel=opts.get("use_beta_sigmoid", False),
        allow_neg_eigval=opts.get("allow_neg_eigval", False),
    )
    o_ref, s_ref = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0.clone(), output_final_state=True, batch_invariant=True, **engine)
    assert torch.equal(o, o_ref)
    assert torch.equal(s1, s_ref)


@pytest.mark.parametrize("heads", [(4, 4), (2, 4), (4, 2)])
def test_matches_reference(heads):
    import cudnn

    q, k, v, g, beta, cu, s0 = _inputs([64, 33, 150], *heads, 64)
    o, s1 = cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True)
    o_ref, s_ref = kda_reference(q[None], k[None], v[None], g[None], beta[None], initial_state=s0, cu_seqlens=cu)
    assert rms_ratio(o, o_ref[0]) < 2e-2
    assert rms_ratio(s1, s_ref) < 2e-2


def test_no_state_and_bf16_state():
    import cudnn

    q, k, v, g, beta, cu, s0 = _inputs([50, 70], 4, 4, 128, state_dtype=torch.bfloat16)
    o, s1 = cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True)
    assert s1.dtype == torch.bfloat16
    o_ref, _ = kda_reference(q[None], k[None], v[None], g[None], beta[None], initial_state=s0.float(), cu_seqlens=cu)
    assert rms_ratio(o, o_ref[0]) < 2e-2
    o, s1 = cudnn.ops.kda_forward(q, k, v, g, beta, cu)
    assert s1 is None
    o_ref, _ = kda_reference(q[None], k[None], v[None], g[None], beta[None], cu_seqlens=cu)
    assert rms_ratio(o, o_ref[0]) < 2e-2


def test_safe_gate():
    import cudnn

    q, k, v, g, beta, cu, s0 = _inputs([80, 20], 4, 4, 128)
    a_log = torch.randn(4, device="cuda") * 0.1
    dt_bias = torch.randn(4, 128, device="cuda") * 0.1
    o, _ = cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=s0, safe_gate=True, a_log=a_log, dt_bias=dt_bias)
    o_ref, _ = kda_reference(q[None], k[None], v[None], g[None], beta[None], initial_state=s0, cu_seqlens=cu, safe_gate=True, a_log=a_log, dt_bias=dt_bias)
    assert rms_ratio(o, o_ref[0]) < 2e-2


def test_execute_neither_allocates_nor_synchronizes():
    """Rule 8: a warm call does no host synchronization (outputs and workspace come from torch's caching allocator)."""
    import cudnn

    q, k, v, g, beta, cu, s0 = _inputs([100, 37], 4, 4, 128)
    cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
