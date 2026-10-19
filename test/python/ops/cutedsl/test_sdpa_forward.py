# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.ops.sdpa_forward: the Python op running the C++ SdpaForward."""

import itertools
import math

import pytest
import torch

from ops_test_utils import requires_ops

pytestmark = [pytest.mark.L0, requires_ops]


def _inputs(lq, lk, hq, hk, d, dtype, seed=0):
    torch.manual_seed(seed)
    cuq = [0] + list(itertools.accumulate(lq))
    cuk = [0] + list(itertools.accumulate(lk))
    q = torch.randn(cuq[-1], hq, d, dtype=dtype, device="cuda")
    k = torch.randn(cuk[-1], hk, d, dtype=dtype, device="cuda")
    v = torch.randn(cuk[-1], hk, d, dtype=dtype, device="cuda")
    as_tensor = lambda cu: torch.tensor(cu, dtype=torch.int32, device="cuda")  # noqa: E731
    return q, k, v, cuq, cuk, as_tensor(cuq), as_tensor(cuk)


def _reference(q, k, v, cuq, cuk, causal, scale):
    """fp32 per-sequence attention: O [Tq, H, D] and the natural-log LSE [Tq, H]."""
    o = torch.empty(q.shape[0], q.shape[1], v.shape[2], device="cuda")
    lse = torch.empty(q.shape[0], q.shape[1], device="cuda")
    group = q.shape[1] // k.shape[1]
    for i in range(len(cuq) - 1):
        qs, ks = slice(cuq[i], cuq[i + 1]), slice(cuk[i], cuk[i + 1])
        qi = q[qs].float().transpose(0, 1)
        ki = k[ks].float().transpose(0, 1).repeat_interleave(group, 0)
        vi = v[ks].float().transpose(0, 1).repeat_interleave(group, 0)
        s = qi @ ki.transpose(1, 2) * scale
        if causal:
            s = s.masked_fill(torch.ones(s.shape[1:], dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        lse[qs] = s.logsumexp(-1).transpose(0, 1)
        o[qs] = (s.softmax(-1) @ vi).transpose(0, 1)
    return o, lse


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [(8, 2), (4, 4)])
def test_matches_reference(dtype, causal, heads):
    import cudnn

    hq, hk = heads
    q, k, v, cuq, cuk, cuq_t, cuk_t = _inputs([100, 37, 300, 1], [100, 37, 300, 1] if causal else [120, 5, 400, 9], hq, hk, 128, dtype)
    o, lse = cudnn.ops.sdpa_forward(q, k, v, cuq_t, cuk_t, causal=causal, return_stats=True)
    o_ref, lse_ref = _reference(q, k, v, cuq, cuk, causal, 1 / math.sqrt(128))
    torch.testing.assert_close(o.float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_scale_and_no_stats():
    import cudnn

    q, k, v, cuq, cuk, cuq_t, cuk_t = _inputs([64, 200], [64, 200], 8, 2, 128, torch.bfloat16)
    o, stats = cudnn.ops.sdpa_forward(q, k, v, cuq_t, cuk_t, scale=0.05, causal=True)
    assert stats is None
    torch.testing.assert_close(o.float(), _reference(q, k, v, cuq, cuk, True, 0.05)[0], atol=2e-2, rtol=2e-2)


def test_one_kernel_serves_every_shape():
    """Token counts, sequence counts and lengths bind per call: one configuration, one compile."""
    import cudnn
    from cudnn.ops import _library

    for lq in ([5], [100, 37, 300], [1] * 17):
        q, k, v, cuq, cuk, cuq_t, cuk_t = _inputs(lq, lq, 8, 2, 128, torch.bfloat16)
        o, _ = cudnn.ops.sdpa_forward(q, k, v, cuq_t, cuk_t, causal=True)
        torch.testing.assert_close(o.float(), _reference(q, k, v, cuq, cuk, True, 1 / math.sqrt(128))[0], atol=2e-2, rtol=2e-2)
    configs = [c for c in _library._REGISTERED if '"sdpa_forward"' in c and '"h_q":8' in c and '"causal":true' in c and '"stats":false' in c]
    assert len(configs) == 1


def test_capture_and_replay():
    """execute() is capture-safe: a graph replays with the inputs' new values."""
    import cudnn

    q, k, v, cuq, cuk, cuq_t, cuk_t = _inputs([100, 37], [100, 37], 8, 2, 128, torch.bfloat16)
    cudnn.ops.sdpa_forward(q, k, v, cuq_t, cuk_t, causal=True)  # plan and kernel outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o, _ = cudnn.ops.sdpa_forward(q, k, v, cuq_t, cuk_t, causal=True)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(o.float(), _reference(q, k, v, cuq, cuk, True, 1 / math.sqrt(128))[0], atol=2e-2, rtol=2e-2)


def test_execute_refuses_other_shapes():
    import cudnn
    from cudnn import _compiled_module
    from cudnn.ops import _tensors

    q, k, v, _, _, cuq_t, cuk_t = _inputs([100, 37], [100, 37], 8, 2, 128, torch.bfloat16)
    cudnn.ops.sdpa_forward(q, k, v, cuq_t, cuk_t)
    o = torch.empty_like(q)
    descs = {n: _tensors.describe(t) for n, t in dict(q=q, k=k, v=v, o=o, cu_seqlens_q=cuq_t, cu_seqlens_kv=cuk_t).items()}
    plan = _compiled_module._SdpaForward(descs, dict(causal=False))
    ws = torch.empty(plan.workspace_size, dtype=torch.uint8, device="cuda")
    shorter = dict(descs, q=_tensors.describe(q[:10]), o=_tensors.describe(o[:10]))
    with pytest.raises(Exception, match="differ from create"):
        plan.execute(shorter, ws.data_ptr(), 0)
