# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.ops.sdpa_backward: the Python op running the C++ SdpaBackward."""

import itertools
import math

import pytest
import torch

from ops_test_utils import requires_ops

pytestmark = [pytest.mark.L0, requires_ops]


def _rms(out, ref):
    return ((out.double() - ref.double()).pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt()).item()


def _check(lens, heads, dtype, causal, max_seqlen=None):
    import cudnn

    torch.manual_seed(0)
    cu = [0] + list(itertools.accumulate(lens))
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
    q, k, v = (torch.randn(cu[-1], heads, 128, dtype=dtype, device="cuda") for _ in range(3))
    o, lse = cudnn.ops.sdpa_forward(q, k, v, cu_t, cu_t, causal=causal, return_stats=True)
    do = torch.randn_like(o)
    dq, dk, dv = cudnn.ops.sdpa_backward(q, k, v, o, do, lse, cu_t, cu_t, causal=causal, max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen)
    leaves = [t.float().requires_grad_() for t in (q, k, v)]
    outs = []
    for i in range(len(lens)):
        s = slice(cu[i], cu[i + 1])
        qi, ki, vi = (t[s].transpose(0, 1) for t in leaves)
        a = qi @ ki.transpose(1, 2) / math.sqrt(128)
        if causal:
            a = a.masked_fill(torch.ones(a.shape[1:], dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        outs.append((a.softmax(-1) @ vi).transpose(0, 1))
    torch.cat(outs).backward(do.float())
    for grad, leaf in zip((dq, dk, dv), leaves):
        assert _rms(grad, leaf.grad) < 2e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [True, False])
def test_matches_autograd(dtype, causal):
    _check([100, 37, 300, 5], 4, dtype, causal)


def test_max_seqlen_bound_and_empty_sequences():
    _check([0, 129, 0, 64], 2, torch.bfloat16, True, max_seqlen=129)


def test_kernel_is_per_sequence_capacity():
    """The backward kernel bakes in the sequence capacity: a second capacity is a second configuration."""
    from cudnn.ops import _library

    _check([50, 60], 4, torch.bfloat16, True)
    _check([50, 60, 70], 4, torch.bfloat16, True)
    configs = [c for c in _library._REGISTERED if '"sdpa_backward"' in c and '"h":4' in c and '"causal":true' in c and '"dtype":"bf16"' in c]
    assert {'"max_seqs":2' in c for c in configs} == {True, False}
