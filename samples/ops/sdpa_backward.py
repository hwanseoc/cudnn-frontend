# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Attention forward with its log-sum-exp, then its gradients, over three packed sequences, from Python."""

import torch

import cudnn
import reference

cu = torch.tensor([0, 100, 137, 437], dtype=torch.int32, device="cuda")  # where each sequence starts
q = torch.randn(437, 4, 128, dtype=torch.bfloat16, device="cuda")  # [tokens, heads, head dim]
k = torch.randn(437, 4, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(437, 4, 128, dtype=torch.bfloat16, device="cuda")

o, stats = cudnn.ops.sdpa_forward(q, k, v, cu, cu, causal=True, return_stats=True)
do = torch.randn_like(o)
dq, dk, dv = cudnn.ops.sdpa_backward(q, k, v, o, do, stats, cu, cu, causal=True)

expected = reference.gradients(lambda q, k, v: reference.sdpa(q, k, v, cu.tolist(), cu.tolist(), causal=True), (q, k, v), do)
error = max(reference.rms_error(x, ref) for x, ref in zip((dq, dk, dv), expected))
print(f"sdpa_backward: worst rms error {error:.2e} vs fp64 reference (dq, dk, dv)")
assert error < 2e-2
