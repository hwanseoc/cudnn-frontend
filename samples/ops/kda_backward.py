# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Kimi Delta Attention gradients over three packed sequences, from Python. The backward recomputes the forward's
states itself; it needs the forward's inputs and dO only."""

import torch

import cudnn
import reference

cu = torch.tensor([0, 50, 63, 138], dtype=torch.int32, device="cuda")  # where each sequence starts
q = torch.randn(138, 4, 64, dtype=torch.bfloat16, device="cuda")  # [tokens, heads, head dim]
k = torch.randn(138, 4, 64, dtype=torch.bfloat16, device="cuda")
v = torch.randn(138, 4, 64, dtype=torch.bfloat16, device="cuda")
g = -0.1 * torch.rand(138, 4, 64, device="cuda")  # log decay per key channel
beta = torch.rand(138, 4, device="cuda")  # write strength
state = 0.1 * torch.randn(3, 4, 64, 64, device="cuda")  # per sequence [heads, V, K]
do = torch.randn_like(v)

dq, dk, dv, dg, dbeta, d_state = cudnn.ops.kda_backward(q, k, v, g, beta, cu, do, initial_state=state, use_qk_l2norm=True)

expected = reference.gradients(lambda q, k, v, g, beta, state: reference.kda(q, k, v, g, beta, cu.tolist(), state), (q, k, v, g, beta, state), do)
error = max(reference.rms_error(x, ref) for x, ref in zip((dq, dk, dv, dg, dbeta, d_state), expected))
print(f"kda_backward: worst rms error {error:.2e} vs fp64 reference (dq, dk, dv, dg, dbeta, d_initial_state)")
assert error < 4e-2
