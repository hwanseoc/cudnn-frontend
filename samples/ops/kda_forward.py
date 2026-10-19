# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Kimi Delta Attention over three packed sequences of different lengths, from Python."""

import torch

import cudnn
import reference

cu = torch.tensor([0, 100, 137, 437], dtype=torch.int32, device="cuda")  # where each sequence starts
q = torch.randn(437, 4, 128, dtype=torch.bfloat16, device="cuda")  # [tokens, heads, head dim]
k = torch.randn(437, 4, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(437, 4, 128, dtype=torch.bfloat16, device="cuda")
g = -0.1 * torch.rand(437, 4, 128, device="cuda")  # log decay per key channel
beta = torch.rand(437, 4, device="cuda")  # write strength
state = 0.1 * torch.randn(3, 4, 128, 128, device="cuda")  # per sequence [heads, V, K]

o, final_state = cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=state, output_final_state=True, use_qk_l2norm=True)

error = reference.rms_error(o, reference.kda(q, k, v, g, beta, cu.tolist(), state))
print(f"kda_forward: rms error {error:.2e} vs fp64 reference")
assert error < 2e-2
