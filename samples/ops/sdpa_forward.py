# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Causal attention over three packed sequences of different lengths, from Python."""

import torch

import cudnn
import reference

cu = torch.tensor([0, 100, 137, 437], dtype=torch.int32, device="cuda")  # where each sequence starts
q = torch.randn(437, 8, 128, dtype=torch.bfloat16, device="cuda")  # [tokens, heads, head dim]
k = torch.randn(437, 2, 128, dtype=torch.bfloat16, device="cuda")  # 2 KV heads, each shared by 4 query heads
v = torch.randn(437, 2, 128, dtype=torch.bfloat16, device="cuda")

o, stats = cudnn.ops.sdpa_forward(q, k, v, cu, cu, causal=True, return_stats=True)

error = reference.rms_error(o, reference.sdpa(q, k, v, cu.tolist(), cu.tolist(), causal=True))
print(f"sdpa_forward: rms error {error:.2e} vs fp64 reference")
assert error < 2e-2
