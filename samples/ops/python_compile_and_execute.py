# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Python compiles and executes: SDPA and Kimi Delta Attention, forward and backward, over three packed sequences.

Each op compiles its kernel on first use and runs the same C++ op a C++ program would (cpp_execute_from_*.cpp).
"""

import torch

import cudnn
import reference

cu = torch.tensor([0, 100, 137, 437], dtype=torch.int32, device="cuda")  # three sequences: 100, 37, 300 tokens
lengths = cu.tolist()


def randn(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")


# SDPA: [tokens, heads, head dim]
q, k, v = randn(437, 4, 128), randn(437, 4, 128), randn(437, 4, 128)
o, stats = cudnn.ops.sdpa_forward(q, k, v, cu, cu, causal=True, return_stats=True)
do = torch.randn_like(o)
dq, dk, dv = cudnn.ops.sdpa_backward(q, k, v, o, do, stats, cu, cu, causal=True)

sdpa = lambda q, k, v: reference.sdpa(q, k, v, lengths, lengths, causal=True)  # noqa: E731
errors = {"sdpa_forward": reference.rms_error(o, sdpa(q, k, v))}
errors["sdpa_backward"] = max(map(reference.rms_error, (dq, dk, dv), reference.gradients(sdpa, (q, k, v), do)))

# KDA: q, k, v [tokens, heads, head dim]; g the log decay per key channel; beta the write strength; a state per sequence
q, k, v = randn(437, 4, 128), randn(437, 4, 128), randn(437, 4, 128)
g, beta = -0.1 * torch.rand(437, 4, 128, device="cuda"), torch.rand(437, 4, device="cuda")
state = 0.1 * torch.randn(3, 4, 128, 128, device="cuda")
o, final_state = cudnn.ops.kda_forward(q, k, v, g, beta, cu, initial_state=state, output_final_state=True, use_qk_l2norm=True)
do = torch.randn_like(o)
grads = cudnn.ops.kda_backward(q, k, v, g, beta, cu, do, initial_state=state, use_qk_l2norm=True)  # dq, dk, dv, dg, dbeta, d_state

kda = lambda q, k, v, g, beta, state: reference.kda(q, k, v, g, beta, lengths, state)  # noqa: E731
errors["kda_forward"] = reference.rms_error(o, kda(q, k, v, g, beta, state))
errors["kda_backward"] = max(map(reference.rms_error, grads, reference.gradients(kda, (q, k, v, g, beta, state), do)))

for op, error in errors.items():
    print(f"{op}: rms error {error:.2e} vs fp64 reference")
assert max(errors.values()) < 4e-2
