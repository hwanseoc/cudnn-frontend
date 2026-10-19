# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch tensors as the (address, cudnn.data_type, shape, strides) tuples the ops bindings take."""

from __future__ import annotations


def describe(t):
    """``t`` as a binding tensor tuple, or None."""
    if t is None:
        return None
    return (t.data_ptr(), data_type(t.dtype), tuple(t.shape), tuple(t.stride()))


def signature(desc):
    """What a plan is keyed by: everything but the address."""
    return None if desc is None else desc[1:]


def data_type(dtype):
    import torch

    import cudnn

    table = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
    }
    if dtype not in table:
        raise TypeError(f"cudnn.ops: unsupported tensor dtype {dtype}")
    return table[dtype]


def call(plans, make_plan, tensors, options, device):
    """Run the plan for these tensors' shapes (creating it once), on the current stream, with a fresh workspace."""
    import torch

    descs = {name: describe(t) for name, t in tensors.items()}
    key = (tuple((name, signature(d)) for name, d in descs.items()), tuple(sorted(options.items())))
    plan = plans.get(key)
    if plan is None:
        plan = plans.setdefault(key, make_plan(descs, options))
    workspace = torch.empty(max(plan.workspace_size, 1), dtype=torch.uint8, device=device)
    plan.execute(descs, workspace.data_ptr(), torch.cuda.current_stream(device).cuda_stream)
