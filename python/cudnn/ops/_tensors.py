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
    """Run the plan for these tensors' shapes (creating it once), on the current stream, with a fresh workspace.

    ``tensors`` lists every operand of the op, None for an absent one, in the order the binding's
    ``execute_pointers`` takes them."""
    import torch

    key = (device.index, tuple(None if t is None else (t.dtype, t.shape, t.stride()) for t in tensors.values()), tuple(options.items()))
    plan = plans.get(key)
    if plan is None:
        plan = plans.setdefault(key, make_plan({name: describe(t) for name, t in tensors.items()}, options))
    from cudnn._torch_stream import _raw_current_stream

    stream = _raw_current_stream(torch, device)  # ~0.1 us; torch.cuda.current_stream() builds a Stream (~2.5 us)
    if stream is None:
        stream = torch.cuda.current_stream(device).cuda_stream
    workspace = torch.empty(max(plan.workspace_size, 1), dtype=torch.uint8, device=device)
    plan.execute_pointers(*(0 if t is None else t.data_ptr() for t in tensors.values()), workspace.data_ptr(), stream)
