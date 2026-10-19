# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradients of attention over packed variable-length sequences (``cudnn_frontend::ops::SdpaBackward``).

The Python op runs the C++ op. Its kernel is compiled on first use for the
current GPU and registered for this process; :func:`cudnn.ops.compile` builds
the same kernels for C++ callers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import _library, _tensors

_PLANS: Dict[Any, Any] = {}


def sdpa_backward(
    q,
    k,
    v,
    o,
    do,
    stats,
    cu_seqlens_q,
    cu_seqlens_kv,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """``(dq, dk, dv)`` of :func:`cudnn.ops.sdpa_forward` for its ``q, k, v``, its ``o`` and ``stats``, and ``do``.

    Compact packed [T, H, 128] tensors with as many KV heads as query heads; ``stats`` is the forward's fp32 [Tq, H] LSE.
    ``max_seqlen_*`` bound any one sequence's length (default: the token count); a tighter bound shrinks the grid.
    """
    import torch

    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=stats, dq=dq, dk=dk, dv=dv, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)
    options = dict(scale=scale, causal=causal, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)
    _tensors.call(_PLANS, _plan, tensors, options, q.device)
    return dq, dk, dv


def _plan(descs, options):
    from cudnn import _compiled_module

    _library.ensure_registered(_compiled_module._sdpa_backward_config(descs, options))
    return _compiled_module._SdpaBackward(descs, options)


def _build(config: Dict[str, Any], target: "_library.Target"):
    """The FROST d128 packed backward host (setup, Delta, the dQ/dK/dV kernel, the dQ cast), compiled for the
    configuration's sequence capacity and head count: (compiled object, entry symbol, binding metadata)."""
    import cutlass

    from cudnn.frost.buffers import cutedsl_state
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_thd_host import compile_host

    version = (cutedsl_state()[1] or (None, "0"))[1]
    if tuple(int(x) for x in version.split(".")[:2] if x.isdigit()) < (4, 8):
        raise RuntimeError(f"cudnn.ops.sdpa_backward needs nvidia-cutlass-dsl >= 4.8; found {version}")
    if target.cc != 100:
        raise NotImplementedError(f"cudnn.ops.sdpa_backward runs on compute capability 10.0, not {target.cc}")
    dtype = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16}[config["dtype"]]
    none = (None,) * 9  # no ragged origins: rows are located from the offsets
    compiled = compile_host((config["max_seqs"], config["h"], True), none, none, dtype, config["causal"], 100)
    return compiled, "frost_sdpa_bwd_sm100_d128_thd_prepared", {}
