# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradients of Kimi Delta Attention over packed variable-length sequences (``cudnn_frontend::ops::KdaBackward``).

The Python op runs the C++ op. Its kernel is compiled on first use for the
current GPU and registered for this process; :func:`cudnn.ops.compile` builds
the same kernels for C++ callers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import _library, _tensors

_PLANS: Dict[Any, Any] = {}


def kda_backward(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    do,
    *,
    scale: Optional[float] = None,
    initial_state=None,
    d_final_state=None,
    use_qk_l2norm: bool = False,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    gate_domain: str = "log",
    gate_lower_bound: Optional[float] = None,
) -> Tuple[Any, Any, Any, Any, Any, Optional[Any]]:
    """``(dq, dk, dv, dg, dbeta, d_initial_state)`` of :func:`cudnn.ops.kda_forward` for the same inputs and options,
    given ``do`` and optionally the final state's gradient. ``d_initial_state`` is None without an ``initial_state``.
    q, k and v have the same head count."""
    import torch

    dq, dk, dv, dg, dbeta = (torch.empty_like(t) for t in (q, k, v, g, beta))
    d_initial_state = torch.empty_like(initial_state) if initial_state is not None else None
    tensors = dict(
        q=q, k=k, v=v, g=g, beta=beta, do=do, dq=dq, dk=dk, dv=dv, dg=dg, dbeta=dbeta, cu_seqlens=cu_seqlens,
        initial_state=initial_state, d_final_state=d_final_state, d_initial_state=d_initial_state,
    )  # fmt: skip
    options = dict(
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        gate_domain=gate_domain,
        gate_lower_bound=gate_lower_bound,
    )
    _tensors.call(_PLANS, _plan, tensors, options, q.device)
    return dq, dk, dv, dg, dbeta, d_initial_state


def _plan(descs, options):
    from cudnn import _compiled_module

    _library.ensure_registered(_compiled_module._kda_backward_config(descs, options))
    return _compiled_module._KdaBackward(descs, options)


def _build(config: Dict[str, Any], target: "_library.Target"):
    """The FROST KDA engine's uncut backward (state recompute, then the gradient pass), traced on placeholder tensors:
    (compiled object, entry symbol, binding metadata)."""
    import cutlass
    import cutlass.cute as cute

    from cudnn.linear_attention.frost.common.host import tensormap_workspace_bytes
    from cudnn.linear_attention.frost.kernel import kda_bprop_f16, kda_recompute_f16
    from cudnn.linear_attention.frost.kernel.kda_uncut_backward_f16 import build_configs, uncut_backward_host

    dtypes = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16, "fp32": cutlass.Float32, "int32": cutlass.Int32, "int64": cutlass.Int64}
    io, g, beta = dtypes[config["io"]], dtypes[config["g"]], dtypes[config["beta"]]
    state = dtypes.get(config["state"], cutlass.Float32)

    def tensor(dtype, rank, align, present=True):
        if not present:
            return None
        shape = tuple(cute.sym_int32() for _ in range(rank))
        stride = tuple(cute.sym_int64() for _ in range(rank - 1)) + (1,)
        return cute.runtime.make_fake_tensor(dtype, shape, stride, assumed_align=align)

    def items():
        return cute.runtime.make_fake_tensor(cutlass.Int32, (cute.sym_int32(), 10), (10, 1), assumed_align=16)

    has_s0 = config["state"] != "none"
    recompute_cfg, bprop_cfg = build_configs(
        io,
        state,
        g,
        recompute=True,
        coarse=False,
        has_state_in=has_s0,
        use_initial_state=has_s0,
        use_dstate_in=config["d_final_state"] != "none",
        use_dstate0=config["d_initial_state"] != "none",
        l2norm=config["use_qk_l2norm"],
        safe_gate=False,
        gate_scale_log2=float(config["gate_lower_bound"]) * kda_bprop_f16.LOG2_E,
        log_gate=config["log_gate"],
        beta_sigmoid=config["use_beta_sigmoid"],
        allow_neg_eigval=config["allow_neg_eigval"],
        max_active_clusters=target.num_sm,
        d_k=config["d_k"],
        d_v=config["d_v"],
    )
    compiled = cute.compile(
        uncut_backward_host,
        kda_bprop_f16.CFG.B_T,
        io,
        True,  # recompute the state series
        True,  # the recompute prologue orders the work table
        False,  # coarse
        False,  # bwd_orders
        recompute_cfg,
        bprop_cfg,
        cutlass.Int32(kda_bprop_f16.CFG.B_T),
        cutlass.Int32(0),
        cutlass.Int32(0),
        1.0,  # scale (a runtime argument)
        *[tensor(io, 3, 16) for _ in range(7)],  # q, k, v, do, dq, dk, dv
        tensor(g, 3, 16),  # gate
        tensor(g, 3, 4, not config["log_gate"]),  # gate_main
        tensor(beta, 2, 4),
        None,  # a_log
        None,  # dt_bias
        tensor(dtypes[config["cu_seqlens"]], 1, 8 if config["cu_seqlens"] == "int64" else 4),
        tensor(io, 4, 16),  # the recomputed state series
        None,  # seed_checkpoints
        tensor(state, 4, 16, has_s0),
        tensor(g, 3, 16),  # dgate
        tensor(beta, 2, 4),  # dbeta
        tensor(state, 4, 16, config["d_initial_state"] != "none"),
        tensor(state, 4, 16, config["d_final_state"] != "none"),
        items(),  # work items
        tensor(cutlass.Int32, 1, 4),  # work count
        items(),  # series items
        tensor(cutlass.Int32, 1, 4),  # series count
        tensor(cutlass.Int32, 1, 4),  # scheduler_all
        tensor(cutlass.Int32, 1, 4),  # scheduler_all, for the recompute
        None,
        tensor(cutlass.Int32, 1, 4),  # scheduler_recompute
        tensor(cutlass.Int32, 1, 4),  # scheduler_bwd
        tensor(cutlass.Int64, 1, 128),  # recompute TMA descriptors
        tensor(cutlass.Int64, 1, 128),  # bprop TMA descriptors
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi --opt-level 2",
    )

    def sizing(module):
        base = tensormap_workspace_bytes(module, 0)
        return {"per_seq": tensormap_workspace_bytes(module, 1) - base, "base": base}

    meta = {"recompute_tensormap_bytes": sizing(kda_recompute_f16), "bprop_tensormap_bytes": sizing(kda_bprop_f16)}
    return compiled, "cudnn_ops_kda_backward", meta
