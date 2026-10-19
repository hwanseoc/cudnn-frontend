# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi Delta Attention forward over packed variable-length sequences (``cudnn_frontend::ops::KdaForward``).

The Python op runs the C++ op. Its kernel is compiled on first use for the
current GPU and registered for this process; :func:`cudnn.ops.compile` builds
the same kernels for C++ callers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import _library, _tensors

_PLANS: Dict[Any, Any] = {}


def kda_forward(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    scale: Optional[float] = None,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_domain: str = "log",
    gate_lower_bound: Optional[float] = None,
    a_log=None,
    dt_bias=None,
) -> Tuple[Any, Optional[Any]]:
    """KDA forward. Returns ``(o, final_state)``; ``final_state`` is None unless ``output_final_state``.

    ``q`` [T, HQ, K], ``k`` [T, HK, K], ``v`` [T, HV, V], ``g`` [T, HO, K] (log decay), ``beta`` [T, HO],
    ``cu_seqlens`` int32/int64 [N + 1], HO = max(HQ, HV); ``initial_state`` [N, HO, V, K] fp32/bf16.
    ``o`` is [T, HO, V] in q's dtype; ``final_state`` has the initial state's dtype, else fp32.
    """
    import torch

    T, HO, V, K = q.shape[0], max(q.shape[1], v.shape[1]), v.shape[2], q.shape[2]
    n = cu_seqlens.shape[0] - 1
    o = torch.empty(T, HO, V, dtype=q.dtype, device=q.device)
    final_state = None
    if output_final_state:
        state_dtype = initial_state.dtype if initial_state is not None else torch.float32
        final_state = torch.empty(n, HO, V, K, dtype=state_dtype, device=q.device)
    tensors = dict(q=q, k=k, v=v, g=g, beta=beta, o=o, cu_seqlens=cu_seqlens, initial_state=initial_state, final_state=final_state)
    tensors.update(a_log=a_log if safe_gate else None, dt_bias=dt_bias if safe_gate else None)
    options = dict(
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        gate_lower_bound=gate_lower_bound,
    )
    _tensors.call(_PLANS, _plan, tensors, options, q.device)
    return o, final_state


def kda_forward_config(tensors: Dict[str, Any], options: Dict[str, Any]) -> str:
    """The kernel configuration ``KdaForward::create`` looks up for these tensors (only their dtypes, shapes, strides)."""
    from cudnn import _compiled_module

    return _compiled_module._kda_forward_config({name: _tensors.describe(t) for name, t in tensors.items()}, options)


def _plan(descs, options):
    from cudnn import _compiled_module

    _library.ensure_registered(_compiled_module._kda_forward_config(descs, options))
    return _compiled_module._KdaForward(descs, options)


# ---------------------------------------------------------------------------
# The kernel: the FROST KDA engine's uncut forward, traced on placeholder tensors
# ---------------------------------------------------------------------------


def _build(config: Dict[str, Any], target: "_library.Target"):
    """Compile the kernel ``config`` names for ``target``: (compiled object, entry symbol, binding metadata)."""
    import cutlass
    import cutlass.cute as cute

    from cudnn.linear_attention.frost.common.host import tensormap_workspace_bytes
    from cudnn.linear_attention.frost.kernel import kda_prefill_f16
    from cudnn.linear_attention.frost.kernel.kda_uncut_forward_f16 import build_configs, uncut_forward_host

    dtypes = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16, "fp32": cutlass.Float32, "int32": cutlass.Int32, "int64": cutlass.Int64}
    io = dtypes[config["io"]]
    state = dtypes.get(config["state"], cutlass.Float32)

    def tensor(dtype, rank, align, present=True):
        # A placeholder with every extent and stride dynamic but the innermost, which is 1:
        # what from_dlpack(t).mark_layout_dynamic(leading_dim=rank - 1) traces.
        if not present:
            return None
        shape = tuple(cute.sym_int32() for _ in range(rank))
        stride = tuple(cute.sym_int64() for _ in range(rank - 1)) + (1,)
        return cute.runtime.make_fake_tensor(dtypes[dtype] if isinstance(dtype, str) else dtype, shape, stride, assumed_align=align)

    prefill_cfg, _ = build_configs(
        io,
        state,
        dtypes[config["g"]],
        use_initial_state=config["initial_state"],
        store_final_state=config["final_state"],
        enable_checkpoints=False,
        l2norm=config["use_qk_l2norm"],
        safe_gate=config["safe_gate"],
        gate_scale_log2=float(config["gate_lower_bound"]) * kda_prefill_f16.LOG2_E,
        log_gate=config["log_gate"],
        beta_sigmoid=config["use_beta_sigmoid"],
        allow_neg_eigval=config["allow_neg_eigval"],
        max_active_clusters=target.num_sm,
        d_k=config["d_k"],
        d_v=config["d_v"],
        tiles_per_head=1,
        prep=False,
    )
    safe = config["safe_gate"]
    compiled = cute.compile(
        uncut_forward_host,
        kda_prefill_f16.CFG.B_T,
        target.num_sm,
        io,
        prefill_cfg,
        1,  # tiles_per_head
        cutlass.Int32(0),  # checkpoint_every_n
        1.0,  # scale (a runtime argument)
        tensor(io, 3, 16),  # q
        tensor(io, 3, 16),  # k
        tensor(io, 3, 16),  # v
        tensor(config["g"], 3, 16),
        tensor(config["a_log"], 1, 4, safe),
        tensor(config["dt_bias"], 2, 16, safe),
        tensor(config["beta"], 2, 4),
        tensor(io, 3, 16),  # o
        tensor(config["cu_seqlens"], 1, 8 if config["cu_seqlens"] == "int64" else 4),
        tensor(state, 4, 16, config["initial_state"]),
        tensor(state, 4, 16, config["final_state"]),
        None,  # seed_indices
        None,  # final_indices
        None,  # checkpoints
        cute.runtime.make_fake_tensor(cutlass.Int32, (cute.sym_int32(), 10), (10, 1), assumed_align=16),  # work items
        tensor(cutlass.Int32, 1, 4),  # work count
        tensor(cutlass.Int32, 1, 4),  # scheduler
        tensor(cutlass.Int64, 1, 128),  # TMA descriptors
        False,  # prep
        None,
        *[None] * 8,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi --opt-level 2",
    )
    base = tensormap_workspace_bytes(kda_prefill_f16, 0)
    meta = {"tensormap_bytes": {"per_seq": tensormap_workspace_bytes(kda_prefill_f16, 1) - base, "base": base}}
    return compiled, "cudnn_ops_kda_forward", meta
