# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scaled dot-product attention forward over packed variable-length sequences (``cudnn_frontend::ops::SdpaForward``).

The Python op runs the C++ op. Its kernel is compiled on first use for the
current GPU and registered for this process; :func:`cudnn.ops.compile` builds
the same kernels for C++ callers.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

from . import _library, _tensors

_PLANS: Dict[Any, Any] = {}


def sdpa_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    return_stats: bool = False,
) -> Tuple[Any, Optional[Any]]:
    """Attention per sequence. Returns ``(o, stats)``; ``stats`` is None unless ``return_stats``.

    ``q`` [Tq, HQ, Dqk], ``k`` [Tkv, HKV, Dqk], ``v`` [Tkv, HKV, Dv], ``cu_seqlens_q`` / ``cu_seqlens_kv`` int32 [N + 1].
    ``o`` is [Tq, HQ, Dv] in q's dtype; ``stats`` is the fp32 [Tq, HQ] natural-log row log-sum-exp.
    ``causal`` is top-left aligned: query i of a sequence sees keys 0..i.
    """
    import torch

    o = torch.empty(q.shape[0], q.shape[1], v.shape[2], dtype=q.dtype, device=q.device)
    stats = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device) if return_stats else None
    tensors = dict(q=q, k=k, v=v, o=o, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, stats=stats)
    _tensors.call(_PLANS, _plan, tensors, dict(scale=scale, causal=causal), q.device)
    return o, stats


def sdpa_forward_config(tensors: Dict[str, Any], options: Dict[str, Any]) -> str:
    """The kernel configuration ``SdpaForward::create`` looks up for these tensors (only their dtypes, shapes, strides)."""
    from cudnn import _compiled_module

    return _compiled_module._sdpa_forward_config({name: _tensors.describe(t) for name, t in tensors.items()}, options)


def _plan(descs, options):
    from cudnn import _compiled_module

    _library.ensure_registered(_compiled_module._sdpa_forward_config(descs, options))
    return _compiled_module._SdpaForward(descs, options)


# ---------------------------------------------------------------------------
# The kernel: the FROST engine's prepared THD launch, taken from a sample graph
# ---------------------------------------------------------------------------

_SAMPLE_SEQS, _SAMPLE_LEN = 2, 256  # >= 2 sequences: one sequence selects a single-batch specialization


def _build(config: Dict[str, Any], target: "_library.Target"):
    """Compile the kernel ``config`` names: (compiled object, entry symbol, binding metadata).

    Builds a sample packed graph on the FROST SDPA engine and keeps its prepared THD launch: the kernel is shape-generic
    (every token count, sequence count and length binds per call), so the sample's shape does not reach it."""
    import cudnn
    from cudnn.sdpa.fwd.engines import engine_name

    dt = {"fp16": cudnn.data_type.HALF, "bf16": cudnn.data_type.BFLOAT16}[config["dtype"]]
    b, s = _SAMPLE_SEQS, _SAMPLE_LEN
    hq, hk, dqk, dv = config["h_q"], config["h_kv"], config["d_qk"], config["d_v"]
    sample_scale = 1.0 / math.sqrt(dqk)
    i32 = cudnn.data_type.INT32

    g = cudnn.pygraph(io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def packed(name, h, d):
        return g.tensor(dim=[b, h, s, d], stride=[s * h * d, d, h * d, 1], data_type=dt, name=name)

    def offsets(name):
        return g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name=name)

    q, k, v = packed("q", hq, dqk), packed("k", hk, dqk), packed("v", hk, dv)
    off_q, off_kv = offsets("off_q"), offsets("off_kv")
    q.set_ragged_offset(off_q)
    k.set_ragged_offset(off_kv)
    v.set_ragged_offset(off_kv)
    o, lse = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=config["stats"],
        attn_scale=sample_scale,
        use_causal_mask=config["causal"],
        use_padding_mask=True,
        cu_seq_len_q=offsets("cu_q"),
        cu_seq_len_kv=offsets("cu_kv"),
        max_total_seq_len_q=b * s,
        max_total_seq_len_kv=b * s,
    )
    o.set_output(True).set_dim([b, hq, s, dv]).set_stride([s * hq * dv, dv, hq * dv, 1]).set_ragged_offset(off_q)
    if config["stats"]:
        lse.set_output(True).set_dim([b, hq, s, 1]).set_stride([s * hq, 1, hq, 1]).set_data_type(cudnn.data_type.FLOAT)
        lse.set_ragged_offset(offsets("off_lse"))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    want = engine_name(arch="sm100")
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next((i for i, n in enumerate(names) if n == want or n.startswith(want + "[")), None)
    if index is None:
        raise NotImplementedError(f"cudnn.ops.sdpa_forward: the FROST engine does not serve {config} on this GPU")
    g.select_plan(index)
    g.check_support()
    g.build_plans()

    plan = g._compiled_plans[g._plan_index]
    spec = getattr(getattr(plan, "_prepared", None), "spec", None)
    if spec is None or getattr(spec, "split_workspace", None) is not None or spec.has_sink or spec.paged:
        raise NotImplementedError(f"cudnn.ops.sdpa_forward: {config} has no plain prepared THD launch")

    # The workspace layout the C++ op computes from capacities (sdpa_forward.h): check it against the engine's.
    align = lambda n: -(-n // 128) * 128  # noqa: E731
    off_o_desc = align(4 * (4 * b + 4))
    if spec.off_o_desc != off_o_desc or spec.scratch_bytes != off_o_desc + align((b + 3) * 128) + align(hq * 4):
        raise NotImplementedError(f"cudnn.ops.sdpa_forward: the THD workspace layout changed ({spec.off_o_desc}, {spec.scratch_bytes})")

    # The engine's adapter, for the kernel module's launch constants (a persistent kernel's resident-cluster cap).
    closure = dict(zip(plan._compiled.__code__.co_freevars, (c.cell_contents for c in plan._compiled.__closure__ or ())))
    km = closure["api"]._k_mod
    max_units = 0
    if getattr(km, "THD_PERSISTENT", False):
        clusters = int(getattr(km, "CGA_SIZE", 0) or getattr(km, "CTA_MMA", 1))
        max_units = max(1, target.num_sm // max(1, clusters))

    def plain(x):
        if x is None or isinstance(x, (bool, int, float)):
            return x
        if isinstance(x, tuple) and all(isinstance(i, int) for i in x):
            return list(x)
        raise NotImplementedError(f"cudnn.ops.sdpa_forward: host argument {x!r} has no JSON form")

    meta = {
        "order": list(spec.order),
        "template": [plain(x) for x in spec.template],
        "cga_tile_m": int(spec.cga_tile_m),
        "max_units": max_units,
        "scale_log2_factor": spec.template[spec.order.index("scale_softmax_log2")] / sample_scale,
    }
    compiled = spec.owner
    entry = getattr(compiled, "_compiled_cache_entry", None)
    symbol = json.load(open(os.path.join(entry, "entry.json")))["symbol"] if entry else "frost_sdpa_fwd"
    return compiled, symbol, meta
