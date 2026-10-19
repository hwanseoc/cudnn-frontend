# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Plain-torch references for the cudnn.ops samples, in fp64. Not part of the API the samples show."""

import math

import torch


def rms_error(out, ref):
    """RMS(out - ref) / RMS(ref)."""
    out, ref = out.double(), ref.double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def sdpa(q, k, v, cu_q, cu_kv, causal):
    """Attention per sequence of packed [tokens, heads, dim] tensors; heads of k/v are shared by groups of q heads."""
    o = torch.empty(q.shape[0], q.shape[1], v.shape[2], dtype=torch.float64, device=q.device)
    group = q.shape[1] // k.shape[1]
    for i in range(len(cu_q) - 1):
        qs, ks = slice(cu_q[i], cu_q[i + 1]), slice(cu_kv[i], cu_kv[i + 1])
        qi = q[qs].double().transpose(0, 1)
        ki = k[ks].double().transpose(0, 1).repeat_interleave(group, 0)
        vi = v[ks].double().transpose(0, 1).repeat_interleave(group, 0)
        s = qi @ ki.transpose(1, 2) / math.sqrt(q.shape[2])
        if causal:  # query j sees keys 0..j of its sequence
            s = s.masked_fill(torch.ones(s.shape[1:], dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        o[qs] = (s.softmax(-1) @ vi).transpose(0, 1)
    return o


def kda(q, k, v, g, beta, cu, initial_state):
    """Kimi Delta Attention per sequence and head with l2-normalized q and k; the state is V-major [N, H, V, K]."""
    q = torch.nn.functional.normalize(q.double(), dim=-1) / math.sqrt(q.shape[2])
    k = torch.nn.functional.normalize(k.double(), dim=-1)
    v, alpha, beta = v.double(), g.double().exp(), beta.double()
    o = torch.empty_like(v)
    for i in range(len(cu) - 1):
        state = initial_state[i].double().clone()  # [H, V, K]
        for t in range(cu[i], cu[i + 1]):
            state = state * alpha[t][:, None, :]  # decay each key channel
            residual = v[t] - (state @ k[t][:, :, None])[..., 0]  # delta rule on the decayed state
            state = state + beta[t][:, None, None] * residual[:, :, None] * k[t][:, None, :]
            o[t] = (state @ q[t][:, :, None])[..., 0]
    return o
