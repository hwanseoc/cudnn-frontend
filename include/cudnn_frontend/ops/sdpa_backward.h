/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Gradients of scaled dot-product attention over packed variable-length
// sequences: dQ, dK and dV from Q, K, V, the forward's O and row log-sum-exp,
// and dO. Q-side tensors (q, o, d_o, dq, stats) share one SeqLayout and
// KV-side tensors (k, v, dk, dv) another. SM100, f16 or bf16, head dim 128,
// as many KV heads as query heads, compact rows.

#include "kernel_library.h"

#include <cmath>
#include <optional>

namespace cudnn_frontend {
namespace ops {

struct SdpaBackwardOptions {
    std::optional<float> scale;  // > 0; default 1 / sqrt(d); the forward's scale
    bool causal = false;         // top-left aligned, as in the forward
};

struct SdpaBackwardTensors {
    RaggedView q, k, v, o, d_o;  // [T, H, 128]; o and d_o on q's layout
    TensorView stats;            // [Tq, H] FLOAT, natural-log LSE from the forward; q's layout
    RaggedView dq, dk, dv;       // outputs, same shapes as q, k, v
};

class SdpaBackward {
   public:
    // Validates, lays out the workspace and takes the kernel from lib. Never
    // compiles: a kernel missing from lib is an error naming its configuration.
    static error_t
    create(SdpaBackwardTensors const &t, SdpaBackwardOptions const &opt, KernelLibrary const &lib, SdpaBackward &out) {
        SdpaBackward p;
        json config;
        CHECK_CUDNN_FRONTEND_ERROR(p.validate(t, opt, config));
        CHECK_CUDNN_FRONTEND_ERROR(lib.find(config, p.kernel_));
        out = std::move(p);
        return {error_code_t::OK, ""};
    }

    static error_t
    kernel_config(SdpaBackwardTensors const &t, SdpaBackwardOptions const &opt, std::string &config) {
        SdpaBackward p;
        json j;
        CHECK_CUDNN_FRONTEND_ERROR(p.validate(t, opt, j));
        config = j.dump();
        return {error_code_t::OK, ""};
    }

    // Bytes; the workspace must be 16-byte aligned and not shared by calls in flight.
    int64_t
    workspace_size() const {
        return workspace_size_;
    }

    error_t
    execute(SdpaBackwardTensors const &t, void *workspace, cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!same_layouts(t),
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "SdpaBackward::execute: tensors differ from create()");
        RETURN_CUDNN_FRONTEND_ERROR_IF(reinterpret_cast<uintptr_t>(workspace) % 16 != 0,
                                       error_code_t::INVALID_VALUE,
                                       "SdpaBackward::execute: the workspace must be 16-byte aligned");
        auto addr = [](void const *p) { return detail::any_int(static_cast<int64_t>(reinterpret_cast<intptr_t>(p))); };
        using detail::any_int;
        using detail::any_none;
        detail::ffi::Any const args[] = {
            any_int(tq_),
            any_int(tkv_),
            any_int(tq_),  // physical capacities: no ragged origins
            any_int(tkv_),
            any_int(max_sq_),
            any_int(max_skv_),
            any_int(0),  // token-major stats
            addr(t.q.values.data),
            addr(t.k.values.data),
            addr(t.v.values.data),
            addr(t.o.values.data),
            addr(t.d_o.values.data),
            addr(t.stats.data),
            addr(t.dq.values.data),
            addr(t.dk.values.data),
            addr(t.dv.values.data),
            addr(t.q.layout->offsets.data),
            addr(t.k.layout->offsets.data),
            any_none(),  // ragged origins of the nine ports: rows are located from the offsets
            any_none(),
            any_none(),
            any_none(),
            any_none(),
            any_none(),
            any_none(),
            any_none(),
            any_none(),
            addr(workspace),
            detail::any_float(scale_),
            any_int(3),  // both offsets arrays are [max_seqs + 1] prefixes
            addr(stream),
        };
        return kernel_->call(args, static_cast<int>(sizeof(args) / sizeof(args[0])));
    }

   private:
    error_t
    validate(SdpaBackwardTensors const &t, SdpaBackwardOptions const &opt, json &config) {
        SeqLayout const *ql = t.q.layout, *kl = t.k.layout;
        RETURN_CUDNN_FRONTEND_ERROR_IF(!ql || !kl || t.o.layout != ql || t.d_o.layout != ql || t.dq.layout != ql ||
                                           t.v.layout != kl || t.dk.layout != kl || t.dv.layout != kl,
                                       error_code_t::INVALID_VALUE,
                                       "SdpaBackward: q, o, d_o and dq share one SeqLayout, k, v, dk and dv another");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ql->max_seqs < 1 || kl->max_seqs != ql->max_seqs || ql->max_tokens < 1 || kl->max_tokens < 1,
            error_code_t::INVALID_VALUE,
            "SdpaBackward: both layouts describe the same max_seqs >= 1 and hold tokens");
        for (auto const *l : {ql, kl}) {
            auto const &cu = l->offsets;
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                cu.dtype != DataType_t::INT32 || cu.rank != 1 || cu.dim[0] != l->max_seqs + 1 || cu.stride[0] != 1,
                error_code_t::INVALID_VALUE,
                "SdpaBackward: offsets are contiguous int32 [max_seqs + 1]");
        }
        int64_t const H = t.q.values.dim[1], TQ = ql->max_tokens, TKV = kl->max_tokens;
        DataType_t const dt = t.q.values.dtype;
        RETURN_CUDNN_FRONTEND_ERROR_IF(dt != DataType_t::HALF && dt != DataType_t::BFLOAT16,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "SdpaBackward: tensors are HALF or BFLOAT16");
        auto packed = [&](TensorView const &x, int64_t tokens) {  // compact [tokens, H, 128]
            return x.dtype == dt && x.rank == 3 && x.dim[0] == tokens && x.dim[1] == H && x.dim[2] == 128 &&
                   x.stride[0] == H * 128 && x.stride[1] == 128 && x.stride[2] == 1 &&
                   reinterpret_cast<uintptr_t>(x.data) % 16 == 0;
        };
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !packed(t.q.values, TQ) || !packed(t.o.values, TQ) || !packed(t.d_o.values, TQ) ||
                !packed(t.dq.values, TQ) || !packed(t.k.values, TKV) || !packed(t.v.values, TKV) ||
                !packed(t.dk.values, TKV) || !packed(t.dv.values, TKV),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "SdpaBackward: every tensor is compact [tokens, H, 128] in one dtype, 16-byte aligned, with as many KV "
            "heads as query heads");
        auto const &s = t.stats;
        RETURN_CUDNN_FRONTEND_ERROR_IF(s.dtype != DataType_t::FLOAT || s.rank != 2 || s.dim[0] != TQ || s.dim[1] != H ||
                                           s.stride[0] != H || s.stride[1] != 1,
                                       error_code_t::INVALID_VALUE,
                                       "SdpaBackward: stats is contiguous FLOAT [Tq, H]");
        scale_ = opt.scale.value_or(static_cast<float>(1.0 / std::sqrt(128.0)));
        RETURN_CUDNN_FRONTEND_ERROR_IF(!(scale_ > 0.0f) || !std::isfinite(scale_ * 1.4426950408889634f),
                                       error_code_t::INVALID_VALUE,
                                       "SdpaBackward: scale must be positive and finite");

        int64_t const B = ql->max_seqs;
        // The kernel is specialized on the sequence capacity and the head count.
        config = {
            {"op", "sdpa_backward"},
            {"dtype", dt == DataType_t::HALF ? "fp16" : "bf16"},
            {"d", 128},
            {"h", H},
            {"max_seqs", B},
            {"causal", opt.causal},
        };
        tq_      = TQ;
        tkv_     = TKV;
        max_sq_  = ql->max_seq_len > 0 ? ql->max_seq_len : TQ;
        max_skv_ = kl->max_seq_len > 0 ? kl->max_seq_len : TKV;
        // dQ accumulator and Delta (fp32), per-sequence metadata, origins and TMA maps, from a 256-byte aligned base.
        auto align         = [](int64_t n) { return (n + 255) / 256 * 256; };
        int64_t const rows = (TQ + 127) / 128 * 128;
        int64_t const meta = 4 * rows * H * 128 + 4 * rows * H;
        int64_t const maps = meta + align(4 * (4 * B + 4)) + align(8 * 9 * B);
        workspace_size_    = maps + 7 * B * 128 + 240;
        seen_              = {*ql,
                              *kl,
                              {t.q.values,
                               t.k.values,
                               t.v.values,
                               t.o.values,
                               t.d_o.values,
                               t.dq.values,
                               t.dk.values,
                               t.dv.values,
                               t.stats}};
        return {error_code_t::OK, ""};
    }

    static bool
    same(TensorView const &a, TensorView const &b) {
        return a.dtype == b.dtype && a.rank == b.rank && a.dim == b.dim && a.stride == b.stride;
    }

    bool
    same_layouts(SdpaBackwardTensors const &t) const {
        auto layout = [](SeqLayout const *a, SeqLayout const &b) {
            return a && a->max_seqs == b.max_seqs && a->max_tokens == b.max_tokens && a->max_seq_len == b.max_seq_len &&
                   same(a->offsets, b.offsets);
        };
        TensorView const *now[] = {&t.q.values,
                                   &t.k.values,
                                   &t.v.values,
                                   &t.o.values,
                                   &t.d_o.values,
                                   &t.dq.values,
                                   &t.dk.values,
                                   &t.dv.values,
                                   &t.stats};
        for (int i = 0; i < 9; i++) {
            if (!same(*now[i], seen_.dense[i])) return false;
        }
        return layout(t.q.layout, seen_.q_layout) && layout(t.k.layout, seen_.kv_layout);
    }

    struct Seen {
        SeqLayout q_layout, kv_layout;
        std::array<TensorView, 9> dense;  // q, k, v, o, d_o, dq, dk, dv, stats
    } seen_;
    std::shared_ptr<detail::Kernel const> kernel_;
    int64_t tq_ = 0, tkv_ = 0, max_sq_ = 0, max_skv_ = 0, workspace_size_ = 0;
    float scale_ = 1.0f;
};

}  // namespace ops
}  // namespace cudnn_frontend
