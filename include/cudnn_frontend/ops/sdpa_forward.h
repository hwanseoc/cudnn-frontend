/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Scaled dot-product attention forward over packed variable-length sequences:
// for every sequence, o = softmax(scale * q k^T [causal]) v, and optionally its
// row log-sum-exp. Q (and O) and KV are packed [tokens, heads, dim] with one
// SeqLayout each; both describe the same number of sequences. SM100 / SM103,
// f16 or bf16.

#include "kernel_library.h"

#include <cmath>
#include <optional>

namespace cudnn_frontend {
namespace ops {

struct SdpaForwardOptions {
    std::optional<float> scale;  // > 0; default 1 / sqrt(d_qk)
    bool causal = false;         // top-left aligned: query i of a sequence sees keys 0..i
};

struct SdpaForwardTensors {
    RaggedView q;                       // [Tq, HQ, Dqk]
    RaggedView k;                       // [Tkv, HKV, Dqk], HQ a multiple of HKV
    RaggedView v;                       // [Tkv, HKV, Dv]
    RaggedView o;                       // [Tq, HQ, Dv], q's layout
    TensorView const *stats = nullptr;  // [Tq, HQ] FLOAT, natural-log LSE of each row; q's layout
};

class SdpaForward {
   public:
    SdpaForward()               = default;
    SdpaForward(SdpaForward &&) = default;
    SdpaForward &
    operator=(SdpaForward &&) = default;

    // Validates, lays out the workspace and takes the kernel from lib. Never
    // compiles: a kernel missing from lib is an error naming its configuration.
    static error_t
    create(SdpaForwardTensors const &t, SdpaForwardOptions const &opt, KernelLibrary const &lib, SdpaForward &out) {
        SdpaForward p;
        json config;
        CHECK_CUDNN_FRONTEND_ERROR(p.validate(t, opt, config));
        CHECK_CUDNN_FRONTEND_ERROR(lib.find(config, p.kernel_));
        CHECK_CUDNN_FRONTEND_ERROR(p.bind_plan(opt));
        out = std::move(p);
        return {error_code_t::OK, ""};
    }

    // The configuration create() looks up, to compile it ahead of time.
    static error_t
    kernel_config(SdpaForwardTensors const &t, SdpaForwardOptions const &opt, std::string &config) {
        SdpaForward p;
        json j;
        CHECK_CUDNN_FRONTEND_ERROR(p.validate(t, opt, j));
        config = j.dump();
        return {error_code_t::OK, ""};
    }

    // Bytes; the workspace must be 128-byte aligned and not shared by calls in flight.
    int64_t
    workspace_size() const {
        return workspace_size_;
    }

    // Shapes, strides and dtypes must match create(); sequence lengths are read on the device.
    // No allocation, no synchronization.
    error_t
    execute(SdpaForwardTensors const &t, void *workspace, cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !same_layouts(t), error_code_t::INVALID_VARIANT_PACK, "SdpaForward::execute: tensors differ from create()");
        RETURN_CUDNN_FRONTEND_ERROR_IF(reinterpret_cast<uintptr_t>(workspace) % 128 != 0,
                                       error_code_t::INVALID_VALUE,
                                       "SdpaForward::execute: the workspace must be 128-byte aligned");
        constexpr size_t MAX_ARGS = 64;
        detail::ffi::Any args[MAX_ARGS];
        std::copy(proto_.begin(), proto_.end(), args);
        auto addr = [](void const *p) { return detail::any_int(static_cast<int64_t>(reinterpret_cast<intptr_t>(p))); };
        auto *ws  = static_cast<char *>(workspace);
        args[slot_.q]       = addr(t.q.values.data);
        args[slot_.k]       = addr(t.k.values.data);
        args[slot_.v]       = addr(t.v.values.data);
        args[slot_.o]       = addr(t.o.values.data);
        args[slot_.q_lens]  = addr(t.q.layout->offsets.data);
        args[slot_.kv_lens] = addr(t.k.layout->offsets.data);
        args[slot_.meta]    = addr(ws);
        args[slot_.o_desc]  = addr(ws + off_o_desc_);
        args[slot_.sinks]   = detail::any_int(0);
        args[slot_.stream]  = addr(stream);
        if (has_stats_) {
            args[slot_.lse] = addr(t.stats->data);
        }
        return kernel_->call(args, static_cast<int>(proto_.size()));
    }

   private:
    error_t
    validate(SdpaForwardTensors const &t, SdpaForwardOptions const &opt, json &config) {
        SeqLayout const *ql = t.q.layout, *kl = t.k.layout;
        RETURN_CUDNN_FRONTEND_ERROR_IF(!ql || !kl || t.o.layout != ql || t.v.layout != kl,
                                       error_code_t::INVALID_VALUE,
                                       "SdpaForward: q and o share one SeqLayout, k and v another");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ql->max_seqs < 1 || kl->max_seqs != ql->max_seqs || ql->max_tokens < 1 || kl->max_tokens < 1,
            error_code_t::INVALID_VALUE,
            "SdpaForward: both layouts describe the same max_seqs >= 1 and hold at least one token");
        for (auto const *l : {ql, kl}) {
            auto const &cu = l->offsets;
            RETURN_CUDNN_FRONTEND_ERROR_IF(cu.dtype != DataType_t::INT32 || cu.rank != 1 ||
                                               cu.dim[0] != l->max_seqs + 1 || cu.stride[0] != 1 ||
                                               reinterpret_cast<uintptr_t>(cu.data) % 4 != 0,
                                           error_code_t::INVALID_VALUE,
                                           "SdpaForward: offsets are contiguous int32 [max_seqs + 1]");
        }
        auto const &q = t.q.values, &k = t.k.values, &v = t.v.values, &o = t.o.values;
        RETURN_CUDNN_FRONTEND_ERROR_IF(q.rank != 3 || k.rank != 3 || v.rank != 3 || o.rank != 3,
                                       error_code_t::INVALID_VALUE,
                                       "SdpaForward: q, k, v and o are [tokens, heads, dim]");
        int64_t const HQ = q.dim[1], HKV = k.dim[1], DQK = q.dim[2], DV = v.dim[2];
        int64_t const TQ = ql->max_tokens, TKV = kl->max_tokens;
        RETURN_CUDNN_FRONTEND_ERROR_IF(q.dim[0] != TQ || o.dim[0] != TQ || k.dim[0] != TKV || v.dim[0] != TKV ||
                                           k.dim[2] != DQK || v.dim[1] != HKV || o.dim[1] != HQ || o.dim[2] != DV ||
                                           HKV < 1 || HQ % HKV != 0,
                                       error_code_t::INVALID_VALUE,
                                       "SdpaForward: shapes must be q [Tq,HQ,Dqk], k [Tkv,HKV,Dqk], v [Tkv,HKV,Dv], "
                                       "o [Tq,HQ,Dv], with Tq and Tkv the layouts' max_tokens and HKV dividing HQ");
        DataType_t const dt = q.dtype;
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (dt != DataType_t::HALF && dt != DataType_t::BFLOAT16) || k.dtype != dt || v.dtype != dt || o.dtype != dt,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "SdpaForward: q, k, v and o must all be HALF or all BFLOAT16");
        for (TensorView const *x : {&q, &k, &v, &o}) {
            int64_t const h = x->dim[1], d = x->dim[2], hs = x->stride[1], ts = x->stride[0];
            RETURN_CUDNN_FRONTEND_ERROR_IF(x->stride[2] != 1 || hs < d || hs % 8 != 0 || ts < (h - 1) * hs + d ||
                                               ts % 8 != 0 || reinterpret_cast<uintptr_t>(x->data) % 16 != 0,
                                           error_code_t::INVALID_VALUE,
                                           "SdpaForward: head dims contiguous, head and token strides covering and "
                                           "16-byte multiples, buffers 16-byte aligned");
        }
        if (t.stats) {
            auto const &s = *t.stats;
            RETURN_CUDNN_FRONTEND_ERROR_IF(s.dtype != DataType_t::FLOAT || s.rank != 2 || s.dim[0] != TQ ||
                                               s.dim[1] != HQ || s.stride[0] != HQ || s.stride[1] != 1 ||
                                               reinterpret_cast<uintptr_t>(s.data) % 4 != 0,
                                           error_code_t::INVALID_VALUE,
                                           "SdpaForward: stats is contiguous FLOAT [Tq, HQ]");
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            opt.scale && !(*opt.scale > 0.0f), error_code_t::INVALID_VALUE, "SdpaForward: scale must be positive");

        config = {
            {"op", "sdpa_forward"},
            {"dtype", dt == DataType_t::HALF ? "fp16" : "bf16"},
            {"d_qk", DQK},
            {"d_v", DV},
            {"h_q", HQ},
            {"h_kv", HKV},
            {"causal", opt.causal},
            {"stats", t.stats != nullptr},
        };
        has_stats_ = t.stats != nullptr;
        seen_      = {*ql, *kl, {q, k, v, o}, {}};
        if (t.stats) seen_.stats = *t.stats;
        scale_ = opt.scale.value_or(static_cast<float>(1.0 / std::sqrt(static_cast<double>(DQK))));
        return {error_code_t::OK, ""};
    }

    // The host function's arguments that do not change between calls, from the
    // template the compiler reported and this plan's capacities.
    error_t
    bind_plan(SdpaForwardOptions const &) {
        auto const &meta  = kernel_->meta;
        auto const &order = meta.at("order");
        auto const &tmpl  = meta.at("template");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !order.is_array() || !tmpl.is_array() || order.size() != tmpl.size() || order.size() > 64,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "SdpaForward: the kernel's argument template is malformed");
        auto index = [&](char const *name, size_t &out) -> error_t {
            for (size_t i = 0; i < order.size(); i++) {
                if (order[i] == name) {
                    out = i;
                    return {error_code_t::OK, ""};
                }
            }
            return {error_code_t::GRAPH_NOT_SUPPORTED, std::string("SdpaForward: the kernel has no argument ") + name};
        };
        CHECK_CUDNN_FRONTEND_ERROR(index("q_ptr", slot_.q));
        CHECK_CUDNN_FRONTEND_ERROR(index("k_ptr", slot_.k));
        CHECK_CUDNN_FRONTEND_ERROR(index("v_ptr", slot_.v));
        CHECK_CUDNN_FRONTEND_ERROR(index("o_ptr", slot_.o));
        CHECK_CUDNN_FRONTEND_ERROR(index("lse_ptr", slot_.lse));
        CHECK_CUDNN_FRONTEND_ERROR(index("sinks_ptr", slot_.sinks));
        CHECK_CUDNN_FRONTEND_ERROR(index("meta_ptr", slot_.meta));
        CHECK_CUDNN_FRONTEND_ERROR(index("o_desc_ptr", slot_.o_desc));
        CHECK_CUDNN_FRONTEND_ERROR(index("thd_q_lens_ptr", slot_.q_lens));
        CHECK_CUDNN_FRONTEND_ERROR(index("thd_kv_lens_ptr", slot_.kv_lens));
        CHECK_CUDNN_FRONTEND_ERROR(index("stream", slot_.stream));

        proto_.assign(order.size(), detail::any_none());
        for (size_t i = 0; i < tmpl.size(); i++) {
            auto const &v = tmpl[i];
            if (v.is_number_integer()) {
                proto_[i] = detail::any_int(v.get<int64_t>());
            } else if (v.is_number_float()) {
                proto_[i] = detail::any_float(v.get<double>());
            } else if (v.is_array()) {
                CHECK_CUDNN_FRONTEND_ERROR(put_array(i, v.get<std::vector<int64_t>>()));
            }
        }

        auto const &q = seen_.dense[0], &k = seen_.dense[1], &v = seen_.dense[2], &o = seen_.dense[3];
        int64_t const B = seen_.q_layout.max_seqs, TQ = seen_.q_layout.max_tokens, TKV = seen_.kv_layout.max_tokens;
        int64_t const HQ = q.dim[1], HKV = k.dim[1];
        auto strides = [](TensorView const &x) { return std::vector<int64_t>{x.stride[0], x.stride[0], x.stride[1]}; };
        size_t i     = 0;
        CHECK_CUDNN_FRONTEND_ERROR(index("q_strides", i));
        CHECK_CUDNN_FRONTEND_ERROR(put_array(i, strides(q)));
        CHECK_CUDNN_FRONTEND_ERROR(index("k_strides", i));
        CHECK_CUDNN_FRONTEND_ERROR(put_array(i, strides(k)));
        CHECK_CUDNN_FRONTEND_ERROR(index("v_strides", i));
        CHECK_CUDNN_FRONTEND_ERROR(put_array(i, strides(v)));
        CHECK_CUDNN_FRONTEND_ERROR(index("o_strides", i));
        CHECK_CUDNN_FRONTEND_ERROR(put_array(i, strides(o)));
        CHECK_CUDNN_FRONTEND_ERROR(index("problem_size", i));
        CHECK_CUDNN_FRONTEND_ERROR(put_array(i, {B, HQ, HKV, TQ, TKV, 0}));

        // A safe grid without reading the lengths: the sum over sequences of
        // ceil(len / tile) is at most (Tq - 1) / tile + B; extra units exit on the
        // device. A persistent kernel launches at most its resident clusters.
        int64_t const tile = meta.at("cga_tile_m").get<int64_t>(), cap = meta.value("max_units", int64_t{0});
        int64_t units = ((TQ - 1) / tile + B) * HQ;
        if (cap > 0) units = std::min(units, cap);
        CHECK_CUDNN_FRONTEND_ERROR(index("n_thd_units", i));
        proto_[i] = detail::any_int(units);
        CHECK_CUDNN_FRONTEND_ERROR(index("scale_softmax_log2", i));
        proto_[i] = detail::any_float(scale_ * meta.at("scale_log2_factor").get<double>());

        // Workspace: per-sequence metadata, output TMA descriptors, the sink placeholder; 128-byte regions.
        auto align      = [](int64_t n) { return (n + 127) / 128 * 128; };
        off_o_desc_     = align(4 * (4 * B + 4));
        workspace_size_ = off_o_desc_ + align((B + 3) * 128) + align(HQ * 4);
        return {error_code_t::OK, ""};
    }

    error_t
    put_array(size_t slot, std::vector<int64_t> const &values) {
        detail::IntArray a;
        CHECK_CUDNN_FRONTEND_ERROR(a.assign(values));
        proto_[slot] = a.any();
        arrays_.push_back(std::move(a));
        return {error_code_t::OK, ""};
    }

    static bool
    same(TensorView const &a, TensorView const &b) {
        return a.dtype == b.dtype && a.rank == b.rank && a.dim == b.dim && a.stride == b.stride;
    }

    static bool
    same(SeqLayout const *a, SeqLayout const &b) {
        return a && a->max_seqs == b.max_seqs && a->max_tokens == b.max_tokens && same(a->offsets, b.offsets);
    }

    bool
    same_layouts(SdpaForwardTensors const &t) const {
        return same(t.q.layout, seen_.q_layout) && same(t.k.layout, seen_.kv_layout) && t.o.layout == t.q.layout &&
               t.v.layout == t.k.layout && same(t.q.values, seen_.dense[0]) && same(t.k.values, seen_.dense[1]) &&
               same(t.v.values, seen_.dense[2]) && same(t.o.values, seen_.dense[3]) &&
               (t.stats != nullptr) == seen_.stats.has_value() && (!t.stats || same(*t.stats, *seen_.stats));
    }

    struct Seen {  // what create() validated, copied: execute() must be handed the same shapes
        SeqLayout q_layout, kv_layout;
        std::array<TensorView, 4> dense;  // q, k, v, o
        std::optional<TensorView> stats;
    } seen_;
    struct Slots {
        size_t q = 0, k = 0, v = 0, o = 0, lse = 0, sinks = 0, meta = 0, o_desc = 0, q_lens = 0, kv_lens = 0,
               stream = 0;
    } slot_;
    std::shared_ptr<detail::Kernel const> kernel_;
    std::vector<detail::ffi::Any> proto_;   // the host function's arguments, per-call slots overwritten
    std::vector<detail::IntArray> arrays_;  // owns the tuple arguments in proto_
    int64_t off_o_desc_ = 0, workspace_size_ = 0;
    bool has_stats_ = false;
    float scale_    = 1.0f;
};

}  // namespace ops
}  // namespace cudnn_frontend
