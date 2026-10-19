/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Kimi Delta Attention forward over packed variable-length sequences.
//
//     S_t = (I - beta_t k_t k_t^T) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T,   o_t = scale * q_t S_t
//
// per sequence and head, with S the (V, K) state. q, k, v, g, beta and o are
// packed [tokens, heads, dim] and share one SeqLayout. SM100 / SM103 / SM107,
// f16 or bf16, head dims 64 or 128.

#include "kernel_library.h"

#include <cmath>
#include <cstdio>
#include <optional>

namespace cudnn_frontend {
namespace ops {

enum class KdaGateDomain { LOG, LINEAR };

// Numerics only: equal options compute the same function.
struct KdaForwardOptions {
    std::optional<float> scale;  // default 1 / sqrt(d_k)
    bool use_qk_l2norm        = false;
    bool use_beta_sigmoid     = false;
    bool allow_neg_eigval     = false;  // needs use_beta_sigmoid
    bool safe_gate            = false;  // g = lower_bound * sigmoid(exp(a_log) * (g + dt_bias)); needs a_log, dt_bias
    KdaGateDomain gate_domain = KdaGateDomain::LOG;
    std::optional<float> gate_lower_bound;  // safe_gate only; default -5
};

struct KdaForwardTensors {
    RaggedView q;                               // [T, HQ, K]
    RaggedView k;                               // [T, HK, K], HK = HQ or HV
    RaggedView v;                               // [T, HV, V]
    RaggedView g;                               // [T, HO, K], HO = max(HQ, HV); FLOAT, HALF or BFLOAT16
    RaggedView beta;                            // [T, HO]; FLOAT or the io type
    RaggedView o;                               // [T, HO, V]
    TensorView const *initial_state = nullptr;  // [max_seqs, HO, V, K]; FLOAT or BFLOAT16
    TensorView const *final_state   = nullptr;  // same; may be initial_state's buffer
    TensorView const *a_log         = nullptr;  // [HO], safe_gate only
    TensorView const *dt_bias       = nullptr;  // [HO, K], safe_gate only
};

class KdaBackward;

class KdaForward {
    friend class KdaBackward;  // shares the input checks

   public:
    // Validates, lays out the workspace and takes the kernel from lib. Never
    // compiles: a kernel missing from lib is an error naming its configuration.
    static error_t
    create(KdaForwardTensors const &t, KdaForwardOptions const &opt, KernelLibrary const &lib, KdaForward &out) {
        KdaForward p;
        json config;
        CHECK_CUDNN_FRONTEND_ERROR(p.validate(t, opt, config));
        CHECK_CUDNN_FRONTEND_ERROR(lib.find(config, p.kernel_));
        int cc = 0, num_sm = 0;
        CHECK_CUDNN_FRONTEND_ERROR(detail::current_device(p.device_, cc, num_sm));

        // The uncut scheme's scratch: scheduler | work items | work count | TMA descriptors, 128-byte regions.
        auto align          = [](int64_t n) { return (n + 127) / 128 * 128; };
        auto const &tm      = p.kernel_->meta.at("tensormap_bytes");
        int64_t const items = p.seqs_ * p.heads_out_;
        p.tm_bytes_         = tm.at("per_seq").get<int64_t>() * p.seqs_ + tm.at("base").get<int64_t>();
        p.off_items_        = 128;
        p.off_count_        = p.off_items_ + align(items * 10 * 4);
        p.off_tm_           = p.off_count_ + 128;
        p.workspace_size_   = p.off_tm_ + align(p.tm_bytes_);
        p.scheduler_shape_  = {2, 1};
        p.items_shape_      = {items, 10};
        p.items_stride_     = {10, 1};
        p.count_shape_      = {1, 1};
        p.tm_shape_         = {p.tm_bytes_ / 8, 1};
        out                 = std::move(p);
        return {error_code_t::OK, ""};
    }

    // The configuration create() looks up, to compile it ahead of time.
    static error_t
    kernel_config(KdaForwardTensors const &t, KdaForwardOptions const &opt, std::string &config) {
        KdaForward p;
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
    execute(KdaForwardTensors const &t, void *workspace, cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !same_layouts(t), error_code_t::INVALID_VARIANT_PACK, "KdaForward::execute: tensors differ from create()");
        RETURN_CUDNN_FRONTEND_ERROR_IF(reinterpret_cast<uintptr_t>(workspace) % 128 != 0,
                                       error_code_t::INVALID_VALUE,
                                       "KdaForward::execute: the workspace must be 128-byte aligned");
        using detail::dl_tensor;
        auto *ws                = static_cast<char *>(workspace);
        detail::ffi::DLTensor q = dl_tensor(t.q.values, device_), k = dl_tensor(t.k.values, device_),
                              v = dl_tensor(t.v.values, device_), g = dl_tensor(t.g.values, device_),
                              beta = dl_tensor(t.beta.values, device_), o = dl_tensor(t.o.values, device_),
                              cu = dl_tensor(t.q.layout->offsets, device_);
        detail::ffi::DLTensor s0{}, s1{}, a_log{}, dt_bias{};
        if (t.initial_state) s0 = dl_tensor(*t.initial_state, device_);
        if (t.final_state) s1 = dl_tensor(*t.final_state, device_);
        if (safe_gate_) {
            a_log   = dl_tensor(*t.a_log, device_);
            dt_bias = dl_tensor(*t.dt_bias, device_);
        }
        auto region = [&](int64_t off, DataType_t dtype, int rank, int64_t const *shape, int64_t const *stride) {
            detail::ffi::DLTensor r{};
            r.data   = ws + off;
            r.device = {2, device_};
            r.ndim   = rank;
            detail::dl_dtype(dtype, r.dtype);
            r.shape   = const_cast<int64_t *>(shape);
            r.strides = const_cast<int64_t *>(stride);
            return r;
        };
        int64_t const one[1] = {1};
        auto const scheduler = region(0, DataType_t::INT32, 1, scheduler_shape_.data(), one);
        auto const items     = region(off_items_, DataType_t::INT32, 2, items_shape_.data(), items_stride_.data());
        auto const count     = region(off_count_, DataType_t::INT32, 1, count_shape_.data(), one);
        auto const tm        = region(off_tm_, DataType_t::INT64, 1, tm_shape_.data(), one);

        using detail::any_tensor;
        detail::ffi::Any const args[] = {
            detail::any_int(0),  // checkpoint_every_n
            detail::any_float(scale_),
            any_tensor(&q),
            any_tensor(&k),
            any_tensor(&v),
            any_tensor(&g),
            any_tensor(safe_gate_ ? &a_log : nullptr),
            any_tensor(safe_gate_ ? &dt_bias : nullptr),
            any_tensor(&beta),
            any_tensor(&o),
            any_tensor(&cu),
            any_tensor(t.initial_state ? &s0 : nullptr),
            any_tensor(t.final_state ? &s1 : nullptr),
            detail::any_none(),  // seed_indices
            detail::any_none(),  // final_indices
            detail::any_none(),  // checkpoints
            any_tensor(&items),
            any_tensor(&count),
            any_tensor(&scheduler),
            any_tensor(&tm),
            detail::any_none(),  // prep_k_decay .. prep_row_count: the uncut scheme has no prep pass
            detail::any_none(),
            detail::any_none(),
            detail::any_none(),
            detail::any_none(),
            detail::any_none(),
            detail::any_none(),
            detail::any_none(),
            detail::any_int(static_cast<int64_t>(reinterpret_cast<intptr_t>(stream))),
        };
        return kernel_->call(args, static_cast<int>(sizeof(args) / sizeof(args[0])));
    }

   private:
    static char const *
    dtype_name(DataType_t t) {
        switch (t) {
            case DataType_t::HALF:
                return "fp16";
            case DataType_t::BFLOAT16:
                return "bf16";
            case DataType_t::FLOAT:
                return "fp32";
            case DataType_t::INT32:
                return "int32";
            case DataType_t::INT64:
                return "int64";
            default:
                return nullptr;
        }
    }

    static bool
    is(TensorView const &v, std::initializer_list<int64_t> dim) {
        if (v.rank != static_cast<int>(dim.size())) return false;
        int i = 0;
        for (int64_t d : dim) {
            if (v.dim[i++] != d) return false;
        }
        return v.stride[v.rank - 1] == 1;
    }

    error_t
    validate(KdaForwardTensors const &t, KdaForwardOptions const &opt, json &config) {
        SeqLayout const *layout = t.q.layout;
        RETURN_CUDNN_FRONTEND_ERROR_IF(layout == nullptr || t.k.layout != layout || t.v.layout != layout ||
                                           t.g.layout != layout || t.beta.layout != layout || t.o.layout != layout,
                                       error_code_t::INVALID_VALUE,
                                       "KdaForward: q, k, v, g, beta and o must share one SeqLayout");
        auto const &q = t.q.values, &k = t.k.values, &v = t.v.values, &g = t.g.values, &beta = t.beta.values,
                   &o   = t.o.values;
        int64_t const T = layout->max_tokens, B = layout->max_seqs;
        RETURN_CUDNN_FRONTEND_ERROR_IF(q.rank != 3 || k.rank != 3 || v.rank != 3,
                                       error_code_t::INVALID_VALUE,
                                       "KdaForward: q, k and v are [tokens, heads, dim]");
        int64_t const HQ = q.dim[1], HK = k.dim[1], HV = v.dim[1], K = q.dim[2], V = v.dim[2];
        int64_t const HO = std::max(HQ, HV);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            B < 1 || T < 1 || !is(layout->offsets, {B + 1}) ||
                (layout->offsets.dtype != DataType_t::INT32 && layout->offsets.dtype != DataType_t::INT64),
            error_code_t::INVALID_VALUE,
            "KdaForward: the SeqLayout needs max_seqs >= 1, max_tokens >= 1 and int32/int64 offsets [max_seqs + 1]");
        RETURN_CUDNN_FRONTEND_ERROR_IF(!is(q, {T, HQ, K}) || !is(k, {T, HK, K}) || !is(v, {T, HV, V}) ||
                                           !is(g, {T, HO, K}) || !is(beta, {T, HO}) || !is(o, {T, HO, V}),
                                       error_code_t::INVALID_VALUE,
                                       "KdaForward: shapes must be q [T,HQ,K], k [T,HK,K], v [T,HV,V], g [T,HO,K], "
                                       "beta [T,HO], o [T,HO,V] with T = max_tokens, HO = max(HQ, HV), and the "
                                       "innermost dimension contiguous");
        RETURN_CUDNN_FRONTEND_ERROR_IF(HO % HQ != 0 || HO % HV != 0 || (HK != HQ && HK != HV),
                                       error_code_t::INVALID_VALUE,
                                       "KdaForward: HQ and HV must divide max(HQ, HV), and HK must equal HQ or HV");
        RETURN_CUDNN_FRONTEND_ERROR_IF((K != 64 && K != 128) || (V != 64 && V != 128),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "KdaForward: head dims must be 64 or 128");
        DataType_t const io = q.dtype;
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (io != DataType_t::HALF && io != DataType_t::BFLOAT16) || k.dtype != io || v.dtype != io || o.dtype != io,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "KdaForward: q, k, v and o must all be HALF or all BFLOAT16");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            g.dtype != DataType_t::FLOAT && g.dtype != DataType_t::HALF && g.dtype != DataType_t::BFLOAT16,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "KdaForward: g must be FLOAT, HALF or BFLOAT16");
        RETURN_CUDNN_FRONTEND_ERROR_IF(beta.dtype != DataType_t::FLOAT && beta.dtype != io,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "KdaForward: beta must be FLOAT or the io type");
        for (auto *s : {t.initial_state, t.final_state}) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                s != nullptr &&
                    (!is(*s, {B, HO, V, K}) || (s->dtype != DataType_t::FLOAT && s->dtype != DataType_t::BFLOAT16)),
                error_code_t::INVALID_VALUE,
                "KdaForward: states are FLOAT or BFLOAT16 [max_seqs, HO, V, K]");
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            t.initial_state && t.final_state && t.initial_state->dtype != t.final_state->dtype,
            error_code_t::INVALID_VALUE,
            "KdaForward: initial and final state must have the same dtype");
        RETURN_CUDNN_FRONTEND_ERROR_IF(opt.allow_neg_eigval && !opt.use_beta_sigmoid,
                                       error_code_t::INVALID_VALUE,
                                       "KdaForward: allow_neg_eigval needs use_beta_sigmoid");
        RETURN_CUDNN_FRONTEND_ERROR_IF(opt.safe_gate && opt.gate_domain != KdaGateDomain::LOG,
                                       error_code_t::INVALID_VALUE,
                                       "KdaForward: safe_gate works in the log gate domain");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            opt.safe_gate && (!t.a_log || !t.dt_bias || !is(*t.a_log, {HO}) || !is(*t.dt_bias, {HO, K}) ||
                              !dtype_name(t.a_log->dtype) || !dtype_name(t.dt_bias->dtype)),
            error_code_t::INVALID_VALUE,
            "KdaForward: safe_gate needs a_log [HO] and dt_bias [HO, K]");
        for (TensorView const *x : {&q, &k, &v, &g, &o}) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(reinterpret_cast<uintptr_t>(x->data) % 16 != 0,
                                           error_code_t::INVALID_VALUE,
                                           "KdaForward: q, k, v, g and o must be 16-byte aligned");
        }

        auto const *state = t.initial_state ? t.initial_state : t.final_state;
        char lower_bound[32];
        std::snprintf(
            lower_bound, sizeof(lower_bound), "%.17g", static_cast<double>(opt.gate_lower_bound.value_or(-5.0f)));
        config = {
            {"op", "kda_forward"},
            {"io", dtype_name(io)},
            {"g", dtype_name(g.dtype)},
            {"beta", dtype_name(beta.dtype)},
            {"cu_seqlens", dtype_name(layout->offsets.dtype)},
            {"state", state ? dtype_name(state->dtype) : "none"},
            {"a_log", opt.safe_gate ? dtype_name(t.a_log->dtype) : "none"},
            {"dt_bias", opt.safe_gate ? dtype_name(t.dt_bias->dtype) : "none"},
            {"d_k", K},
            {"d_v", V},
            {"log_gate", opt.gate_domain == KdaGateDomain::LOG},
            {"safe_gate", opt.safe_gate},
            {"gate_lower_bound", std::string(lower_bound)},
            {"use_qk_l2norm", opt.use_qk_l2norm},
            {"use_beta_sigmoid", opt.use_beta_sigmoid},
            {"allow_neg_eigval", opt.allow_neg_eigval},
            {"initial_state", t.initial_state != nullptr},
            {"final_state", t.final_state != nullptr},
        };
        seqs_                        = B;
        heads_out_                   = HO;
        safe_gate_                   = opt.safe_gate;
        scale_                       = opt.scale.value_or(static_cast<float>(1.0 / std::sqrt(static_cast<double>(K))));
        seen_                        = {*layout, {q, k, v, g, beta, o}, {}};
        TensorView const *optional[] = {t.initial_state, t.final_state, t.a_log, t.dt_bias};
        for (int i = 0; i < 4; i++) {
            if (optional[i]) seen_.optional[i] = *optional[i];
        }
        return {error_code_t::OK, ""};
    }

    static bool
    same(TensorView const &a, TensorView const &b) {
        return a.dtype == b.dtype && a.rank == b.rank && a.dim == b.dim && a.stride == b.stride;
    }

    static bool
    same(TensorView const *a, std::optional<TensorView> const &b) {
        return (a == nullptr) == !b.has_value() && (a == nullptr || same(*a, *b));
    }

    bool
    same_layouts(KdaForwardTensors const &t) const {
        auto const &l = seen_.layout;
        if (!t.q.layout || t.q.layout->max_seqs != l.max_seqs || t.q.layout->max_tokens != l.max_tokens ||
            !same(t.q.layout->offsets, l.offsets)) {
            return false;
        }
        TensorView const *dense[] = {&t.q.values, &t.k.values, &t.v.values, &t.g.values, &t.beta.values, &t.o.values};
        TensorView const *optional[] = {t.initial_state, t.final_state, t.a_log, t.dt_bias};
        for (int i = 0; i < 6; i++) {
            if (!same(*dense[i], seen_.dense[i])) return false;
        }
        for (int i = 0; i < 4; i++) {
            if (!same(optional[i], seen_.optional[i])) return false;
        }
        return true;
    }

    std::shared_ptr<detail::Kernel const> kernel_;
    struct Seen {  // what create() validated, copied: execute() must be handed the same shapes
        SeqLayout layout;
        std::array<TensorView, 6> dense;                    // q, k, v, g, beta, o
        std::array<std::optional<TensorView>, 4> optional;  // initial_state, final_state, a_log, dt_bias
    } seen_;
    int device_   = 0;
    int64_t seqs_ = 0, heads_out_ = 0, tm_bytes_ = 0, off_items_ = 0, off_count_ = 0, off_tm_ = 0, workspace_size_ = 0;
    std::array<int64_t, 2> scheduler_shape_{}, items_shape_{}, items_stride_{}, count_shape_{}, tm_shape_{};
    bool safe_gate_ = false;
    float scale_    = 1.0f;
};

}  // namespace ops
}  // namespace cudnn_frontend
