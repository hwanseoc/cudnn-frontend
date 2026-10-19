/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Gradients of Kimi Delta Attention (kda_forward.h) over packed
// variable-length sequences: dq, dk, dv, dg, dbeta, and optionally the
// initial state's, from d_o and optionally the final state's. The forward
// states are recomputed from the inputs. SM100 / SM103 / SM107, f16 or bf16,
// head dims 64 or 128, as many heads for q, k and v.

#include "kda_forward.h"

namespace cudnn_frontend {
namespace ops {

// The forward's numerics options; safe_gate is not served by the backward yet.
using KdaBackwardOptions = KdaForwardOptions;

struct KdaBackwardTensors {
    RaggedView q, k, v, g, beta;                  // the forward's inputs
    RaggedView d_o;                               // [T, H, V]
    RaggedView dq, dk, dv, dg, dbeta;             // outputs, shaped like q, k, v, g, beta
    TensorView const *initial_state   = nullptr;  // the forward's, [max_seqs, H, V, K]
    TensorView const *d_final_state   = nullptr;  // gradient of the final state, same shape
    TensorView const *d_initial_state = nullptr;  // output, same shape
};

class KdaBackward {
   public:
    // Validates, lays out the workspace and takes the kernel from lib. Never
    // compiles: a kernel missing from lib is an error naming its configuration.
    static error_t
    create(KdaBackwardTensors const &t, KdaBackwardOptions const &opt, KernelLibrary const &lib, KdaBackward &out) {
        KdaBackward p;
        json config;
        CHECK_CUDNN_FRONTEND_ERROR(p.validate(t, opt, config));
        CHECK_CUDNN_FRONTEND_ERROR(lib.find(config, p.kernel_));
        int cc = 0, num_sm = 0;
        CHECK_CUDNN_FRONTEND_ERROR(detail::current_device(p.device_, cc, num_sm));

        // scheduler | work items | work count | recomputed state series | recompute and bprop TMA descriptors
        auto align = [](int64_t n) { return (n + 127) / 128 * 128; };
        auto tm    = [&](char const *which) {
            auto const &m = p.kernel_->meta.at(which);
            return m.at("per_seq").get<int64_t>() * p.seqs_ + m.at("base").get<int64_t>();
        };
        int64_t const items        = p.seqs_ * p.heads_;
        p.series_rows_             = std::max<int64_t>(p.tokens_ / 16 + p.seqs_, 1);
        int64_t const series_bytes = p.series_rows_ * p.heads_ * p.dk_ * p.dv_ * 2;
        p.off_items_               = 128;
        p.off_count_               = p.off_items_ + align(items * 40);
        p.off_series_              = p.off_count_ + 128;
        p.off_rec_tm_              = p.off_series_ + align(series_bytes);
        p.rec_tm_bytes_            = tm("recompute_tensormap_bytes");
        p.off_bwd_tm_              = p.off_rec_tm_ + align(p.rec_tm_bytes_);
        p.bwd_tm_bytes_            = tm("bprop_tensormap_bytes");
        p.workspace_size_          = p.off_bwd_tm_ + align(p.bwd_tm_bytes_);
        p.items_shape_             = {items, 10};
        p.series_shape_            = {p.series_rows_, p.heads_, p.dv_, p.dk_};
        p.series_stride_           = {p.heads_ * p.dv_ * p.dk_, p.dv_ * p.dk_, p.dk_, 1};
        out                        = std::move(p);
        return {error_code_t::OK, ""};
    }

    static error_t
    kernel_config(KdaBackwardTensors const &t, KdaBackwardOptions const &opt, std::string &config) {
        KdaBackward p;
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

    error_t
    execute(KdaBackwardTensors const &t, void *workspace, cudaStream_t stream) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !same_layouts(t), error_code_t::INVALID_VARIANT_PACK, "KdaBackward::execute: tensors differ from create()");
        RETURN_CUDNN_FRONTEND_ERROR_IF(reinterpret_cast<uintptr_t>(workspace) % 128 != 0,
                                       error_code_t::INVALID_VALUE,
                                       "KdaBackward::execute: the workspace must be 128-byte aligned");
        using detail::any_int;
        using detail::any_none;
        using detail::any_tensor;
        using detail::dl_tensor;
        auto *ws                = static_cast<char *>(workspace);
        int const dev           = device_;
        detail::ffi::DLTensor q = dl_tensor(t.q.values, dev), k = dl_tensor(t.k.values, dev),
                              v = dl_tensor(t.v.values, dev), g = dl_tensor(t.g.values, dev),
                              beta = dl_tensor(t.beta.values, dev), d_o = dl_tensor(t.d_o.values, dev),
                              dq = dl_tensor(t.dq.values, dev), dk = dl_tensor(t.dk.values, dev),
                              dv = dl_tensor(t.dv.values, dev), dg = dl_tensor(t.dg.values, dev),
                              dbeta = dl_tensor(t.dbeta.values, dev), cu = dl_tensor(t.q.layout->offsets, dev);
        detail::ffi::DLTensor s0{}, ds1{}, ds0{};
        if (t.initial_state) s0 = dl_tensor(*t.initial_state, dev);
        if (t.d_final_state) ds1 = dl_tensor(*t.d_final_state, dev);
        if (t.d_initial_state) ds0 = dl_tensor(*t.d_initial_state, dev);
        auto region = [&](int64_t off, DataType_t dtype, int rank, int64_t const *shape, int64_t const *stride) {
            detail::ffi::DLTensor r{};
            r.data   = ws + off;
            r.device = {2, dev};
            r.ndim   = rank;
            detail::dl_dtype(dtype, r.dtype);
            r.shape   = const_cast<int64_t *>(shape);
            r.strides = const_cast<int64_t *>(stride);
            return r;
        };
        static int64_t const one[1] = {1}, two[1] = {2}, four[1] = {4}, items_stride[2] = {10, 1};
        int64_t const rec_words[1] = {rec_tm_bytes_ / 8}, bwd_words[1] = {bwd_tm_bytes_ / 8};
        auto const sched_all = region(0, DataType_t::INT32, 1, four, one);
        auto const sched_rec = region(0, DataType_t::INT32, 1, two, one);
        auto const sched_bwd = region(8, DataType_t::INT32, 1, two, one);
        auto const items     = region(off_items_, DataType_t::INT32, 2, items_shape_.data(), items_stride);
        auto const count     = region(off_count_, DataType_t::INT32, 1, one, one);
        auto const series    = region(off_series_, io_, 4, series_shape_.data(), series_stride_.data());
        auto const rec_tm    = region(off_rec_tm_, DataType_t::INT64, 1, rec_words, one);
        auto const bwd_tm    = region(off_bwd_tm_, DataType_t::INT64, 1, bwd_words, one);

        detail::ffi::Any const args[] = {
            any_int(16),  // checkpoint_every_n: the series keeps one state per 16-token chunk
            any_int(0),   // seed_span_chunks
            any_int(0),   // seed_every_n
            detail::any_float(scale_),
            any_tensor(&q),
            any_tensor(&k),
            any_tensor(&v),
            any_tensor(&d_o),
            any_tensor(&dq),
            any_tensor(&dk),
            any_tensor(&dv),
            any_tensor(&g),
            any_tensor(log_gate_ ? nullptr : &g),  // gate_main: the linear-domain gate
            any_tensor(&beta),
            any_none(),  // a_log
            any_none(),  // dt_bias
            any_tensor(&cu),
            any_tensor(&series),
            any_none(),  // seed_checkpoints
            any_tensor(t.initial_state ? &s0 : nullptr),
            any_tensor(&dg),
            any_tensor(&dbeta),
            any_tensor(t.d_initial_state ? &ds0 : nullptr),
            any_tensor(t.d_final_state ? &ds1 : nullptr),
            any_tensor(&items),
            any_tensor(&count),
            any_tensor(&items),  // series items: the recompute walks the same table
            any_tensor(&count),
            any_tensor(&sched_all),
            any_tensor(&sched_all),
            any_none(),
            any_tensor(&sched_rec),
            any_tensor(&sched_bwd),
            any_tensor(&rec_tm),
            any_tensor(&bwd_tm),
            any_int(static_cast<int64_t>(reinterpret_cast<intptr_t>(stream))),
        };
        return kernel_->call(args, static_cast<int>(sizeof(args) / sizeof(args[0])));
    }

   private:
    error_t
    validate(KdaBackwardTensors const &t, KdaBackwardOptions const &opt, json &config) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            opt.safe_gate, error_code_t::GRAPH_NOT_SUPPORTED, "KdaBackward: safe_gate is not served yet");
        // The forward's checks cover the inputs; the gradients mirror them.
        KdaForwardTensors f;
        f.q = t.q, f.k = t.k, f.v = t.v, f.g = t.g, f.beta = t.beta, f.o = t.d_o;
        f.initial_state = t.initial_state;
        json forward;
        KdaForward check;
        CHECK_CUDNN_FRONTEND_ERROR(check.validate(f, opt, forward));
        SeqLayout const *layout = t.q.layout;
        RETURN_CUDNN_FRONTEND_ERROR_IF(t.dq.layout != layout || t.dk.layout != layout || t.dv.layout != layout ||
                                           t.dg.layout != layout || t.dbeta.layout != layout,
                                       error_code_t::INVALID_VALUE,
                                       "KdaBackward: the gradients share the inputs' SeqLayout");
        auto like = [](TensorView const &a, TensorView const &b) {
            return a.dtype == b.dtype && a.rank == b.rank && a.dim == b.dim && a.stride[a.rank - 1] == 1 &&
                   reinterpret_cast<uintptr_t>(a.data) % 4 == 0;
        };
        RETURN_CUDNN_FRONTEND_ERROR_IF(!like(t.dq.values, t.q.values) || !like(t.dk.values, t.k.values) ||
                                           !like(t.dv.values, t.v.values) || !like(t.dg.values, t.g.values) ||
                                           !like(t.dbeta.values, t.beta.values),
                                       error_code_t::INVALID_VALUE,
                                       "KdaBackward: each gradient has its input's shape and dtype");
        int64_t const HQ = t.q.values.dim[1];
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            t.k.values.dim[1] != HQ || t.v.values.dim[1] != HQ,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "KdaBackward: q, k and v must have the same head count (grouped heads are not served yet)");
        DataType_t const state = t.initial_state ? t.initial_state->dtype : DataType_t::FLOAT;
        for (auto *s : {t.d_final_state, t.d_initial_state}) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                s && (s->dtype != state || s->rank != 4 || s->dim[0] != layout->max_seqs || s->dim[1] != HQ ||
                      s->dim[2] != t.v.values.dim[2] || s->dim[3] != t.q.values.dim[2] || s->stride[3] != 1),
                error_code_t::INVALID_VALUE,
                "KdaBackward: state gradients are [max_seqs, H, V, K] in the state's dtype "
                "(FLOAT without an initial state)");
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(t.d_initial_state && !t.initial_state,
                                       error_code_t::INVALID_VALUE,
                                       "KdaBackward: d_initial_state needs the initial_state");
        auto name = [](TensorView const *s) -> std::string {
            if (!s) return "none";
            return s->dtype == DataType_t::FLOAT ? "fp32" : "bf16";
        };
        config = {
            {"op", "kda_backward"},
            {"io", forward.at("io")},
            {"g", forward.at("g")},
            {"beta", forward.at("beta")},
            {"cu_seqlens", forward.at("cu_seqlens")},
            {"state", name(t.initial_state)},
            {"d_final_state", name(t.d_final_state)},
            {"d_initial_state", name(t.d_initial_state)},
            {"d_k", forward.at("d_k")},
            {"d_v", forward.at("d_v")},
            {"log_gate", forward.at("log_gate")},
            {"gate_lower_bound", forward.at("gate_lower_bound")},
            {"use_qk_l2norm", opt.use_qk_l2norm},
            {"use_beta_sigmoid", opt.use_beta_sigmoid},
            {"allow_neg_eigval", opt.allow_neg_eigval},
        };
        io_                       = t.q.values.dtype;
        log_gate_                 = opt.gate_domain == KdaGateDomain::LOG;
        seqs_                     = layout->max_seqs;
        tokens_                   = layout->max_tokens;
        heads_                    = HQ;
        dk_                       = t.q.values.dim[2];
        dv_                       = t.v.values.dim[2];
        scale_                    = opt.scale.value_or(static_cast<float>(1.0 / std::sqrt(static_cast<double>(dk_))));
        seen_.layout              = *layout;
        TensorView const *dense[] = {&t.q.values,
                                     &t.k.values,
                                     &t.v.values,
                                     &t.g.values,
                                     &t.beta.values,
                                     &t.d_o.values,
                                     &t.dq.values,
                                     &t.dk.values,
                                     &t.dv.values,
                                     &t.dg.values,
                                     &t.dbeta.values};
        for (int i = 0; i < 11; i++) seen_.dense[i] = *dense[i];
        TensorView const *optional[] = {t.initial_state, t.d_final_state, t.d_initial_state};
        for (int i = 0; i < 3; i++) {
            if (optional[i]) seen_.optional[i] = *optional[i];
        }
        return {error_code_t::OK, ""};
    }

    static bool
    same(TensorView const &a, TensorView const &b) {
        return a.dtype == b.dtype && a.rank == b.rank && a.dim == b.dim && a.stride == b.stride;
    }

    bool
    same_layouts(KdaBackwardTensors const &t) const {
        auto const &l = seen_.layout;
        if (!t.q.layout || t.q.layout->max_seqs != l.max_seqs || t.q.layout->max_tokens != l.max_tokens ||
            !same(t.q.layout->offsets, l.offsets)) {
            return false;
        }
        TensorView const *dense[] = {&t.q.values,
                                     &t.k.values,
                                     &t.v.values,
                                     &t.g.values,
                                     &t.beta.values,
                                     &t.d_o.values,
                                     &t.dq.values,
                                     &t.dk.values,
                                     &t.dv.values,
                                     &t.dg.values,
                                     &t.dbeta.values};
        for (int i = 0; i < 11; i++) {
            if (!same(*dense[i], seen_.dense[i])) return false;
        }
        TensorView const *optional[] = {t.initial_state, t.d_final_state, t.d_initial_state};
        for (int i = 0; i < 3; i++) {
            if ((optional[i] == nullptr) != !seen_.optional[i].has_value()) return false;
            if (optional[i] && !same(*optional[i], *seen_.optional[i])) return false;
        }
        return true;
    }

    struct Seen {
        SeqLayout layout;
        std::array<TensorView, 11> dense;
        std::array<std::optional<TensorView>, 3> optional;
    } seen_;
    std::shared_ptr<detail::Kernel const> kernel_;
    DataType_t io_ = DataType_t::NOT_SET;
    bool log_gate_ = true;
    int device_    = 0;
    int64_t seqs_ = 0, tokens_ = 0, heads_ = 0, dk_ = 0, dv_ = 0, series_rows_ = 0;
    int64_t off_items_ = 0, off_count_ = 0, off_series_ = 0, off_rec_tm_ = 0, off_bwd_tm_ = 0;
    int64_t rec_tm_bytes_ = 0, bwd_tm_bytes_ = 0, workspace_size_ = 0;
    std::array<int64_t, 2> items_shape_{};
    std::array<int64_t, 4> series_shape_{}, series_stride_{};
    float scale_ = 1.0f;
};

}  // namespace ops
}  // namespace cudnn_frontend
