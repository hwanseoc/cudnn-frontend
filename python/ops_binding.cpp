// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Python bindings of cudnn_frontend::ops (include/cudnn_frontend/ops/). The
// Python ops (python/cudnn/ops/*_forward.py) run the same C++ plans C++ callers
// do, with kernels from the in-process registry (cudnn.ops.register).
// Tensors arrive as (address, cudnn.data_type, shape, strides) tuples.

#include <memory>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "cudnn_frontend.h"
#include "cudnn_frontend/ops/kda_forward.h"
#include "cudnn_frontend/ops/sdpa_forward.h"

namespace py = nanobind;

namespace cudnn_frontend {
namespace python_bindings {

void
throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const &error_msg);

namespace {

using namespace cudnn_frontend::ops;

void
check(error_t const &e) {
    throw_if(e.is_bad(), e.code, e.err_msg);
}

TensorView
view(py::handle desc) {
    auto t = py::cast<py::tuple>(desc);
    TensorView v;
    v.data     = reinterpret_cast<void *>(py::cast<intptr_t>(t[0]));
    v.dtype    = py::cast<DataType_t>(t[1]);
    auto shape = py::cast<py::sequence>(t[2]), stride = py::cast<py::sequence>(t[3]);
    v.rank = static_cast<int>(py::len(shape));
    if (v.rank > 8 || static_cast<int>(py::len(stride)) != v.rank) {
        throw std::invalid_argument("cudnn.ops: a tensor has rank at most 8 and one stride per dimension");
    }
    for (int i = 0; i < v.rank; i++) {
        v.dim[i]    = py::cast<int64_t>(shape[i]);
        v.stride[i] = py::cast<int64_t>(stride[i]);
    }
    return v;
}

std::optional<TensorView>
optional_view(py::dict const &d, char const *name) {
    if (!d.contains(name) || d[name].is_none()) return std::nullopt;
    return view(d[name]);
}

template <typename T>
T
option(py::dict const &d, char const *name, T fallback) {
    return d.contains(name) && !d[name].is_none() ? py::cast<T>(d[name]) : fallback;
}

std::optional<float>
optional_float(py::dict const &d, char const *name) {
    if (!d.contains(name) || d[name].is_none()) return std::nullopt;
    return py::cast<float>(d[name]);
}

// One call's operands. Holds the views the op's tensors point into, so it is never copied or moved.
struct KdaCall {
    SeqLayout layout;
    std::optional<TensorView> s0, s1, a_log, dt_bias;
    KdaForwardTensors t;

    explicit KdaCall(py::dict const &d) {
        layout.offsets     = view(d["cu_seqlens"]);
        layout.max_seqs    = layout.offsets.dim[0] - 1;
        TensorView const q = view(d["q"]);
        layout.max_tokens  = q.dim[0];
        t.q                = {q, &layout};
        t.k                = {view(d["k"]), &layout};
        t.v                = {view(d["v"]), &layout};
        t.g                = {view(d["g"]), &layout};
        t.beta             = {view(d["beta"]), &layout};
        t.o                = {view(d["o"]), &layout};
        s0                 = optional_view(d, "initial_state");
        s1                 = optional_view(d, "final_state");
        a_log              = optional_view(d, "a_log");
        dt_bias            = optional_view(d, "dt_bias");
        t.initial_state    = s0 ? &*s0 : nullptr;
        t.final_state      = s1 ? &*s1 : nullptr;
        t.a_log            = a_log ? &*a_log : nullptr;
        t.dt_bias          = dt_bias ? &*dt_bias : nullptr;
    }
    KdaCall(KdaCall const &) = delete;
};

KdaForwardOptions
kda_options(py::dict const &d) {
    KdaForwardOptions o;
    o.scale            = optional_float(d, "scale");
    o.use_qk_l2norm    = option(d, "use_qk_l2norm", false);
    o.use_beta_sigmoid = option(d, "use_beta_sigmoid", false);
    o.allow_neg_eigval = option(d, "allow_neg_eigval", false);
    o.safe_gate        = option(d, "safe_gate", false);
    o.gate_domain = option(d, "gate_domain", std::string("log")) == "log" ? KdaGateDomain::LOG : KdaGateDomain::LINEAR;
    o.gate_lower_bound = optional_float(d, "gate_lower_bound");
    return o;
}

struct SdpaCall {
    SeqLayout q_layout, kv_layout;
    std::optional<TensorView> stats;
    SdpaForwardTensors t;

    explicit SdpaCall(py::dict const &d) {
        q_layout.offsets   = view(d["cu_seqlens_q"]);
        kv_layout.offsets  = view(d["cu_seqlens_kv"]);
        TensorView const q = view(d["q"]), k = view(d["k"]);
        q_layout.max_seqs    = q_layout.offsets.dim[0] - 1;
        kv_layout.max_seqs   = kv_layout.offsets.dim[0] - 1;
        q_layout.max_tokens  = q.dim[0];
        kv_layout.max_tokens = k.dim[0];
        t.q                  = {q, &q_layout};
        t.k                  = {k, &kv_layout};
        t.v                  = {view(d["v"]), &kv_layout};
        t.o                  = {view(d["o"]), &q_layout};
        stats                = optional_view(d, "stats");
        t.stats              = stats ? &*stats : nullptr;
    }
    SdpaCall(SdpaCall const &) = delete;
};

SdpaForwardOptions
sdpa_options(py::dict const &d) {
    SdpaForwardOptions o;
    o.scale  = optional_float(d, "scale");
    o.causal = option(d, "causal", false);
    return o;
}

// A plan and the views it was created with. execute_pointers() rebinds only the
// addresses, so a warm Python call converts no shapes; it runs under the GIL,
// which serializes the rebinding.
template <typename Plan, typename Call>
struct Prepared {
    std::unique_ptr<Call> call;
    Plan plan;
};
using KdaPrepared  = Prepared<KdaForward, KdaCall>;
using SdpaPrepared = Prepared<SdpaForward, SdpaCall>;

void *
ptr(intptr_t p) {
    return reinterpret_cast<void *>(p);
}

}  // namespace

void
init_ops_submodule(py::module_ &m) {
    py::class_<KdaPrepared>(m, "_KdaForward")
        .def("__init__",
             [](KdaPrepared *self, py::dict tensors, py::dict options) {
                 auto call = std::make_unique<KdaCall>(tensors);
                 KdaForward plan;
                 check(KdaForward::create(call->t, kda_options(options), KernelLibrary::registry(), plan));
                 new (self) KdaPrepared{std::move(call), std::move(plan)};
             })
        .def_prop_ro("workspace_size", [](KdaPrepared const &self) { return self.plan.workspace_size(); })
        .def("execute",
             [](KdaPrepared const &self, py::dict tensors, intptr_t workspace, intptr_t stream) {
                 KdaCall call(tensors);
                 check(self.plan.execute(call.t, ptr(workspace), reinterpret_cast<cudaStream_t>(stream)));
             })
        .def("execute_pointers",
             [](KdaPrepared &self,
                intptr_t q,
                intptr_t k,
                intptr_t v,
                intptr_t g,
                intptr_t beta,
                intptr_t o,
                intptr_t cu_seqlens,
                intptr_t initial_state,
                intptr_t final_state,
                intptr_t a_log,
                intptr_t dt_bias,
                intptr_t workspace,
                intptr_t stream) {
                 auto &c               = *self.call;
                 c.t.q.values.data     = ptr(q);
                 c.t.k.values.data     = ptr(k);
                 c.t.v.values.data     = ptr(v);
                 c.t.g.values.data     = ptr(g);
                 c.t.beta.values.data  = ptr(beta);
                 c.t.o.values.data     = ptr(o);
                 c.layout.offsets.data = ptr(cu_seqlens);
                 if (c.s0) c.s0->data = ptr(initial_state);
                 if (c.s1) c.s1->data = ptr(final_state);
                 if (c.a_log) c.a_log->data = ptr(a_log);
                 if (c.dt_bias) c.dt_bias->data = ptr(dt_bias);
                 check(self.plan.execute(c.t, ptr(workspace), reinterpret_cast<cudaStream_t>(stream)));
             });
    m.def("_kda_forward_config", [](py::dict tensors, py::dict options) {
        KdaCall call(tensors);
        std::string config;
        check(KdaForward::kernel_config(call.t, kda_options(options), config));
        return config;
    });

    py::class_<SdpaPrepared>(m, "_SdpaForward")
        .def("__init__",
             [](SdpaPrepared *self, py::dict tensors, py::dict options) {
                 auto call = std::make_unique<SdpaCall>(tensors);
                 SdpaForward plan;
                 check(SdpaForward::create(call->t, sdpa_options(options), KernelLibrary::registry(), plan));
                 new (self) SdpaPrepared{std::move(call), std::move(plan)};
             })
        .def_prop_ro("workspace_size", [](SdpaPrepared const &self) { return self.plan.workspace_size(); })
        .def("execute",
             [](SdpaPrepared const &self, py::dict tensors, intptr_t workspace, intptr_t stream) {
                 SdpaCall call(tensors);
                 check(self.plan.execute(call.t, ptr(workspace), reinterpret_cast<cudaStream_t>(stream)));
             })
        .def("execute_pointers",
             [](SdpaPrepared &self,
                intptr_t q,
                intptr_t k,
                intptr_t v,
                intptr_t o,
                intptr_t cu_seqlens_q,
                intptr_t cu_seqlens_kv,
                intptr_t stats,
                intptr_t workspace,
                intptr_t stream) {
                 auto &c                  = *self.call;
                 c.t.q.values.data        = ptr(q);
                 c.t.k.values.data        = ptr(k);
                 c.t.v.values.data        = ptr(v);
                 c.t.o.values.data        = ptr(o);
                 c.q_layout.offsets.data  = ptr(cu_seqlens_q);
                 c.kv_layout.offsets.data = ptr(cu_seqlens_kv);
                 if (c.stats) c.stats->data = ptr(stats);
                 check(self.plan.execute(c.t, ptr(workspace), reinterpret_cast<cudaStream_t>(stream)));
             });
    m.def("_sdpa_forward_config", [](py::dict tensors, py::dict options) {
        SdpaCall call(tensors);
        std::string config;
        check(SdpaForward::kernel_config(call.t, sdpa_options(options), config));
        return config;
    });
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
