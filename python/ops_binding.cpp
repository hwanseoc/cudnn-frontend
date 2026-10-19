// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Python bindings of cudnn_frontend::ops (include/cudnn_frontend/ops/). The
// Python ops (python/cudnn/ops/*_forward.py) run the same C++ plans C++ callers
// do, with kernels from the in-process registry (cudnn.ops.register).
// Tensors arrive as (address, cudnn.data_type, shape, strides) tuples.

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

}  // namespace

void
init_ops_submodule(py::module_ &m) {
    py::class_<KdaForward>(m, "_KdaForward")
        .def("__init__",
             [](KdaForward *self, py::dict tensors, py::dict options) {
                 KdaCall call(tensors);
                 KdaForward plan;
                 check(KdaForward::create(call.t, kda_options(options), KernelLibrary::registry(), plan));
                 new (self) KdaForward(std::move(plan));
             })
        .def_prop_ro("workspace_size", &KdaForward::workspace_size)
        .def("execute", [](KdaForward const &self, py::dict tensors, intptr_t workspace, intptr_t stream) {
            KdaCall call(tensors);
            check(self.execute(call.t, reinterpret_cast<void *>(workspace), reinterpret_cast<cudaStream_t>(stream)));
        });
    m.def("_kda_forward_config", [](py::dict tensors, py::dict options) {
        KdaCall call(tensors);
        std::string config;
        check(KdaForward::kernel_config(call.t, kda_options(options), config));
        return config;
    });

    py::class_<SdpaForward>(m, "_SdpaForward")
        .def("__init__",
             [](SdpaForward *self, py::dict tensors, py::dict options) {
                 SdpaCall call(tensors);
                 SdpaForward plan;
                 check(SdpaForward::create(call.t, sdpa_options(options), KernelLibrary::registry(), plan));
                 new (self) SdpaForward(std::move(plan));
             })
        .def_prop_ro("workspace_size", &SdpaForward::workspace_size)
        .def("execute", [](SdpaForward const &self, py::dict tensors, intptr_t workspace, intptr_t stream) {
            SdpaCall call(tensors);
            check(self.execute(call.t, reinterpret_cast<void *>(workspace), reinterpret_cast<cudaStream_t>(stream)));
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
