/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

// Kimi Delta Attention gradients over three packed sequences, from C++. The backward recomputes the
// forward's states itself; it needs the forward's inputs and dO only.
//
//     kda_backward --config           print the kernel configuration this sample needs
//     kda_backward kernels.cudnn      run it with kernels compiled by `python -m cudnn.ops`

#include <cudnn_frontend/ops/kda_backward.h>

#include <algorithm>
#include <cstring>
#include <iterator>

#include "reference.h"

namespace ops = cudnn_frontend::ops;
using cudnn_frontend::DataType_t;

int
main(int argc, char **argv) {
    int const heads = 4, d = 64;
    std::vector<int> cu = {0, 50, 63, 138};  // sequence lengths 50, 13, 75
    int const seqs = 3, tokens = cu.back();
    size_t const n = size_t(tokens) * heads * d;

    auto unit = [&](unsigned seed) {  // unit-norm rows keep the recurrence bounded
        auto x = reference::random(n, -1, 1, seed);
        for (size_t row = 0; row < n; row += d) {
            float norm = 0;
            for (int i = 0; i < d; i++) norm += x[row + i] * x[row + i];
            for (int i = 0; i < d; i++) x[row + i] /= std::sqrt(norm);
        }
        return reference::rounded(x);
    };
    auto q_host     = unit(1);
    auto k_host     = unit(2);
    auto v_host     = reference::rounded(reference::random(n, -1, 1, 3));
    auto g_host     = reference::random(n, -0.1f, 0, 4);
    auto beta_host  = reference::random(size_t(tokens) * heads, 0, 1, 5);
    auto state_host = reference::random(size_t(seqs) * heads * d * d, -0.1f, 0.1f, 6);
    auto do_host    = reference::rounded(reference::random(n, -1, 1, 7));
    reference::Device<__nv_bfloat16> q(reference::to_bf16(q_host)), k(reference::to_bf16(k_host)),
        v(reference::to_bf16(v_host)), d_o(reference::to_bf16(do_host)), dq(n), dk(n), dv(n);
    reference::Device<float> g(g_host), beta(beta_host), state(state_host), dg(n), dbeta(beta_host.size()),
        d_state(state_host.size());
    reference::Device<int> cu_dev(cu);

    ops::SeqLayout layout{ops::make_tensor_view(cu_dev.ptr, DataType_t::INT32, {seqs + 1}), seqs, tokens};
    auto packed = [&](void *data, DataType_t type) {
        return ops::RaggedView{ops::make_tensor_view(data, type, {tokens, heads, d}), &layout};
    };
    auto per_token = [&](void *data) {
        return ops::RaggedView{ops::make_tensor_view(data, DataType_t::FLOAT, {tokens, heads}), &layout};
    };
    auto initial   = ops::make_tensor_view(state.ptr, DataType_t::FLOAT, {seqs, heads, d, d});
    auto d_initial = ops::make_tensor_view(d_state.ptr, DataType_t::FLOAT, {seqs, heads, d, d});
    ops::KdaBackwardTensors tensors;
    tensors.q               = packed(q.ptr, DataType_t::BFLOAT16);
    tensors.k               = packed(k.ptr, DataType_t::BFLOAT16);
    tensors.v               = packed(v.ptr, DataType_t::BFLOAT16);
    tensors.g               = packed(g.ptr, DataType_t::FLOAT);
    tensors.beta            = per_token(beta.ptr);
    tensors.d_o             = packed(d_o.ptr, DataType_t::BFLOAT16);
    tensors.dq              = packed(dq.ptr, DataType_t::BFLOAT16);
    tensors.dk              = packed(dk.ptr, DataType_t::BFLOAT16);
    tensors.dv              = packed(dv.ptr, DataType_t::BFLOAT16);
    tensors.dg              = packed(dg.ptr, DataType_t::FLOAT);
    tensors.dbeta           = per_token(dbeta.ptr);
    tensors.initial_state   = &initial;
    tensors.d_initial_state = &d_initial;
    ops::KdaBackwardOptions options;  // the forward's options

    if (argc > 1 && std::strcmp(argv[1], "--config") == 0) {
        std::string config;
        CHECK_FE(ops::KdaBackward::kernel_config(tensors, options, config));
        std::printf("%s\n", config.c_str());
        return 0;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s --config | %s <kernel library>\n", argv[0], argv[0]);
        return 2;
    }

    ops::KernelLibrary library;
    CHECK_FE(library.load_file(argv[1]));
    ops::KdaBackward plan;
    CHECK_FE(ops::KdaBackward::create(tensors, options, library, plan));
    reference::Device<char> workspace(plan.workspace_size());
    CHECK_FE(plan.execute(tensors, workspace.ptr, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto const expected =
        reference::kda_backward(q_host, k_host, v_host, g_host, beta_host, state_host, do_host, cu, heads, d);
    double const errors[] = {
        reference::rms_ratio(reference::to_float(dq.host()), expected[0]),
        reference::rms_ratio(reference::to_float(dk.host()), expected[1]),
        reference::rms_ratio(reference::to_float(dv.host()), expected[2]),
        reference::rms_ratio(dg.host(), expected[3]),
        reference::rms_ratio(dbeta.host(), expected[4]),
        reference::rms_ratio(d_state.host(), expected[5]),
    };
    double const error = *std::max_element(std::begin(errors), std::end(errors));
    std::printf("kda_backward: worst rms error %.2e vs fp64 reference (dq, dk, dv, dg, dbeta, d_initial_state)\n",
                error);
    return error < 4e-2 ? 0 : 1;
}
