/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

// Attention forward with its log-sum-exp, then its gradients, over three packed sequences, from C++.
//
//     sdpa_backward --config           print the kernel configurations this sample needs
//     sdpa_backward kernels.cudnn      run it with kernels compiled by `python -m cudnn.ops`

#include <cudnn_frontend/ops/sdpa_backward.h>
#include <cudnn_frontend/ops/sdpa_forward.h>

#include <cstring>

#include "reference.h"

namespace ops = cudnn_frontend::ops;
using cudnn_frontend::DataType_t;

int
main(int argc, char **argv) {
    int const heads = 4, d = 128;
    std::vector<int> cu = {0, 100, 137, 437};  // sequence lengths 100, 37, 300
    int const seqs = 3, tokens = cu.back();
    size_t const n = size_t(tokens) * heads * d;

    auto q_host  = reference::rounded(reference::random(n, -1, 1, 1));
    auto k_host  = reference::rounded(reference::random(n, -1, 1, 2));
    auto v_host  = reference::rounded(reference::random(n, -1, 1, 3));
    auto do_host = reference::rounded(reference::random(n, -1, 1, 4));
    reference::Device<__nv_bfloat16> q(reference::to_bf16(q_host)), k(reference::to_bf16(k_host)),
        v(reference::to_bf16(v_host)), d_o(reference::to_bf16(do_host)), o(n), dq(n), dk(n), dv(n);
    reference::Device<float> stats(size_t(tokens) * heads);
    reference::Device<int> cu_dev(cu);

    // One layout for all: queries and keys are the same sequences. 300 bounds any one sequence.
    ops::SeqLayout layout{ops::make_tensor_view(cu_dev.ptr, DataType_t::INT32, {seqs + 1}), seqs, tokens, 300};
    auto packed = [&](void *data) {
        return ops::RaggedView{ops::make_tensor_view(data, DataType_t::BFLOAT16, {tokens, heads, d}), &layout};
    };
    auto stats_view = ops::make_tensor_view(stats.ptr, DataType_t::FLOAT, {tokens, heads});

    ops::SdpaForwardTensors forward;
    forward.q     = packed(q.ptr);
    forward.k     = packed(k.ptr);
    forward.v     = packed(v.ptr);
    forward.o     = packed(o.ptr);
    forward.stats = &stats_view;
    ops::SdpaForwardOptions forward_options;
    forward_options.causal = true;

    ops::SdpaBackwardTensors backward;
    backward.q     = packed(q.ptr);
    backward.k     = packed(k.ptr);
    backward.v     = packed(v.ptr);
    backward.o     = packed(o.ptr);
    backward.d_o   = packed(d_o.ptr);
    backward.stats = stats_view;
    backward.dq    = packed(dq.ptr);
    backward.dk    = packed(dk.ptr);
    backward.dv    = packed(dv.ptr);
    ops::SdpaBackwardOptions backward_options;
    backward_options.causal = true;

    if (argc > 1 && std::strcmp(argv[1], "--config") == 0) {
        std::string config;
        CHECK_FE(ops::SdpaForward::kernel_config(forward, forward_options, config));
        std::printf("%s\n", config.c_str());
        CHECK_FE(ops::SdpaBackward::kernel_config(backward, backward_options, config));
        std::printf("%s\n", config.c_str());
        return 0;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s --config | %s <kernel library>\n", argv[0], argv[0]);
        return 2;
    }

    ops::KernelLibrary library;
    CHECK_FE(library.load_file(argv[1]));
    ops::SdpaForward forward_plan;
    ops::SdpaBackward backward_plan;
    CHECK_FE(ops::SdpaForward::create(forward, forward_options, library, forward_plan));
    CHECK_FE(ops::SdpaBackward::create(backward, backward_options, library, backward_plan));
    reference::Device<char> workspace(std::max(forward_plan.workspace_size(), backward_plan.workspace_size()));
    CHECK_FE(forward_plan.execute(forward, workspace.ptr, nullptr));
    CHECK_FE(backward_plan.execute(backward, workspace.ptr, nullptr));  // same stream: runs after the forward
    CHECK_CUDA(cudaDeviceSynchronize());

    auto const expected = reference::sdpa_backward(q_host, k_host, v_host, do_host, cu, heads, d, true);
    double error        = 0;
    for (int i = 0; i < 3; i++) {
        auto const &grad = i == 0 ? dq : i == 1 ? dk : dv;
        error            = std::max(error, reference::rms_ratio(reference::to_float(grad.host()), expected[i]));
    }
    std::printf("sdpa_backward: worst rms error %.2e vs fp64 reference (dq, dk, dv)\n", error);
    return error < 2e-2 ? 0 : 1;
}
