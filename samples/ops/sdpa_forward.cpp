/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

// Causal attention over three packed sequences of different lengths, from C++.
//
//     sdpa_forward --config            print the kernel configuration this sample needs
//     sdpa_forward kernels.cudnn       run it with kernels compiled by `python -m cudnn.ops`

#include <cudnn_frontend/ops/sdpa_forward.h>

#include <cstring>

#include "reference.h"

namespace ops = cudnn_frontend::ops;
using cudnn_frontend::DataType_t;

int
main(int argc, char **argv) {
    int const heads_q = 8, heads_kv = 2, d = 128;
    std::vector<int> cu_q  = {0, 100, 137, 437};  // query lengths 100, 37, 300
    std::vector<int> cu_kv = {0, 100, 137, 437};  // keys: the same sequences
    int const seqs = 3, tokens = cu_q.back();

    auto q_host = reference::rounded(reference::random(size_t(tokens) * heads_q * d, -1, 1, 1));
    auto k_host = reference::rounded(reference::random(size_t(tokens) * heads_kv * d, -1, 1, 2));
    auto v_host = reference::rounded(reference::random(size_t(tokens) * heads_kv * d, -1, 1, 3));
    reference::Device<__nv_bfloat16> q(reference::to_bf16(q_host)), k(reference::to_bf16(k_host)),
        v(reference::to_bf16(v_host)), o(size_t(tokens) * heads_q * d);
    reference::Device<int> cu_q_dev(cu_q), cu_kv_dev(cu_kv);

    // Packed operands and the sequence layouts they follow.
    ops::SeqLayout q_layout{ops::make_tensor_view(cu_q_dev.ptr, DataType_t::INT32, {seqs + 1}), seqs, tokens};
    ops::SeqLayout kv_layout{ops::make_tensor_view(cu_kv_dev.ptr, DataType_t::INT32, {seqs + 1}), seqs, tokens};
    ops::SdpaForwardTensors tensors;
    tensors.q = {ops::make_tensor_view(q.ptr, DataType_t::BFLOAT16, {tokens, heads_q, d}), &q_layout};
    tensors.k = {ops::make_tensor_view(k.ptr, DataType_t::BFLOAT16, {tokens, heads_kv, d}), &kv_layout};
    tensors.v = {ops::make_tensor_view(v.ptr, DataType_t::BFLOAT16, {tokens, heads_kv, d}), &kv_layout};
    tensors.o = {ops::make_tensor_view(o.ptr, DataType_t::BFLOAT16, {tokens, heads_q, d}), &q_layout};
    ops::SdpaForwardOptions options;
    options.causal = true;

    if (argc > 1 && std::strcmp(argv[1], "--config") == 0) {
        std::string config;
        CHECK_FE(ops::SdpaForward::kernel_config(tensors, options, config));
        std::printf("%s\n", config.c_str());
        return 0;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s --config | %s <kernel library>\n", argv[0], argv[0]);
        return 2;
    }

    ops::KernelLibrary library;
    CHECK_FE(library.load_file(argv[1]));
    ops::SdpaForward plan;
    CHECK_FE(ops::SdpaForward::create(tensors, options, library, plan));
    reference::Device<char> workspace(plan.workspace_size());
    CHECK_FE(plan.execute(tensors, workspace.ptr, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto const expected = reference::sdpa(q_host, k_host, v_host, cu_q, cu_kv, heads_q, heads_kv, d, true);
    double const error  = reference::rms_ratio(reference::to_float(o.host()), expected);
    std::printf("sdpa_forward: rms error %.2e vs fp64 reference\n", error);
    return error < 2e-2 ? 0 : 1;
}
