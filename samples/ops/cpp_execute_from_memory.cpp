/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

// Python compiles, C++ executes from memory: Kimi Delta Attention forward over three packed sequences.
// The library arrives as bytes (here on stdin, as it would from a compile service) and never touches disk.
//
//     cpp_execute_from_memory --config > configs.txt                  the kernel configuration this sample needs
//     python -m cudnn.ops configs.txt -o - | cpp_execute_from_memory  Python compiles, C++ loads the bytes and runs

#include <cudnn_frontend/ops/kda_forward.h>

#include <cstring>
#include <iostream>
#include <iterator>

#include "reference.h"

namespace ops = cudnn_frontend::ops;
using cudnn_frontend::DataType_t;

int
main(int argc, char **argv) {
    int const heads = 4, d = 128;
    std::vector<int> cu = {0, 100, 137, 437};  // sequence lengths 100, 37, 300
    int const seqs = 3, tokens = cu.back();
    size_t const n = size_t(tokens) * heads * d;

    auto q_host     = reference::rounded(reference::random(n, -1, 1, 1));
    auto k_host     = reference::rounded(reference::random(n, -1, 1, 2));
    auto v_host     = reference::rounded(reference::random(n, -1, 1, 3));
    auto g_host     = reference::random(n, -0.1f, 0, 4);                   // log decay per key channel
    auto beta_host  = reference::random(size_t(tokens) * heads, 0, 1, 5);  // write strength
    auto state_host = reference::random(size_t(seqs) * heads * d * d, -0.1f, 0.1f, 6);
    reference::Device<__nv_bfloat16> q(reference::to_bf16(q_host)), k(reference::to_bf16(k_host)),
        v(reference::to_bf16(v_host)), o(n);
    reference::Device<float> g(g_host), beta(beta_host), state(state_host), final_state(state_host.size());
    reference::Device<int> cu_dev(cu);

    // Packed operands share one sequence layout; the states are per sequence.
    ops::SeqLayout layout{ops::make_tensor_view(cu_dev.ptr, DataType_t::INT32, {seqs + 1}), seqs, tokens};
    auto initial = ops::make_tensor_view(state.ptr, DataType_t::FLOAT, {seqs, heads, d, d});
    auto final   = ops::make_tensor_view(final_state.ptr, DataType_t::FLOAT, {seqs, heads, d, d});
    ops::KdaForwardTensors tensors;
    tensors.q             = {ops::make_tensor_view(q.ptr, DataType_t::BFLOAT16, {tokens, heads, d}), &layout};
    tensors.k             = {ops::make_tensor_view(k.ptr, DataType_t::BFLOAT16, {tokens, heads, d}), &layout};
    tensors.v             = {ops::make_tensor_view(v.ptr, DataType_t::BFLOAT16, {tokens, heads, d}), &layout};
    tensors.g             = {ops::make_tensor_view(g.ptr, DataType_t::FLOAT, {tokens, heads, d}), &layout};
    tensors.beta          = {ops::make_tensor_view(beta.ptr, DataType_t::FLOAT, {tokens, heads}), &layout};
    tensors.o             = {ops::make_tensor_view(o.ptr, DataType_t::BFLOAT16, {tokens, heads, d}), &layout};
    tensors.initial_state = &initial;
    tensors.final_state   = &final;
    ops::KdaForwardOptions options;
    options.use_qk_l2norm = true;

    if (argc > 1 && std::strcmp(argv[1], "--config") == 0) {
        std::string config;
        CHECK_FE(ops::KdaForward::kernel_config(tensors, options, config));
        std::printf("%s\n", config.c_str());
        return 0;
    }
    std::string const blob((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    ops::KernelLibrary library;
    CHECK_FE(library.load_bytes(blob.data(), blob.size()));
    ops::KdaForward plan;
    CHECK_FE(ops::KdaForward::create(tensors, options, library, plan));
    reference::Device<char> workspace(plan.workspace_size());
    CHECK_FE(plan.execute(tensors, workspace.ptr, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto const expected = reference::kda(q_host, k_host, v_host, g_host, beta_host, state_host, cu, heads, d);
    double const error  = reference::rms_ratio(reference::to_float(o.host()), expected);
    std::printf("kda_forward: rms error %.2e vs fp64 reference\n", error);
    return error < 2e-2 ? 0 : 1;
}
