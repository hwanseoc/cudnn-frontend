/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// The operand types every cudnn_frontend::ops function takes.
//
// A TensorView borrows one dense device buffer: the op never keeps it past the
// call and never allocates. Variable-length (packed) operands share a
// SeqLayout: one device array of sequence offsets plus the host capacities the
// op plans for. The host never reads the offsets, so a call does not
// synchronize to learn the sequence lengths, and a CUDA graph captured at a
// given capacity replays any batch that fits it.

#include "../../cudnn_frontend.h"

#include <array>
#include <cstdint>

namespace cudnn_frontend {
namespace ops {

struct TensorView {
    void *data       = nullptr;
    DataType_t dtype = DataType_t::NOT_SET;
    int rank         = 0;
    std::array<int64_t, 8> dim{};     // rank <= CUDNN_DIM_MAX
    std::array<int64_t, 8> stride{};  // in elements
};

// How packed tensors split into sequences. offsets[i] is the first token of
// sequence i, offsets[max_seqs] the end of the last; unused trailing sequences
// repeat the last offset (empty sequences).
struct SeqLayout {
    TensorView offsets;  // device, [max_seqs + 1], INT32
    int64_t max_seqs   = 0;
    int64_t max_tokens = 0;
};

// Packed values [max_tokens, ...] and the layout they follow.
struct RaggedView {
    TensorView values;
    SeqLayout const *layout = nullptr;
};

// A compact (row-major) view.
inline TensorView
make_tensor_view(void *data, DataType_t dtype, std::initializer_list<int64_t> dim) {
    TensorView t;
    t.data  = data;
    t.dtype = dtype;
    t.rank  = static_cast<int>(dim.size());
    std::copy(dim.begin(), dim.end(), t.dim.begin());
    int64_t s = 1;
    for (int i = t.rank - 1; i >= 0; i--) {
        t.stride[i] = s;
        s *= t.dim[i];
    }
    return t;
}

}  // namespace ops
}  // namespace cudnn_frontend
