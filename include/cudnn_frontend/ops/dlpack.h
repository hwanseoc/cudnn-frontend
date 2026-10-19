/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// TensorViews from DLPack tensors, for callers that already hold them. Needs
// dlpack/dlpack.h (DLPack 1.x) on the include path; nothing else in
// cudnn_frontend::ops does.

#include <dlpack/dlpack.h>

#include "tensor_view.h"

namespace cudnn_frontend {
namespace ops {

// A view of DLPack's DLTensor: same address (plus byte_offset), dims and
// strides (compact when strides is null). Borrows nothing past the call.
inline error_t
from_dlpack(DLTensor const &t, TensorView &out) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(t.ndim < 0 || t.ndim > 8 || t.dtype.lanes != 1,
                                   error_code_t::INVALID_VALUE,
                                   "from_dlpack: rank must be at most 8, one lane per element");
    DataType_t dtype = DataType_t::NOT_SET;
    if (t.dtype.code == kDLFloat && t.dtype.bits == 16) dtype = DataType_t::HALF;
    if (t.dtype.code == kDLBfloat && t.dtype.bits == 16) dtype = DataType_t::BFLOAT16;
    if (t.dtype.code == kDLFloat && t.dtype.bits == 32) dtype = DataType_t::FLOAT;
    if (t.dtype.code == kDLInt && t.dtype.bits == 32) dtype = DataType_t::INT32;
    if (t.dtype.code == kDLInt && t.dtype.bits == 64) dtype = DataType_t::INT64;
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        dtype == DataType_t::NOT_SET, error_code_t::INVALID_VALUE, "from_dlpack: unsupported data type");
    TensorView v;
    v.data          = static_cast<char *>(t.data) + t.byte_offset;
    v.dtype         = dtype;
    v.rank          = t.ndim;
    int64_t compact = 1;
    for (int i = t.ndim - 1; i >= 0; i--) {
        v.dim[i]    = t.shape[i];
        v.stride[i] = t.strides ? t.strides[i] : compact;
        compact *= t.shape[i];
    }
    out = v;
    return {error_code_t::OK, ""};
}

}  // namespace ops
}  // namespace cudnn_frontend
