/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

// Helpers and CPU references for the cudnn_frontend::ops samples: device
// buffers, random inputs, and plain-loop SDPA and KDA to check against. Not
// part of the API the samples show.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK_CUDA(x)                                                                        \
    do {                                                                                     \
        cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

#define CHECK_FE(x)                                                                            \
    do {                                                                                       \
        auto s_ = (x);                                                                         \
        if (s_.is_bad()) {                                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, s_.get_message().c_str()); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

namespace reference {

// A device copy of a host vector, freed on scope exit.
template <typename T>
struct Device {
    T *ptr   = nullptr;
    size_t n = 0;
    explicit Device(std::vector<T> const &host) : n(host.size()) {
        CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(ptr, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    explicit Device(size_t count) : n(count) { CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(T))); }
    Device(Device const &) = delete;
    ~Device() { cudaFree(ptr); }
    std::vector<T>
    host() const {
        std::vector<T> out(n);
        CHECK_CUDA(cudaMemcpy(out.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
};

inline std::vector<float>
random(size_t n, float lo, float hi, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (auto &x : out) x = dist(gen);
    return out;
}

inline std::vector<__nv_bfloat16>
to_bf16(std::vector<float> const &x) {
    std::vector<__nv_bfloat16> out(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = __float2bfloat16(x[i]);
    return out;
}

inline std::vector<float>
to_float(std::vector<__nv_bfloat16> const &x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = __bfloat162float(x[i]);
    return out;
}

// What the device sees after rounding to bf16.
inline std::vector<float>
rounded(std::vector<float> const &x) {
    return to_float(to_bf16(x));
}

// RMS(out - ref) / RMS(ref).
inline double
rms_ratio(std::vector<float> const &out, std::vector<double> const &ref) {
    double diff = 0, norm = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        diff += (out[i] - ref[i]) * (out[i] - ref[i]);
        norm += ref[i] * ref[i];
    }
    return std::sqrt(diff / std::max(norm, 1e-30));
}

// Attention per sequence over packed [tokens, heads, dim] rows; o is [Tq, HQ, DV].
inline std::vector<double>
sdpa(std::vector<float> const &q,
     std::vector<float> const &k,
     std::vector<float> const &v,
     std::vector<int> const &cu_q,
     std::vector<int> const &cu_kv,
     int hq,
     int hkv,
     int d,
     bool causal) {
    std::vector<double> o(static_cast<size_t>(cu_q.back()) * hq * d, 0.0);
    double const scale = 1.0 / std::sqrt(static_cast<double>(d));
    for (size_t s = 0; s + 1 < cu_q.size(); s++) {
        for (int h = 0; h < hq; h++) {
            int const hk = h / (hq / hkv);
            for (int i = cu_q[s]; i < cu_q[s + 1]; i++) {
                int const last = causal ? std::min(cu_kv[s + 1], cu_kv[s] + (i - cu_q[s]) + 1) : cu_kv[s + 1];
                std::vector<double> p;
                double max = -INFINITY;
                for (int j = cu_kv[s]; j < last; j++) {
                    double dot = 0;
                    for (int x = 0; x < d; x++) dot += q[(i * hq + h) * d + x] * k[(j * hkv + hk) * d + x];
                    p.push_back(dot * scale);
                    max = std::max(max, p.back());
                }
                double sum = 0;
                for (auto &x : p) sum += (x = std::exp(x - max));
                for (int j = cu_kv[s]; j < last; j++) {
                    for (int x = 0; x < d; x++)
                        o[(i * hq + h) * d + x] += p[j - cu_kv[s]] / sum * v[(j * hkv + hk) * d + x];
                }
            }
        }
    }
    return o;
}

// KDA per sequence and head, with q and k l2-normalized: o is [T, H, D], the state [N, H, D, D] (V-major).
inline std::vector<double>
kda(std::vector<float> const &q,
    std::vector<float> const &k,
    std::vector<float> const &v,
    std::vector<float> const &g,
    std::vector<float> const &beta,
    std::vector<float> const &state0,
    std::vector<int> const &cu,
    int heads,
    int d) {
    std::vector<double> o(q.size(), 0.0);
    double const scale = 1.0 / std::sqrt(static_cast<double>(d));
    auto normalized    = [&](std::vector<float> const &x, size_t at) {
        double n = 0;
        for (int i = 0; i < d; i++) n += double(x[at + i]) * x[at + i];
        std::vector<double> out(d);
        for (int i = 0; i < d; i++) out[i] = x[at + i] / std::sqrt(n + 1e-6);
        return out;
    };
    for (size_t s = 0; s + 1 < cu.size(); s++) {
        for (int h = 0; h < heads; h++) {
            std::vector<double> S(state0.begin() + ((s * heads + h) * d * d),
                                  state0.begin() + ((s * heads + h + 1) * d * d));
            for (int t = cu[s]; t < cu[s + 1]; t++) {
                size_t const row = (static_cast<size_t>(t) * heads + h) * d;
                auto const qt = normalized(q, row), kt = normalized(k, row);
                for (int vi = 0; vi < d; vi++) {  // decay, then the delta rule on the decayed state
                    double ks = 0;
                    for (int ki = 0; ki < d; ki++) ks += (S[vi * d + ki] *= std::exp(g[row + ki])) * kt[ki];
                    double const update = beta[t * heads + h] * (v[row + vi] - ks);
                    double out          = 0;
                    for (int ki = 0; ki < d; ki++) out += (S[vi * d + ki] += update * kt[ki]) * qt[ki];
                    o[row + vi] = scale * out;
                }
            }
        }
    }
    return o;
}

}  // namespace reference
