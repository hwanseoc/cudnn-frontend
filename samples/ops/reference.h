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

// Gradients of sdpa() with as many KV heads as query heads: {dq, dk, dv}, each [T, H, D].
inline std::vector<std::vector<double>>
sdpa_backward(std::vector<float> const &q,
              std::vector<float> const &k,
              std::vector<float> const &v,
              std::vector<float> const &d_o,
              std::vector<int> const &cu,
              int heads,
              int d,
              bool causal) {
    std::vector<std::vector<double>> grads(3, std::vector<double>(q.size(), 0.0));
    auto &dq = grads[0], &dk = grads[1], &dv = grads[2];
    double const scale = 1.0 / std::sqrt(static_cast<double>(d));
    auto at            = [&](int token, int h) { return (static_cast<size_t>(token) * heads + h) * d; };
    for (size_t s = 0; s + 1 < cu.size(); s++) {
        int const first = cu[s], n = cu[s + 1] - cu[s];
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < n; i++) {
                int const keys = causal ? i + 1 : n;
                std::vector<double> p(keys), dp(keys);
                double max = -INFINITY, sum = 0, delta = 0;
                for (int j = 0; j < keys; j++) {
                    double dot = 0;
                    for (int x = 0; x < d; x++) dot += q[at(first + i, h) + x] * k[at(first + j, h) + x];
                    max = std::max(max, p[j] = dot * scale);
                }
                for (auto &x : p) sum += (x = std::exp(x - max));
                for (int j = 0; j < keys; j++) {
                    p[j] /= sum;
                    for (int x = 0; x < d; x++) dp[j] += d_o[at(first + i, h) + x] * v[at(first + j, h) + x];
                    delta += p[j] * dp[j];  // rowsum(dO * O)
                }
                for (int j = 0; j < keys; j++) {
                    double const ds = p[j] * (dp[j] - delta) * scale;
                    for (int x = 0; x < d; x++) {
                        dv[at(first + j, h) + x] += p[j] * d_o[at(first + i, h) + x];
                        dq[at(first + i, h) + x] += ds * k[at(first + j, h) + x];
                        dk[at(first + j, h) + x] += ds * q[at(first + i, h) + x];
                    }
                }
            }
        }
    }
    return grads;
}

// Gradients of the KDA recurrence (no l2 normalization, log-domain gate, state V-major [V, K]) by backpropagation
// through time: {dq, dk, dv, dg, dbeta, d_initial_state}.
inline std::vector<std::vector<double>>
kda_backward(std::vector<float> const &q,
             std::vector<float> const &k,
             std::vector<float> const &v,
             std::vector<float> const &g,
             std::vector<float> const &beta,
             std::vector<float> const &state0,
             std::vector<float> const &d_o,
             std::vector<int> const &cu,
             int heads,
             int d) {
    std::vector<std::vector<double>> grads = {std::vector<double>(q.size()),
                                              std::vector<double>(q.size()),
                                              std::vector<double>(q.size()),
                                              std::vector<double>(q.size()),
                                              std::vector<double>(beta.size()),
                                              std::vector<double>(state0.size())};
    auto &dq = grads[0], &dk = grads[1], &dv = grads[2], &dg = grads[3], &dbeta = grads[4], &ds0 = grads[5];
    double const scale = 1.0 / std::sqrt(static_cast<double>(d));
    size_t const dd    = static_cast<size_t>(d) * d;
    for (size_t s = 0; s + 1 < cu.size(); s++) {
        for (int h = 0; h < heads; h++) {
            int const n = cu[s + 1] - cu[s];
            std::vector<std::vector<double>> states(n + 1);  // states[t] is the state before token t
            states[0].assign(state0.begin() + (s * heads + h) * dd, state0.begin() + (s * heads + h + 1) * dd);
            auto row  = [&](int t) { return (static_cast<size_t>(cu[s] + t) * heads + h) * d; };
            auto step = [&](int t, std::vector<double> &a, std::vector<double> &r) {  // decayed state, residual
                size_t const x = row(t);
                a              = states[t];
                r.assign(d, 0.0);
                for (int vi = 0; vi < d; vi++) {
                    double u = 0;
                    for (int ki = 0; ki < d; ki++) u += (a[vi * d + ki] *= std::exp(g[x + ki])) * k[x + ki];
                    r[vi] = v[x + vi] - u;
                }
            };
            std::vector<double> a, r;
            for (int t = 0; t < n; t++) {
                step(t, a, r);
                double const b = beta[(cu[s] + t) * heads + h];
                for (int vi = 0; vi < d; vi++)
                    for (int ki = 0; ki < d; ki++) a[vi * d + ki] += b * r[vi] * k[row(t) + ki];
                states[t + 1] = a;
            }
            std::vector<double> ds(dd, 0.0), dr(d);
            for (int t = n - 1; t >= 0; t--) {
                size_t const x   = row(t);
                size_t const bt  = static_cast<size_t>(cu[s] + t) * heads + h;
                auto const &next = states[t + 1];
                for (int vi = 0; vi < d; vi++) {  // o = scale * S q
                    for (int ki = 0; ki < d; ki++) {
                        ds[vi * d + ki] += scale * d_o[x + vi] * q[x + ki];
                        dq[x + ki] += scale * next[vi * d + ki] * d_o[x + vi];
                    }
                }
                step(t, a, r);
                for (int vi = 0; vi < d; vi++) {  // S = A + beta r k^T
                    double dsk = 0;
                    for (int ki = 0; ki < d; ki++) dsk += ds[vi * d + ki] * k[x + ki];
                    dbeta[bt] += dsk * r[vi];
                    dr[vi]     = beta[bt] * dsk;
                    dv[x + vi] = dr[vi];  // r = v - A k
                }
                for (int ki = 0; ki < d; ki++) {
                    double alpha = std::exp(g[x + ki]), dalpha = 0;
                    for (int vi = 0; vi < d; vi++) {
                        double const da = ds[vi * d + ki] - dr[vi] * k[x + ki];
                        dk[x + ki] += beta[bt] * ds[vi * d + ki] * r[vi] - a[vi * d + ki] * dr[vi];
                        dalpha += da * states[t][vi * d + ki];
                        ds[vi * d + ki] = da * alpha;  // A = S diag(alpha)
                    }
                    dg[x + ki] = dalpha * alpha;
                }
            }
            std::copy(ds.begin(), ds.end(), ds0.begin() + (s * heads + h) * dd);
        }
    }
    return grads;
}

}  // namespace reference
