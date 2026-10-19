# cudnn.ops from C++: SDPA and KDA, forward and backward

**This FE-OSS API is experimental and subject to change.**

`cudnn_frontend::ops` calls CuTeDSL ops directly from C++, with the same kernels the Python ops
`cudnn.ops.sdpa_forward`, `sdpa_backward`, `kda_forward` and `kda_backward` run. The kernels are compiled at run time by
Python, in the calling process, in another process, or on another machine with the same GPU. The C++
ops never run Python and never compile: they take kernels from a `KernelLibrary` the caller supplies.

| Op | C++ header | Python |
|---|---|---|
| Attention forward over packed variable-length sequences (SM100/SM103, f16/bf16, causal or none, optional LSE) | `cudnn_frontend/ops/sdpa_forward.h` | `cudnn.ops.sdpa_forward` |
| Its gradients (SM100, f16/bf16, head dim 128, as many KV heads as query heads, compact rows) | `cudnn_frontend/ops/sdpa_backward.h` | `cudnn.ops.sdpa_backward` |
| Kimi Delta Attention forward over packed variable-length sequences (SM100/SM103, f16/bf16, head dims 64/128) | `cudnn_frontend/ops/kda_forward.h` | `cudnn.ops.kda_forward` |
| Its gradients, recomputing the states (same heads for q, k, v; no safe gate) | `cudnn_frontend/ops/kda_backward.h` | `cudnn.ops.kda_backward` |

## Python

```python
o, stats = cudnn.ops.sdpa_forward(q, k, v, cu_seqlens_q, cu_seqlens_kv, causal=True, return_stats=True)
dq, dk, dv = cudnn.ops.sdpa_backward(q, k, v, o, do, stats, cu_seqlens_q, cu_seqlens_kv, causal=True)
o, final_state = cudnn.ops.kda_forward(q, k, v, g, beta, cu_seqlens, initial_state=s0, output_final_state=True, use_qk_l2norm=True)
dq, dk, dv, dg, dbeta, d_s0 = cudnn.ops.kda_backward(q, k, v, g, beta, cu_seqlens, do, initial_state=s0, use_qk_l2norm=True)
```

Tensors are packed `[tokens, heads, dim]`; `cu_seqlens*` are the int32 sequence offsets `[N + 1]`. The
first call for a kernel configuration compiles it (seconds) and registers it for the process; later
calls with any shape reuse it.

## C++

```cpp
#include <cudnn_frontend/ops/sdpa_forward.h>
namespace ops = cudnn_frontend::ops;

ops::SeqLayout layout{ops::make_tensor_view(cu_ptr, DataType_t::INT32, {n + 1}), n, tokens};
ops::SdpaForwardTensors t;
t.q = {ops::make_tensor_view(q_ptr, DataType_t::BFLOAT16, {tokens, 8, 128}), &layout};
// ... k, v, o
ops::KernelLibrary lib;
lib.load_file("kernels.cudnn");                     // or load_bytes(data, size), or KernelLibrary::registry()
ops::SdpaForward plan;
ops::SdpaForward::create(t, options, lib, plan);    // never compiles; a missing kernel is an error naming it
// workspace of plan.workspace_size() bytes, 128-byte aligned
plan.execute(t, workspace, stream);                 // no allocation, no synchronization, capture-safe
```

- **`TensorView`** borrows one device buffer: address, `DataType_t`, rank, and up to 8 dims and
  strides in elements. `from_dlpack()` in `cudnn_frontend/ops/dlpack.h` converts a DLPack `DLTensor`.
- **`SeqLayout`** is shared by the packed tensors of one sequence set: the device offsets, and the
  host capacities `max_seqs` and `max_tokens` (and optionally `max_seq_len`, a bound on any one
  sequence that tightens the SDPA backward's grid). The host never reads the offsets, so a plan created for
  a capacity serves any batch that fits it, and a captured CUDA graph replays it.
- **`execute()`** must get tensors with the shapes, strides and dtypes `create()` validated. Two
  calls in flight must not share a workspace.

## Kernel libraries

A kernel is keyed by its configuration: a JSON object naming the op, dtypes, head dims and flags,
never a shape. `kernel_config()` returns the one a call needs; a failed `create()` prints it.

```bash
./my_app --print-kernel-configs > configs.txt        # whatever kernel_config() returned
python -m cudnn.ops configs.txt -o kernels.cudnn      # compile them for this GPU
```

```python
lib = cudnn.ops.compile(configs)        # for the current GPU (cudnn.ops.Target.current())
lib.save("kernels.cudnn")               # KernelLibrary::load_file
blob = lib.to_bytes()                   # KernelLibrary::load_bytes
cudnn.ops.register(lib)                 # KernelLibrary::registry() in this process
```

A library is for one GPU (compute capability and SM count) and one cudnn-frontend version; any other
is refused when it is loaded. Its kernels link against `libtvm_ffi.so` and `libcute_dsl_runtime.so`,
found through the rpath recorded at compile time or through `LD_LIBRARY_PATH`. Writing a library
links each kernel with the system C compiler. Linux only.

`samples/ops/` runs both ops from Python and C++ against fp64 references.

## Current limits

- SDPA: packed Q and KV with one offsets array each, no paged KV, split-KV, sinks, windows or fp8. The
  backward needs head dim 128, equal Q and KV head counts and cutlass-dsl 4.8, and its kernel is
  compiled per sequence capacity (`max_seqs`) and head count.
- KDA: the batch-invariant single-pass schedule only (no d<sub>v</sub> split or piece chain), no checkpoints or
  pooled state indices. The backward has no safe gate or grouped heads yet.
- `cudnn.ops.compile` compiles for the current GPU only.
