/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Compiled kernels for cudnn_frontend::ops, supplied by the caller.
//
// The kernels are CuTeDSL, so Python compiles them, at run time, in any
// process or on any machine (cudnn.ops.compile). C++ never runs Python: it
// takes the result from a KernelLibrary, read from a file or from memory
// (what cudnn.ops.KernelLibrary.save / to_bytes wrote), or from the kernels
// Python registered in this process (cudnn.ops.register). An op's create()
// looks its kernels up here and never compiles.
//
// Kernels are keyed by their static configuration (a JSON object: op, dtypes,
// head dims, flags), never by a shape. A library holds kernels for one GPU
// (compute capability and SM count) and is rejected on any other.
//
// A loaded kernel links against libtvm_ffi.so and libcute_dsl_runtime.so,
// which must be resolvable when the library is loaded: the Python package
// loads them, a C++ process finds them through LD_LIBRARY_PATH or the rpath
// the kernel was linked with. Calls go through the tvm-ffi C ABI, mirrored
// below, so no tvm-ffi header is needed.
//
// Linux only; elsewhere loading a library reports GRAPH_NOT_SUPPORTED.

#include "tensor_view.h"
#include "../../cudnn_frontend_version.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace cudnn_frontend {
namespace ops {

class KernelLibrary;

namespace detail {

// ---------------------------------------------------------------------------
// The tvm-ffi C ABI (tvm/ffi/c_api.h, apache-tvm-ffi 0.1) as far as calling a
// compiled kernel needs it, and the DLPack tensor tensors are passed as.
// ---------------------------------------------------------------------------
namespace ffi {

enum TypeIndex : int32_t {
    kNone        = 0,
    kInt         = 1,
    kBool        = 2,
    kFloat       = 3,
    kDLTensorPtr = 7,
    kSmallStr    = 11,
    kObject      = 64,  // first heap-object type index; everything >= is ref counted
    kStr         = 65,
    kError       = 67,
    kArray       = 71,
};

struct Object {
    uint64_t combined_ref_count;
    int32_t type_index;
    uint32_t padding;
    void (*deleter)(void *self, int flags);
};

struct Any {
    int32_t type_index;
    uint32_t small_len;  // a small string's length; zero otherwise
    union {
        int64_t v_int64;
        double v_float64;
        void const *v_ptr;
        Object *v_obj;
        char v_bytes[8];
    };
};

struct ByteArray {
    char const *data;
    size_t size;
};

struct ErrorCell {  // the fields after the Object header
    ByteArray kind;
    ByteArray message;
};

// DLPack's DLTensor (dlpack/dlpack.h).
struct DLDevice {
    int32_t device_type;  // kDLCUDA = 2
    int32_t device_id;
};
struct DLDataType {
    uint8_t code;  // kDLInt = 0, kDLFloat = 2, kDLBfloat = 4
    uint8_t bits;
    uint16_t lanes;
};
struct DLTensor {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
};

using SafeCall       = int (*)(void *self, Any const *args, int32_t num_args, Any *result);
using GetGlobal      = int (*)(ByteArray const *name, void **out);
using FunctionCall   = int (*)(void *func, Any const *args, int32_t num_args, Any *result);
using DecRef         = int (*)(void *obj);
using MoveFromRaised = void (*)(void **result);

static_assert(sizeof(Object) == 24, "tvm-ffi Object header layout");
static_assert(sizeof(Any) == 16, "tvm-ffi Any layout");
static_assert(sizeof(DLTensor) == 48, "DLPack DLTensor layout");

}  // namespace ffi

inline ffi::Any
any_int(int64_t v) {
    ffi::Any a{};
    a.type_index = ffi::kInt;
    a.v_int64    = v;
    return a;
}

inline ffi::Any
any_float(double v) {
    ffi::Any a{};
    a.type_index = ffi::kFloat;
    a.v_float64  = v;
    return a;
}

inline ffi::Any
any_none() {
    return ffi::Any{};
}

inline ffi::Any
any_tensor(ffi::DLTensor const *t) {
    if (t == nullptr) {
        return any_none();
    }
    ffi::Any a{};
    a.type_index = ffi::kDLTensorPtr;
    a.v_ptr      = t;
    return a;
}

inline bool
dl_dtype(DataType_t t, ffi::DLDataType &out) {
    switch (t) {
        case DataType_t::HALF:
            out = {2, 16, 1};
            return true;
        case DataType_t::BFLOAT16:
            out = {4, 16, 1};
            return true;
        case DataType_t::FLOAT:
            out = {2, 32, 1};
            return true;
        case DataType_t::INT32:
            out = {0, 32, 1};
            return true;
        case DataType_t::INT64:
            out = {0, 64, 1};
            return true;
        default:
            return false;
    }
}

// A DLTensor over a view. The view's dim and stride arrays back it, so the view
// must outlive the call it is passed to.
inline ffi::DLTensor
dl_tensor(TensorView const &v, int device) {
    ffi::DLTensor t{};
    t.data   = v.data;
    t.device = {2, device};
    t.ndim   = v.rank;
    dl_dtype(v.dtype, t.dtype);  // validated by the op
    t.shape   = const_cast<int64_t *>(v.dim.data());
    t.strides = const_cast<int64_t *>(v.stride.data());
    return t;
}

#if defined(__linux__)

// The tvm-ffi entry points this file calls. libtvm_ffi.so is loaded by whoever
// came first: Python, or a kernel this library loaded (it depends on it).
struct Runtime {
    ffi::GetGlobal get_global            = nullptr;
    ffi::FunctionCall call               = nullptr;
    ffi::DecRef dec_ref                  = nullptr;
    ffi::MoveFromRaised move_from_raised = nullptr;
    void *array_ctor                     = nullptr;  // the "ffi.Array" global function
};

inline Runtime &
runtime() {
    static Runtime rt;
    return rt;
}

inline error_t
bind_runtime() {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    auto &rt = runtime();
    if (rt.call != nullptr) {
        return {error_code_t::OK, ""};
    }
    void *lib = ::dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        lib == nullptr,
        error_code_t::GRAPH_NOT_SUPPORTED,
        "libtvm_ffi.so is not loaded: load a kernel library first, or import cudnn in Python");
    Runtime r;
    r.get_global       = reinterpret_cast<ffi::GetGlobal>(::dlsym(lib, "TVMFFIFunctionGetGlobal"));
    r.call             = reinterpret_cast<ffi::FunctionCall>(::dlsym(lib, "TVMFFIFunctionCall"));
    r.dec_ref          = reinterpret_cast<ffi::DecRef>(::dlsym(lib, "TVMFFIObjectDecRef"));
    r.move_from_raised = reinterpret_cast<ffi::MoveFromRaised>(::dlsym(lib, "TVMFFIErrorMoveFromRaised"));
    RETURN_CUDNN_FRONTEND_ERROR_IF(!r.get_global || !r.call || !r.dec_ref || !r.move_from_raised,
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "libtvm_ffi.so does not export the tvm-ffi C API this build expects");
    ffi::ByteArray name = {"ffi.Array", 9};
    RETURN_CUDNN_FRONTEND_ERROR_IF(r.get_global(&name, &r.array_ctor) != 0 || r.array_ctor == nullptr,
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "libtvm_ffi.so has no global function ffi.Array");
    rt = r;
    return {error_code_t::OK, ""};
}

// The pending tvm-ffi error as text, consumed.
inline std::string
take_error() {
    auto const &rt = runtime();
    void *err      = nullptr;
    rt.move_from_raised(&err);
    if (err == nullptr) {
        return "unknown tvm-ffi error";
    }
    std::string msg = "tvm-ffi error";
    if (static_cast<ffi::Object const *>(err)->type_index == ffi::kError) {
        auto const *cell =
            reinterpret_cast<ffi::ErrorCell const *>(static_cast<char const *>(err) + sizeof(ffi::Object));
        msg =
            std::string(cell->kind.data, cell->kind.size) + ": " + std::string(cell->message.data, cell->message.size);
    }
    rt.dec_ref(err);
    return msg;
}

// dlopen needs a path: an anonymous in-memory file gives a kernel one without
// writing it anywhere. Modules are loaded once per distinct content and never
// unloaded (a captured CUDA graph may keep launching their kernels).
inline error_t
load_module(std::string const &bytes, void *&handle) {
    static std::mutex mu;
    static std::unordered_map<std::string, void *> loaded;
    std::lock_guard<std::mutex> lock(mu);
    if (auto it = loaded.find(bytes); it != loaded.end()) {
        handle = it->second;
        return {error_code_t::OK, ""};
    }
    int const fd = static_cast<int>(::syscall(SYS_memfd_create, "cudnn_frontend_ops", 1u /* MFD_CLOEXEC */));
    RETURN_CUDNN_FRONTEND_ERROR_IF(fd < 0, error_code_t::GRAPH_NOT_SUPPORTED, "memfd_create failed");
    size_t done = 0;
    while (done < bytes.size()) {
        ssize_t const n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n <= 0) {
            break;
        }
        done += static_cast<size_t>(n);
    }
    std::string const path = "/proc/self/fd/" + std::to_string(fd);
    handle                 = done == bytes.size() ? ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL) : nullptr;
    if (handle == nullptr) {
        char const *e = ::dlerror();
        ::close(fd);
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                std::string("cannot load a kernel: ") + (e ? e : "short write") +
                    " (libtvm_ffi.so and libcute_dsl_runtime.so must be resolvable)"};
    }
    // The descriptor stays open: once closed, its number (and so the path the
    // loader keys this module by) could name a different module later.
    loaded.emplace(bytes, handle);
    return {error_code_t::OK, ""};
}

#endif  // __linux__

// One compiled kernel: an entry point of a loaded module, or a function Python
// registered in this process.
class Kernel {
   public:
    Kernel()               = default;
    Kernel(Kernel const &) = delete;
    Kernel &
    operator=(Kernel const &) = delete;
    ~Kernel() {
#if defined(__linux__)
        if (function_ != nullptr) {
            runtime().dec_ref(function_);
        }
#endif
    }

    json meta;  // what the compiler reported for binding the kernel's arguments

    // Calls the kernel; arguments are borrowed for the call.
    error_t
    call(ffi::Any const *args, int num_args) const {
#if defined(__linux__)
        ffi::Any result{};
        int rc = 0;
        // The CuTeDSL runtime initializes a function on its first call under a
        // process-wide lock that concurrent first calls can deadlock on
        // (nvidia-cutlass-dsl 4.7), so first calls are serialized.
        if (warm_.load(std::memory_order_acquire)) {
            rc = invoke(args, num_args, &result);
        } else {
            static std::mutex first_call;
            std::lock_guard<std::mutex> lock(first_call);
            rc = invoke(args, num_args, &result);
            if (rc == 0) {
                warm_.store(true, std::memory_order_release);
            }
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            rc != 0, error_code_t::GRAPH_EXECUTION_FAILED, "cudnn_frontend::ops kernel failed: " + take_error());
        if (result.type_index >= ffi::kObject && result.v_obj != nullptr) {
            runtime().dec_ref(result.v_obj);
        }
        return {error_code_t::OK, ""};
#else
        (void)args;
        (void)num_args;
        return {error_code_t::GRAPH_NOT_SUPPORTED, "cudnn_frontend::ops kernels run on Linux only"};
#endif
    }

   private:
    friend class cudnn_frontend::ops::KernelLibrary;

    int
    invoke(ffi::Any const *args, int num_args, ffi::Any *result) const {
#if defined(__linux__)
        return entry_ != nullptr ? entry_(nullptr, args, num_args, result)
                                 : runtime().call(function_, args, num_args, result);
#else
        (void)args;
        (void)num_args;
        (void)result;
        return -1;
#endif
    }

    ffi::SafeCall entry_ = nullptr;  // from a loaded module
    void *function_      = nullptr;  // from the registry (owned reference)
    mutable std::atomic<bool> warm_{false};
};

// A tuple of integers as a tvm-ffi Array (an owned reference).
class IntArray {
   public:
    IntArray()                 = default;
    IntArray(IntArray const &) = delete;
    IntArray &
    operator=(IntArray const &) = delete;
    IntArray(IntArray &&other) noexcept : obj_(other.obj_) { other.obj_ = nullptr; }
    IntArray &
    operator=(IntArray &&other) noexcept {
        std::swap(obj_, other.obj_);
        return *this;
    }
    ~IntArray() {
#if defined(__linux__)
        if (obj_ != nullptr) {
            runtime().dec_ref(obj_);
        }
#endif
    }

    error_t
    assign(std::vector<int64_t> const &values) {
#if defined(__linux__)
        std::vector<ffi::Any> items;
        for (int64_t v : values) {
            items.push_back(any_int(v));
        }
        auto const &rt = runtime();
        ffi::Any result{};
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            rt.call(rt.array_ctor, items.data(), static_cast<int32_t>(items.size()), &result) != 0 ||
                result.type_index != ffi::kArray,
            error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
            "cannot build a tuple argument");
        if (obj_ != nullptr) {
            rt.dec_ref(obj_);
        }
        obj_ = result.v_obj;
        return {error_code_t::OK, ""};
#else
        (void)values;
        return {error_code_t::GRAPH_NOT_SUPPORTED, "cudnn_frontend::ops kernels run on Linux only"};
#endif
    }

    ffi::Any
    any() const {
        ffi::Any a{};
        a.type_index = ffi::kArray;
        a.v_obj      = obj_;
        return a;
    }

   private:
    ffi::Object *obj_ = nullptr;
};

// The current device's ordinal, compute capability (major * 10 + minor) and SM count.
inline error_t
current_device(int &device, int &cc, int &num_sm) {
    int major = 0, minor = 0;
    using cudnn_frontend::detail::cuda_device_get_attribute;
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        cudnn_frontend::detail::cuda_get_device(&device) != cudaSuccess ||
            cuda_device_get_attribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cuda_device_get_attribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
            cuda_device_get_attribute(&num_sm, cudaDevAttrMultiProcessorCount, device) != cudaSuccess,
        error_code_t::INVALID_CUDA_DEVICE,
        "cannot query the current CUDA device");
    cc = major * 10 + minor;
    return {error_code_t::OK, ""};
}

}  // namespace detail

class KernelLibrary {
   public:
    // A library cudnn.ops.KernelLibrary.save() wrote.
    error_t
    load_file(std::string const &path) {
        std::ifstream f(path, std::ios::binary);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!f, error_code_t::INVALID_VALUE, "cannot read kernel library " + path);
        std::string const bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return load_bytes(bytes.data(), bytes.size());
    }

    // The same, from memory (cudnn.ops.KernelLibrary.to_bytes()). Adds to what is loaded.
    error_t
    load_bytes(void const *data, size_t size) {
#if defined(__linux__)
        // Layout: "CUDNNOPS", uint64 header length, header JSON, payload.
        auto const *p        = static_cast<char const *>(data);
        uint64_t header_size = 0;
        RETURN_CUDNN_FRONTEND_ERROR_IF(size < 16 || std::memcmp(p, "CUDNNOPS", 8) != 0,
                                       error_code_t::INVALID_VALUE,
                                       "not a cudnn_frontend::ops kernel library");
        std::memcpy(&header_size, p + 8, 8);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            header_size > size - 16, error_code_t::INVALID_VALUE, "truncated kernel library");
        json const header = json::parse(p + 16, p + 16 + header_size, nullptr, false);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            header.is_discarded() || !header.is_object(), error_code_t::INVALID_VALUE, "corrupt kernel library header");
        RETURN_CUDNN_FRONTEND_ERROR_IF(header.value("format", 0) != FORMAT,
                                       error_code_t::INVALID_VALUE,
                                       "kernel library format " + std::to_string(header.value("format", 0)) +
                                           "; this build reads format " + std::to_string(FORMAT));
        int const version = header.value("fe_version", 0);
        RETURN_CUDNN_FRONTEND_ERROR_IF(version != CUDNN_FRONTEND_VERSION,
                                       error_code_t::INVALID_VALUE,
                                       "kernel library built by cudnn-frontend " + std::to_string(version) +
                                           "; this is " + std::to_string(CUDNN_FRONTEND_VERSION));
        CHECK_CUDNN_FRONTEND_ERROR(check_target(header.at("target")));

        char const *payload       = p + 16 + header_size;
        size_t const payload_size = size - 16 - header_size;
        for (auto const &e : header.at("entries")) {
            auto const offset = e.at("offset").get<uint64_t>();
            auto const length = e.at("size").get<uint64_t>();
            RETURN_CUDNN_FRONTEND_ERROR_IF(offset > payload_size || length > payload_size - offset,
                                           error_code_t::INVALID_VALUE,
                                           "truncated kernel library");
            void *handle = nullptr;
            CHECK_CUDNN_FRONTEND_ERROR(detail::load_module(std::string(payload + offset, length), handle));
            auto const symbol = "__tvm_ffi_" + e.at("symbol").get<std::string>();
            auto kernel       = std::make_shared<detail::Kernel>();
            kernel->entry_    = reinterpret_cast<detail::ffi::SafeCall>(::dlsym(handle, symbol.c_str()));
            kernel->meta      = e.value("meta", json::object());
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                kernel->entry_ == nullptr, error_code_t::INVALID_VALUE, "kernel library entry has no " + symbol);
            kernels_[e.at("config").dump()] = std::move(kernel);
        }
        return detail::bind_runtime();  // the kernels loaded libtvm_ffi.so
#else
        (void)data;
        (void)size;
        return {error_code_t::GRAPH_NOT_SUPPORTED, "cudnn_frontend::ops kernels run on Linux only"};
#endif
    }

    // The kernels Python registered in this process (cudnn.ops.register), looked up when an op is created.
    static KernelLibrary
    registry() {
        KernelLibrary lib;
        lib.registry_ = true;
        return lib;
    }

    // The kernel for a configuration; an error naming it when there is none.
    error_t
    find(json const &config, std::shared_ptr<detail::Kernel const> &out) const {
        std::string const key = config.dump();
        if (auto it = kernels_.find(key); it != kernels_.end()) {
            out = it->second;
            return {error_code_t::OK, ""};
        }
#if defined(__linux__)
        if (registry_ && detail::bind_runtime().is_good()) {
            auto const &rt           = detail::runtime();
            std::string name         = "cudnn.ops.kernel." + key;
            void *fn                 = nullptr;
            void *meta_fn            = nullptr;
            detail::ffi::ByteArray n = {name.data(), name.size()};
            if (rt.get_global(&n, &fn) == 0 && fn != nullptr) {
                auto kernel       = std::make_shared<detail::Kernel>();
                kernel->function_ = fn;
                name              = "cudnn.ops.meta." + key;
                n                 = {name.data(), name.size()};
                detail::ffi::Any result{};
                if (rt.get_global(&n, &meta_fn) == 0 && meta_fn != nullptr &&
                    rt.call(meta_fn, nullptr, 0, &result) == 0) {
                    std::string text;
                    if (result.type_index == detail::ffi::kStr) {
                        auto const *s = reinterpret_cast<detail::ffi::ByteArray const *>(
                            reinterpret_cast<char const *>(result.v_obj) + sizeof(detail::ffi::Object));
                        text.assign(s->data, s->size);
                        rt.dec_ref(result.v_obj);
                    } else if (result.type_index == detail::ffi::kSmallStr) {
                        text.assign(result.v_bytes, result.small_len);
                    }
                    kernel->meta = json::parse(text, nullptr, false);
                }
                if (meta_fn != nullptr) {
                    rt.dec_ref(meta_fn);
                }
                RETURN_CUDNN_FRONTEND_ERROR_IF(kernel->meta.is_discarded() || !kernel->meta.is_object(),
                                               error_code_t::GRAPH_NOT_SUPPORTED,
                                               "the registry has kernel " + key + " but no metadata for it");
                out = std::move(kernel);
                return {error_code_t::OK, ""};
            }
        }
#endif
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "the kernel library has no kernel for " + key + "; compile it with cudnn.ops.compile(['" + key +
                    "']) and save, load or register the result"};
    }

    static constexpr int FORMAT = 1;

   private:
    // A library holds kernels for one GPU: the cubins are architecture-specific
    // and FROST kernels bake the SM count into their schedules.
    static error_t
    check_target(json const &target) {
        int device = 0, cc = 0, num_sm = 0;
        CHECK_CUDNN_FRONTEND_ERROR(detail::current_device(device, cc, num_sm));
        int const want_cc = target.value("cc", 0), want_sm = target.value("num_sm", 0);
        RETURN_CUDNN_FRONTEND_ERROR_IF(want_cc != cc || want_sm != num_sm,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "kernel library is for compute capability " + std::to_string(want_cc) +
                                           " with " + std::to_string(want_sm) + " SMs; device " +
                                           std::to_string(device) + " is " + std::to_string(cc) + " with " +
                                           std::to_string(num_sm) + " SMs");
        return {error_code_t::OK, ""};
    }

    bool registry_ = false;
    std::unordered_map<std::string, std::shared_ptr<detail::Kernel const>> kernels_;
};

}  // namespace ops
}  // namespace cudnn_frontend
