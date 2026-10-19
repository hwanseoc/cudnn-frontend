# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile kernels for ``cudnn_frontend::ops`` and hand them to C++.

The C++ ops (``include/cudnn_frontend/ops/``) never compile: they look kernels
up in a ``cudnn_frontend::ops::KernelLibrary``. This module produces them, at
run time, for a target GPU::

    lib = cudnn.ops.compile([config, ...], target=cudnn.ops.Target.current())
    lib.save("kernels.cudnn")     # KernelLibrary::load_file
    blob = lib.to_bytes()         # KernelLibrary::load_bytes
    cudnn.ops.register(lib)       # KernelLibrary::registry(), this process

A configuration is the JSON object a C++ op's ``kernel_config()`` returns (or
a dict of the same): the op, dtypes, head dims and flags, never a shape.

Library file layout (read by ``KernelLibrary::load_bytes``): ``b"CUDNNOPS"``,
the header's length as a little-endian uint64, the header (JSON: format,
frontend version, target, and per entry its configuration, entry symbol,
binding metadata, and the byte range of its shared object), then the shared
objects.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

FORMAT = 1

# op name -> "module:function"; the function takes (config dict, Target) and
# returns (compiled CuTeDSL object, entry symbol, binding metadata dict).
_BUILDERS = {
    "kda_forward": "cudnn.ops.kda_forward:_build",
    "sdpa_forward": "cudnn.ops.sdpa_forward:_build",
}

_LOCK = threading.Lock()
_COMPILED: Dict[Tuple[str, "Target"], Tuple[Any, str, Dict[str, Any]]] = {}  # this process's compiles
_REGISTERED: Dict[str, Any] = {}  # canonical config -> what keeps its registry entry alive

Config = Union[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Target:
    """The GPU a library is for: compute capability (major * 10 + minor) and SM count.

    Kernels are architecture-specific SASS, and FROST schedules bake in the SM count."""

    cc: int
    num_sm: int

    @classmethod
    def current(cls, device: int | None = None) -> "Target":
        from cudnn.frost.device import compute_capability, multiprocessor_count, resolve_device

        device = resolve_device(device)
        major, minor = compute_capability(device)
        return cls(cc=major * 10 + minor, num_sm=multiprocessor_count(device))


def canonical(config: Config) -> str:
    """The configuration as C++ prints it: sorted keys, no spaces (values are strings, integers and booleans)."""
    data = json.loads(config) if isinstance(config, str) else dict(config)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


class KernelLibrary:
    """Compiled kernels for one :class:`Target`, keyed by canonical configuration."""

    def __init__(self, target: Target):
        self.target = target
        self._entries: Dict[str, Tuple[Any, str, Dict[str, Any]]] = {}

    def __contains__(self, config: Config) -> bool:
        return canonical(config) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def configs(self) -> list:
        return list(self._entries)

    def to_bytes(self) -> bytes:
        """The library as ``KernelLibrary::load_bytes`` reads it. Links each kernel into a shared object (needs a C compiler)."""
        import cudnn

        entries, blobs, offset = [], [], 0
        for key, (compiled, symbol, meta) in self._entries.items():
            blob = _shared_object(compiled, symbol)
            entries.append({"config": json.loads(key), "symbol": symbol, "meta": meta, "offset": offset, "size": len(blob)})
            blobs.append(blob)
            offset += len(blob)
        major, minor, patch = (int(x) for x in cudnn.__version__.split(".")[:3])
        header = json.dumps(
            {
                "format": FORMAT,
                "fe_version": major * 10000 + minor * 100 + patch,
                "target": {"cc": self.target.cc, "num_sm": self.target.num_sm},
                "producer": _producer(),
                "entries": entries,
            }
        ).encode()
        return b"CUDNNOPS" + struct.pack("<Q", len(header)) + header + b"".join(blobs)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Write :meth:`to_bytes` to ``path`` (``KernelLibrary::load_file``)."""
        tmp = f"{os.fspath(path)}.{os.getpid()}.tmp"
        Path(tmp).write_bytes(self.to_bytes())
        os.replace(tmp, path)


def compile(configs: Iterable[Config], target: Target | None = None) -> KernelLibrary:  # noqa: A001 -- the public verb
    """Compile ``configs`` for ``target`` (default: the current GPU) into a :class:`KernelLibrary`.

    Compiles at most once per configuration and target in a process."""
    from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old

    installed, dsl = cutedsl_state()
    if not installed or cutedsl_too_old(dsl):
        floor = ".".join(map(str, CUTEDSL_MIN_VERSION))
        raise RuntimeError(f"cudnn.ops.compile needs nvidia-cutlass-dsl >= {floor}; found {dsl[1] if dsl else 'none'}")
    target = target or Target.current()
    if target != Target.current():
        raise NotImplementedError(f"cudnn.ops.compile: compiling for {target} on a different GPU ({Target.current()}) is not supported yet")
    lib = KernelLibrary(target)
    for config in configs:
        key = canonical(config)
        with _LOCK:
            built = _COMPILED.get((key, target))
            if built is None:
                data = json.loads(key)
                module, function = _BUILDERS[data["op"]].split(":")
                built = getattr(importlib.import_module(module), function)(data, target)
                _COMPILED[(key, target)] = built
        lib._entries[key] = built
    return lib


def register(lib: KernelLibrary) -> None:
    """Publish ``lib``'s kernels to this process: ``KernelLibrary::registry()`` finds them."""
    import tvm_ffi
    from cudnn.frost.compiled_cache import positional_entry

    if lib.target != Target.current():
        raise ValueError(f"cudnn.ops.register: the library is for {lib.target}, this GPU is {Target.current()}")
    for key, (compiled, symbol, meta) in lib._entries.items():
        fn = positional_entry(compiled)
        if fn is None:
            raise RuntimeError(f"cudnn.ops.register: kernel {key} exposes no tvm-ffi entry")
        text = json.dumps(meta)
        tvm_ffi.register_global_func(f"cudnn.ops.kernel.{key}", fn, override=True)
        tvm_ffi.register_global_func(f"cudnn.ops.meta.{key}", lambda text=text: text, override=True)
        _REGISTERED[key] = compiled


def registered(config: Config) -> bool:
    return canonical(config) in _REGISTERED


def ensure_registered(config: str) -> None:
    """Compile and register ``config`` for the current GPU unless it already is."""
    if config not in _REGISTERED:
        register(compile([config]))


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


def _producer() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ("nvidia-cutlass-dsl", "apache-tvm-ffi"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = "unknown"
    return out


def _runtime_libraries() -> list:
    import cutlass.cute as cute

    return [str(p) for p in cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)]


_LINKED: Dict[str, bytes] = {}


def _shared_object(compiled: Any, symbol: str) -> bytes:
    """``compiled``'s exported object linked into a shared object that resolves its runtime libraries by SONAME, with an
    rpath to where they are installed here."""
    with tempfile.TemporaryDirectory(prefix="cudnn_ops_") as tmp:
        obj = os.path.join(tmp, "kernel.o")
        entry = getattr(compiled, "_compiled_cache_entry", None)
        if entry is not None:  # reloaded from the compiled-plan cache: the exported object is on disk
            shutil.copyfile(os.path.join(entry, "kernel.o"), obj)
        else:
            compiled.export_to_c(obj, function_name=symbol)
        data = Path(obj).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with _LOCK:
            if digest in _LINKED:
                return _LINKED[digest]
        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            raise RuntimeError("cudnn.ops: writing a kernel library links each kernel into a shared object and needs a C compiler ($CC, cc, gcc or clang)")
        libs = _runtime_libraries()
        rpath = ":".join(sorted({os.path.dirname(p) for p in libs}))
        out = os.path.join(tmp, "kernel.so")
        # -Bsymbolic: the kernel's own references bind inside it, whatever else the process loaded.
        cmd = [cc, "-shared", "-Wl,-Bsymbolic", f"-Wl,-rpath,{rpath}", "-o", out, obj, *libs]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"cudnn.ops: linking a kernel failed:\n{' '.join(cmd)}\n{proc.stderr}")
        linked = Path(out).read_bytes()
    with _LOCK:
        _LINKED[digest] = linked
    return linked
