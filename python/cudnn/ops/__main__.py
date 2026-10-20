# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile kernel configurations into a library for ``cudnn_frontend::ops::KernelLibrary``.

    python -m cudnn.ops CONFIGS -o kernels.cudnn      # KernelLibrary::load_file
    python -m cudnn.ops CONFIGS -o - | my_app          # bytes on stdout: KernelLibrary::load_bytes

CONFIGS is a file of configurations, one JSON object per line: what a C++ op's
``kernel_config()`` returns, or what a failed ``create()`` names. Compiles for
the current GPU.
"""

import argparse
import os
import sys

from cudnn.ops import _library

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m cudnn.ops", description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", help="file with one kernel configuration (JSON) per line")
    parser.add_argument("-o", "--output", required=True, help="library file to write, or - for stdout")
    args = parser.parse_args()
    with open(args.configs) as f:
        configs = sorted({_library.canonical(line) for line in f if line.strip()})
    to_stdout = args.output == "-"
    if to_stdout:
        # Only the library may reach stdout: whatever compiling prints (Python or native) goes to stderr.
        stdout = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    lib = _library.compile(configs)
    if to_stdout:
        with os.fdopen(stdout, "wb") as out:
            out.write(lib.to_bytes())
    else:
        lib.save(args.output)
    print(f"wrote {len(lib)} kernel(s) for {lib.target} to {'stdout' if to_stdout else args.output}", file=sys.stderr)
