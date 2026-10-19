# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile kernel configurations into a library file for ``cudnn_frontend::ops::KernelLibrary::load_file``.

    python -m cudnn.ops CONFIGS -o kernels.cudnn

CONFIGS is a file of configurations, one JSON object per line: what a C++ op's
``kernel_config()`` returns, or what a failed ``create()`` names. Compiles for
the current GPU.
"""

import argparse

from cudnn.ops import _library

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m cudnn.ops", description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", help="file with one kernel configuration (JSON) per line")
    parser.add_argument("-o", "--output", required=True, help="library file to write")
    args = parser.parse_args()
    with open(args.configs) as f:
        configs = sorted({_library.canonical(line) for line in f if line.strip()})
    lib = _library.compile(configs)
    lib.save(args.output)
    print(f"wrote {len(lib)} kernel(s) for {lib.target} to {args.output}")
