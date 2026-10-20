# cudnn_frontend::ops samples

Python compiles the kernels; Python or C++ executes them. Every sample checks its results against an
fp64 reference, kept in `reference.py` and `reference.h` so the samples show only the API.

| Sample | Compiles | Executes |
|---|---|---|
| `python_compile_and_execute.py` | Python, on first use | Python: SDPA and KDA, forward and backward |
| `cpp_execute_from_memory.cpp` | Python, piped as bytes | C++, `KernelLibrary::load_bytes`: KDA forward |
| `cpp_execute_from_file.cpp` | Python, into a file | C++, `KernelLibrary::load_file`: KDA forward |
| `sdpa_forward.cpp`, `sdpa_backward.cpp`, `kda_backward.cpp` | Python, into a file | C++: the other ops |

No Python runs in the C++ processes.

```bash
cd samples/ops
python python_compile_and_execute.py
```

```bash
CUDNN=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")
build() { g++ -std=c++17 -I include -I $CUDA_HOME/include -I $CUDNN/include samples/ops/$1.cpp \
              -L $CUDA_HOME/lib64 -L $CUDNN/lib -lcudart -l:libcudnn.so.9 -ldl -o $1; }

# from memory: the library goes from Python to C++ through a pipe
build cpp_execute_from_memory
./cpp_execute_from_memory --config > configs.txt           # the kernel configuration the sample needs
python -m cudnn.ops configs.txt -o - | ./cpp_execute_from_memory

# from a file (any machine with this GPU can write it)
build cpp_execute_from_file
./cpp_execute_from_file --config > configs.txt
python -m cudnn.ops configs.txt -o kernels.cudnn
./cpp_execute_from_file kernels.cudnn
```

The per-op C++ samples run the same way as `cpp_execute_from_file`. A kernel library is for one GPU
(compute capability and SM count) and one cudnn-frontend version. The kernels find
`libtvm_ffi.so` and `libcute_dsl_runtime.so` through the rpath they were linked with, or through
`LD_LIBRARY_PATH` on another machine.
