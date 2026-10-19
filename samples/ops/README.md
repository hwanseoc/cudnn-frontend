# cudnn_frontend::ops samples

Each op (SDPA and KDA, forward and backward) is called once on three packed sequences of different
lengths and checked against an fp64 reference. The references and helpers are in `reference.py` and `reference.h`, so the samples show
only the API.

| Sample | What it shows |
|---|---|
| `sdpa_forward.py`, `sdpa_backward.py`, `kda_forward.py`, `kda_backward.py` | The Python ops. A kernel compiles on first use. |
| `sdpa_forward.cpp`, `sdpa_backward.cpp`, `kda_forward.cpp`, `kda_backward.cpp` | The C++ ops, with kernels from a library file that Python compiled. No Python runs in the C++ process. `sdpa_backward.cpp` runs the forward with its log-sum-exp first. |

Python:

```bash
cd samples/ops
for s in sdpa_forward sdpa_backward kda_forward kda_backward; do python $s.py; done
```

C++:

```bash
CUDNN=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")
for s in sdpa_forward sdpa_backward kda_forward kda_backward; do
  g++ -std=c++17 -I include -I $CUDA_HOME/include -I $CUDNN/include samples/ops/$s.cpp \
      -L $CUDA_HOME/lib64 -L $CUDNN/lib -lcudart -l:libcudnn.so.9 -ldl -o $s
done

for s in sdpa_forward sdpa_backward kda_forward kda_backward; do ./$s --config; done > configs.txt   # the kernels the samples need
python -m cudnn.ops configs.txt -o kernels.cudnn   # compile them (any machine with this GPU)
for s in sdpa_forward sdpa_backward kda_forward kda_backward; do ./$s kernels.cudnn; done
```

A kernel library is for one GPU (compute capability and SM count) and one cudnn-frontend version.
The kernels find `libtvm_ffi.so` and `libcute_dsl_runtime.so` through the rpath they were linked with,
or through `LD_LIBRARY_PATH` on another machine.
