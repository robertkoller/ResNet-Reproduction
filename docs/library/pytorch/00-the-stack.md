# 00 — The Stack Beneath `nn.Conv2d`

What actually happens between your Python line and the arithmetic.

---

## The layers

```
    the model code            block(x)                        Python
         │
         ▼
    nn.Module            __call__ → hooks → forward      Python
         │
         ▼
    torch.nn.functional  F.conv2d(...)                   Python, thin
         │
         ▼
    the dispatcher       pick kernel by device + dtype   C++
         │
         ├──────────────┬──────────────┬─────────────┐
         ▼              ▼              ▼             ▼
    ATen CPU        cuDNN          MPSGraph      oneDNN        C++/CUDA/Metal
         │              │              │             │
         ▼              ▼              ▼             ▼
    BLAS            CUDA cores     Apple GPU     AVX/AMX       assembly
    (Accelerate)
```

Only the top two boxes are Python. Everything below is compiled native code,
and that's where essentially all the time goes.

---

## Layer by layer

### `nn.Module` — organisation only

`nn.Conv2d` is a Python object holding a weight tensor and some configuration.
Calling it runs `__call__`, which fires hooks and then `forward`, which
immediately delegates. It performs no arithmetic itself. Its whole job is
owning parameters and making them discoverable — see
[02 — nn.Module](02-nn-module.md).

### `torch.nn.functional` — the stateless entry point

`F.conv2d(input, weight, bias, stride, padding)` is the actual operation, as a
function with no state.

The relationship is worth internalising:

- `nn.Conv2d` — an **object** that owns weights and calls the function
- `F.conv2d` — the **function**, which takes weights as an argument

Same computation. You use the module form so the optimizer can find the
parameters; you use the functional form when you already hold the weights, as
in the from-scratch tests.

### The dispatcher — where Python ends

This is the interesting part. `torch.conv2d` isn't one implementation; it's a
name with many registered implementations. The **dispatcher** picks one at
runtime based on:

- **device** — CPU, CUDA, MPS
- **dtype** — float32, float16, bfloat16
- **layout** — dense, sparse, channels-last
- **autograd state** — whether a backward graph is being recorded

`x.to("mps")` changes nothing about the Python you write. It changes which
kernel the dispatcher selects. That single mechanism is why your identical code
runs on your M4 and on a Colab T4.

Every dispatch costs a few microseconds. Irrelevant for a 128×16×32×32 batch;
very relevant if you call an operator in a Python loop, which is exactly why
the naive convolution in the tests is 43,000× slower than `nn.Conv2d` — the
arithmetic is the same, the overhead is not.

### The backends

**ATen** (A TENsor library) is PyTorch's C++ core. Every operator has a
reference CPU implementation here, which then calls out to whatever specialised
library is available.

| backend | used on | covers |
|---|---|---|
| BLAS | CPU everywhere | matrix multiply, the foundation |
| oneDNN / MKL-DNN | Intel CPUs | fused conv + BN + ReLU |
| cuDNN | NVIDIA GPUs | conv, pooling, normalization, RNNs |
| MPSGraph | Apple Silicon | the Metal path an Apple Silicon Mac uses |

**On this machine specifically:** `torch.__config__.show()` reports
`BLAS_INFO=accelerate` — PyTorch was built against Apple's **Accelerate**
framework, its hand-tuned BLAS for Apple Silicon. `ATen/Parallel` reports the
OpenMP backend with 4 threads out of 10 hardware cores. `MKLDNN not found`,
which is expected — that's an Intel library.

On Colab you'll get a completely different lower half of this stack: CUDA,
cuDNN, and NVIDIA's BLAS. Same Python, different machine code. It's also one
reason seeds don't reproduce across the two platforms.

---

## What this explains

### Why the naive convolution is 43,000× slower

Both versions do the same multiply-accumulates. The hand-written one does them
one at a time through the Python interpreter, paying a dispatch on every single
scalar operation. `nn.Conv2d` makes **one** dispatch and hands the entire
problem to native code that runs vectorized across 4 threads.

You are not competing on algorithm. You're competing on execution model, and
you lose by four orders of magnitude before writing a line.

### Why `.to(device)` is all you change

There's no CUDA in the model code because device selection happens at the dispatcher
in C++. Your `get_device()` returning `"mps"` or `"cuda"` steers that choice
and nothing else needs to know.

### Why the same code gives slightly different numbers on different machines

Different backends use different summation orders, different vector widths, and
sometimes different algorithms entirely for the same operation. Float addition
isn't associative, so a different order gives a different last bit. That
compounds over 64,000 iterations.

This is why `docs/block/` says to note the platform in the deviations section,
and why you shouldn't mix Mac runs and Colab runs inside one comparison.

### Why fused operations exist

`conv → BN → ReLU` as three separate calls writes the full activation tensor to
memory three times. Backends like cuDNN and oneDNN provide **fused** kernels
that do all three in one pass, keeping intermediates in registers. Often a
large win, because these operations are memory-bandwidth-bound rather than
compute-bound.

You don't need to do anything about this — `torch.compile` and the backends
handle it — but it explains why the layer boundaries you write in Python are
not the boundaries that execute.

---

## Seeing it yourself

```python
import torch

print(torch.__config__.show())          # build flags, including BLAS_INFO
print(torch.__config__.parallel_info()) # thread counts and parallel backend
print(torch.get_num_threads())          # CPU threads ATen will use
print(torch.backends.mps.is_available())
```

To watch the dispatcher choose:

```python
import torch
from torch.utils.flop_counter import FlopCounterMode

with FlopCounterMode(display=True):
    torch.nn.Conv2d(16, 32, 3, padding=1)(torch.randn(1, 16, 32, 32))
```

---

## Terms

- **ATen** — PyTorch's C++ tensor library; every operator's home.
- **Dispatcher** — the runtime mechanism selecting an implementation by device
  and dtype.
- **Kernel** — in this context, one compiled implementation of an operation.
  (Unrelated to a convolution kernel. The word is overloaded; context
  disambiguates.)
- **Backend** — the library providing kernels: BLAS, cuDNN, MPSGraph, oneDNN.
- **Fusion** — combining several operations into one kernel to avoid
  round-trips to memory.
- **Accelerate** — Apple's BLAS/LAPACK framework; what your build uses.

---

Next: [01 — BLAS and GEMM](01-blas-and-gemm.md)
