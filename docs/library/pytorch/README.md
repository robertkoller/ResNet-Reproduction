# PyTorch Internals

What the library is doing underneath the API you call.

New to the vocabulary? Start with
[`docs/foundations/training-vocabulary.md`](../../foundations/training-vocabulary.md).
Want the architecture rather than the machinery?
[`docs/block/`](../../block/README.md).

---

## Reading order

### [00 — The Stack Beneath `nn.Conv2d`](00-the-stack.md)

Every layer between your Python line and the arithmetic: `nn.Module`,
`torch.nn.functional`, the C++ dispatcher, and the backends (ATen, BLAS, cuDNN,
MPSGraph). Explains why `.to(device)` is the only change needed to run on
Colab, why the same code gives slightly different numbers on different
machines, and why the hand-written convolution is 43,000× slower despite doing
identical arithmetic.

### [01 — BLAS and GEMM](01-blas-and-gemm.md)

The 1979 Fortran interface underneath all of it. The three BLAS levels and why
Level 3 is the only one that can saturate a processor. Arithmetic intensity.
Why convolution gets contorted into a matrix multiply. What LAPACK is.
**Measured on this machine: GEMM sustains 1,709 GFLOP/s against GEMV's 58 — a
28× gap on identical data.**

### [02 — `nn.Module`](02-nn-module.md)

The `__setattr__` interception that makes `self.conv = nn.Conv2d(...)` register
a submodule, and why a plain Python list breaks it silently. `Parameter` vs
tensor, buffers, hooks, `state_dict`, and what `train()`/`eval()` actually do.

### [03 — The `nn` Layers We Use](03-the-nn-layers.md)

`Conv2d`, `BatchNorm2d`, `ReLU`, `Linear`, and `Sequential` from the library
side: full signatures, every argument including the ones you don't use,
what each stores, and what runs underneath.

---

## Room for later

- `04-autograd.md` — the tape, `grad_fn`, how `backward()` walks the graph,
  leaf tensors, `no_grad()`, and why `zero_grad()` is required.
- `05-tensors-and-memory.md` — storage, strides, views vs copies, contiguity,
  `channels_last`, and why `.reshape()` sometimes copies and `.view()` errors.
- `06-dataloader.md` — workers, prefetching, `collate_fn`, and seeding across
  worker processes. Relevant to the data pipeline.
- `07-optimizers.md` — what `SGD` actually stores, how `momentum` and
  `weight_decay` are implemented, and the parameter-group mechanism behind
  learning rate schedules.

---

## Checking any of this yourself

```python
import torch

print(torch.__config__.show())            # build flags, including BLAS_INFO
print(torch.__config__.parallel_info())   # threading backend and counts
print(torch.get_num_threads())
print(torch.backends.mps.is_available())
```

**On this machine:** `BLAS_INFO=accelerate` (Apple's Accelerate framework),
OpenMP with 4 threads of 10 hardware cores, `MKLDNN not found` — expected,
that's an Intel library. On Colab the entire lower half of the stack is
different: CUDA, cuDNN, cuBLAS.
