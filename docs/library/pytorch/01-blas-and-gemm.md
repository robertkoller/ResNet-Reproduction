# 01 — BLAS and GEMM

The 1979 Fortran interface that every neural network on earth ultimately runs
on.

---

## What BLAS is

**BLAS** — Basic Linear Algebra Subprograms — is a *specification*, not a
library. It defines a set of function names and argument conventions for
fundamental linear algebra operations. Anyone can write an implementation; if
it matches the interface, existing code can link against it unchanged.

That separation is the whole trick. NumPy, PyTorch, MATLAB, R, and most
scientific software call BLAS. Intel, Apple, AMD, and NVIDIA each ship a
version tuned to their own silicon. Swap the library, keep the code, get the
speed.

It dates to 1979 and the interface has barely changed. Function names still
carry Fortran-era abbreviations: `SGEMM` is **S**ingle-precision **GE**neral
**M**atrix **M**ultiply. The `S` is the dtype (`D` double, `C` complex, `Z`
double complex), and that naming convention outlived the language it was
designed for.

---

## The three levels

BLAS is organised by how much arithmetic you get per byte of memory moved.
This is the single most important idea on this page.

### Level 1 (1979) — vector-vector

`y = αx + y`, dot products, norms. For vectors of length *n*: **O(n)** data,
**O(n)** flops.

### Level 2 (1988) — matrix-vector

`y = αAx + βy` (GEMV). **O(n²)** data, **O(n²)** flops.

### Level 3 (1990) — matrix-matrix

`C = αAB + βC` (GEMM). **O(n²)** data, **O(n³)** flops.

### Why level 3 wins

Look at the ratio of arithmetic to memory traffic — the **arithmetic
intensity**:

| level | data | flops | flops per byte |
|---|---|---|---|
| 1 | O(n) | O(n) | constant |
| 2 | O(n²) | O(n²) | constant |
| 3 | O(n²) | O(n³) | **grows with n** |

Levels 1 and 2 do a fixed small amount of arithmetic per number loaded. They're
**memory-bandwidth-bound** — the processor idles waiting for RAM, and a faster
CPU buys you nothing.

Level 3 loads a tile of the matrices into cache once and does *n* operations on
each element. It's **compute-bound**, and can actually saturate the hardware.

Measured on this machine, float32 CPU:

| operation | time | throughput |
|---|---|---|
| GEMV, 2048×2048 | 0.15 ms | 58 GFLOP/s |
| GEMM, 2048×2048 | 10.1 ms | **1,709 GFLOP/s** |

**GEMM achieves about 28× the throughput of GEMV on identical data.** Not 28×
more work in the same time — 28× better use of the same hardware, per
operation.

That number is the reason for everything else on this page.

---

## GEMM

```
C ← α·A·B + β·C
```

Three matrices, two scalars. The `β·C` term means it accumulates into an
existing matrix rather than overwriting, which lets it serve as a building
block without extra copies.

Fifty years of effort have gone into this one function: cache blocking, loop
reordering, register tiling, hand-written SIMD, prefetching, multithreading,
and per-microarchitecture tuning. A naive triple loop reaches perhaps 1–2% of
peak. A good GEMM reaches 80–95%.

Look again at the table above. GEMM at 128×128 gets 943 GFLOP/s; at 512 and
beyond it settles around 1,700. Small matrices can't amortise the setup and
don't fill the cache blocks. Bigger is better, up to memory limits — which is
an argument for larger batch sizes, and why batch size 1 wastes a GPU.

---

## Why this makes convolution a matrix multiply

Here's where it connects to this project.

Convolution is not naturally a matrix multiply. It's a sliding window. But
GEMM is so overwhelmingly the best-optimised routine in computing that it is
worth **contorting the problem to fit it**.

That's **im2col** — covered in
[`docs/block/01-convolution.md`](../../block/01-convolution.md). Unfold every
patch the kernel will see into a column, stack the columns into a matrix, and
the entire convolution becomes:

```
Y = W X
```

with `W` the reshaped weights `(C_out, C_in·k·k)` and `X` the patch matrix
`(C_in·k·k, positions)`.

For a 16→32 conv on a 32×32 image that's `(32, 144) @ (144, 1024)`. One GEMM
call. Verified in the from-scratch tests to match `nn.Conv2d` at exactly zero
difference.

The cost is memory: `X` duplicates every input value up to `k²` times. For a
3×3 kernel that's 9× the input in scratch space. **It's worth it anyway**,
because the alternative is a memory-bound loop nest at 1/28th the throughput.

That trade — burn memory to reach GEMM — is one of the load-bearing ideas in
practical deep learning.

---

## What else runs on GEMM

Nearly everything:

- **Fully connected layers** — literally a GEMM, no contortion needed. Your
  final `64 → 10` classifier.
- **Convolutions** — via im2col, or via Winograd/FFT for particular sizes.
- **Attention in transformers** — `QKᵀ` and the value projection are GEMMs.
- **The backward pass** — gradients with respect to weights and inputs are both
  GEMMs, which is why backprop costs roughly 2× the forward pass rather than
  something worse.

Modern accelerators have gone further and built silicon for it directly:
NVIDIA's Tensor Cores and Apple's AMX units are, functionally, small matrix
multipliers in hardware. When people say a chip does "N TFLOPS for AI," they
mean *at GEMM*.

---

## Which implementation you're actually using

**On this Mac:** `torch.__config__.show()` reports `BLAS_INFO=accelerate` —
Apple's **Accelerate** framework, tuned for Apple Silicon and able to use the
AMX matrix coprocessor. `ATen/Parallel` reports OpenMP with 4 threads of 10
hardware cores.

**On Colab:** a CUDA build, where GPU matrix multiplies go to cuBLAS and
convolutions to cuDNN.

The common implementations you'll see referenced:

| implementation | notes |
|---|---|
| Reference BLAS | Netlib's Fortran original. Correct, slow, a baseline. |
| **Accelerate** | Apple's. What you're running. |
| OpenBLAS | Open source, hand-tuned assembly per microarchitecture. |
| Intel MKL | Intel's. Usually fastest on Intel CPUs. |
| BLIS | Research framework; a small kernel plus generated surroundings. |
| cuBLAS | NVIDIA GPU. The path taken on CUDA hardware. |

Check yours:

```python
import torch
print([line for line in torch.__config__.show().split("\n") if "BLAS" in line])
print(torch.__config__.parallel_info())
```

---

## LAPACK, the layer above

**LAPACK** (Linear Algebra PACKage, 1992) provides the higher-level routines —
solving linear systems, eigenvalues, SVD, QR and Cholesky decompositions.

It is deliberately built **on top of BLAS Level 3**. Its algorithms are
restructured into blocked forms specifically so the inner work becomes GEMM
calls. That's why `torch.linalg.svd` is fast: not because someone optimised
SVD, but because it was rewritten to spend its time inside GEMM.

You won't call LAPACK directly in this project, but it's behind
`torch.linalg.*`, and the pattern — *restructure your algorithm until the hot
loop is a GEMM* — is the same one im2col applies.

---

## What to take from this

1. **Arithmetic intensity is the thing that matters.** Flops per byte moved
   determines whether you're compute-bound or memory-bound, and memory-bound
   code cannot be rescued by a faster processor.
2. **GEMM is the fast path.** Reshaping a problem into a matrix multiply, even
   at real cost in memory, usually wins.
3. **Interfaces outlive implementations.** BLAS is 45 years old and still the
   substrate. The Fortran naming is a fossil worth recognising.
4. **The library boundary is where your performance lives.** One `nn.Conv2d`
   call reaching a tuned GEMM beats hand-written loops by four orders of
   magnitude — measured, in the from-scratch tests, at 43,654×.

---

## Terms

- **BLAS** — the specification for basic linear algebra routines.
- **GEMM** — GEneral Matrix Multiply, `C ← αAB + βC`. BLAS Level 3.
- **GEMV** — GEneral Matrix-Vector multiply. BLAS Level 2.
- **Level 1 / 2 / 3** — vector-vector, matrix-vector, matrix-matrix.
- **Arithmetic intensity** — flops performed per byte of memory moved.
- **Compute-bound / memory-bound** — limited by the processor versus limited by
  memory bandwidth.
- **im2col** — unfolding patches into columns so convolution becomes a GEMM.
- **LAPACK** — higher-level linear algebra built on BLAS Level 3.
- **Accelerate / OpenBLAS / MKL / cuBLAS** — BLAS implementations.
- **Tensor Core / AMX** — hardware units that perform small matrix multiplies
  directly.

---

Previous: [00 — The Stack](00-the-stack.md) ·
Next: [02 — nn.Module](02-nn-module.md)
