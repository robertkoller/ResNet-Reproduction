# 03 — The `nn` Layers We Use

The four layers in the basic block, from the library's side: full signatures,
every argument, what's stored, and what runs.

For *why* each layer is in the architecture, see
[`docs/block/`](../../block/README.md). This page is the API.

---

## The module/functional pairing

Every layer exists twice in PyTorch:

| module form | functional form |
|---|---|
| `nn.Conv2d(16, 32, 3)` | `F.conv2d(input, weight, bias, ...)` |
| `nn.BatchNorm2d(16)` | `F.batch_norm(input, running_mean, ...)` |
| `nn.ReLU()` | `F.relu(input)` |
| `nn.Linear(64, 10)` | `F.linear(input, weight, bias)` |

The module **owns state and calls the function**. It exists so `nn.Module`'s
registration can find the weights — see [02](02-nn-module.md).

Use the module form in models. Use the functional form when you already hold
the weights, which is exactly the situation in
`tests/test_conv_from_scratch.py`.

For stateless operations like ReLU, the choice is pure style. Many
implementations write `F.relu(out)` in `forward` and never create a module at
all.

---

## `nn.Conv2d`

```python
nn.Conv2d(
    in_channels,            # required
    out_channels,           # required
    kernel_size,            # required — int or (h, w)
    stride=1,               # int or (h, w)
    padding=0,              # int, (h, w), or "same" / "valid"
    dilation=1,             # spacing between kernel elements
    groups=1,               # 1 = normal; in_channels = depthwise
    bias=True,
    padding_mode="zeros",   # or "reflect", "replicate", "circular"
    device=None,
    dtype=None,
)
```

### What it stores

| attribute | shape | notes |
|---|---|---|
| `weight` | `(out_channels, in_channels // groups, kh, kw)` | the kernels |
| `bias` | `(out_channels,)` | `None` when `bias=False` |

Initialized by **Kaiming uniform** with `a=√5` by default — a historical choice
that is not the He initialization the paper specifies, so the models in
`models/` reinitialize every convolution explicitly.

### Arguments you use

**`stride`** — window step. 2 halves the output. Only the block's first conv
uses it.

**`padding`** — border of zeros. `padding=1` preserves size for a 3×3 kernel.
`padding="same"` also works but fails for even strides, so prefer the explicit
integer.

**`bias=False`** — because BatchNorm follows and would cancel it. See
[`docs/block/01-convolution.md`](../../block/01-convolution.md).

### Arguments you don't, but should recognise

**`groups`** — splits channels into independent groups, each conv'd separately.
`groups=in_channels` gives a **depthwise** convolution, the basis of MobileNet
and EfficientNet. ResNeXt uses intermediate values. Cuts parameters and FLOPs
sharply.

**`dilation`** — spreads the kernel out with gaps, enlarging the receptive
field without more parameters. Used in segmentation, where you need broad
context at full resolution.

**`padding_mode`** — `"reflect"` and `"replicate"` avoid the dark border zeros
create. Rarely matters at 32×32.

### What runs

`F.conv2d` → dispatcher → a backend kernel. On CPU that's typically im2col plus
a BLAS GEMM; on NVIDIA it's cuDNN, which picks among several algorithms
(implicit GEMM, Winograd, FFT) by benchmarking; on an Apple Silicon Mac it's MPSGraph. See
[00](00-the-stack.md) and [01](01-blas-and-gemm.md).

---

## `nn.BatchNorm2d`

```python
nn.BatchNorm2d(
    num_features,              # required — the channel count
    eps=1e-5,                  # added to variance before the square root
    momentum=0.1,              # running-statistics update rate
    affine=True,               # learn gamma and beta
    track_running_stats=True,  # keep running statistics for eval
    device=None,
    dtype=None,
)
```

### What it stores

| name | kind | shape |
|---|---|---|
| `weight` (gamma) | parameter | `(C,)` |
| `bias` (beta) | parameter | `(C,)` |
| `running_mean` | buffer | `(C,)` |
| `running_var` | buffer | `(C,)` |
| `num_batches_tracked` | buffer | scalar |

Two parameters per channel — 32 for `BatchNorm2d(16)`, verified. Buffers are
saved in checkpoints but never see a gradient.

### Arguments worth understanding

**`eps`** — stops division by zero when a channel has no variance. Also a
numerical safety margin in float16.

**`momentum=0.1`** — the running-statistics update rate:

```
running_mean = (1 − 0.1) × running_mean + 0.1 × batch_mean
```

**Completely unrelated to SGD's `momentum: 0.9`** in `configs/`. Same word,
different mechanism, opposite convention — PyTorch's BatchNorm momentum is the
weight on the *new* value, whereas SGD's is the weight on the *old* one. A
genuinely unfortunate naming collision.

**`affine=False`** drops gamma and beta, leaving pure normalization with no
parameters. Occasionally used in the last BN of a residual block, initialized
so the block starts as an exact identity. Not in this paper.

**`track_running_stats=False`** makes it use batch statistics even in eval
mode. Sometimes used with tiny batches. Don't — it makes evaluation depend on
batch composition.

### The relatives

Same idea, different pooling dimensions:

| layer | normalizes over | independent of batch? |
|---|---|---|
| `BatchNorm2d` | batch, height, width — per channel | no |
| `LayerNorm` | all features of one sample | yes |
| `GroupNorm` | groups of channels within one sample | yes |
| `InstanceNorm2d` | height, width — per sample per channel | yes |

BatchNorm's dependence on other images in the batch is what makes it awkward at
small batch sizes and in sequence models. `LayerNorm` is why transformers use
that instead.

---

## `nn.ReLU`

```python
nn.ReLU(inplace=False)
```

No parameters, no buffers, no state. `max(0, x)` elementwise.

`inplace=True` overwrites the input buffer instead of allocating. Saves memory;
the only hazard is modifying a tensor autograd still needs, which surfaces as:

```
RuntimeError: a variable needed for gradient computation has been
modified by an inplace operation
```

Set `inplace=False` and it goes away.

Because it's stateless you can share one instance across a `forward` — your
block does exactly that, using `self.relu` twice.

The family, for recognition: `LeakyReLU` (small slope for negatives, avoids
dead units), `ELU`, `GELU` (the transformer default), `SiLU`/`Swish`. All
smoother, all more expensive. ResNet v1 uses plain ReLU.

---

## `nn.Linear`

Not in the block, but it's the final classifier of the network.

```python
nn.Linear(in_features, out_features, bias=True)
```

Computes `y = xWᵀ + b`. Stores `weight` of shape `(out_features, in_features)`
and `bias` of shape `(out_features,)`.

The classifier is `nn.Linear(64, 10)`: `64 × 10 + 10 = 650` parameters.

Note the weight is stored **transposed** relative to the maths — `(out, in)`
rather than `(in, out)` — for memory layout reasons. It trips people reading
shapes.

This is a bare GEMM, no contortion required. See [01](01-blas-and-gemm.md).

---

## `nn.Sequential`

The container used to assemble blocks into stages.

```python
self.stage1 = nn.Sequential(
    BasicBlock(16, 16),
    BasicBlock(16, 16),
    BasicBlock(16, 16),
)
```

Registers each child and runs them in order, so `forward` becomes
`out = self.stage1(x)`.

From a list, unpack with `*`:

```python
blocks = [BasicBlock(16, 16) for _ in range(n)]
self.stage1 = nn.Sequential(*blocks)
```

Use `nn.ModuleList` instead when you need to write the loop yourself — it
registers but doesn't run. Use `nn.ModuleDict` for name-keyed access.

**Never a plain Python list.** Registration fails silently; see
[02](02-nn-module.md).

---

## Reading the docs yourself

Every layer page at `pytorch.org/docs/stable/nn.html` has the same structure:

1. **Signature** — arguments and defaults
2. **Description** — the mathematics
3. **Shape** — exact input and output shapes, with formulas. *Read this first.*
4. **Variables** — what's stored, and its shape
5. **Examples** — runnable

The **Shape** section resolves almost every shape error.

From the terminal:

```bash
.venv/bin/python -c "from torch import nn; help(nn.Conv2d)" | head -60
```

---

## Terms

- **Module form / functional form** — stateful object vs stateless function.
- **`groups`** — splitting channels into independent convolution groups.
- **Depthwise convolution** — `groups == in_channels`.
- **`dilation`** — gaps in the kernel, widening the receptive field.
- **Affine** — BatchNorm's learned scale and shift.
- **`track_running_stats`** — whether BatchNorm maintains eval-time statistics.
- **`nn.Sequential` / `ModuleList` / `ModuleDict`** — containers that register.

---

Previous: [02 — nn.Module](02-nn-module.md) ·
Back to: [pytorch/README.md](README.md)
