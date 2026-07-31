# 05 — Putting the Block Together

The four pieces assembled, traced end to end with real shapes and real
parameter counts.

Prerequisites: documents [00](00-tensors-and-shapes.md) through
[04](04-shortcut-and-addition.md).

---

## The block in full

```
        input x  (N, in_channels, H, W)
          │
          ├──────────────────────────────────┐
          │                                  │
          ▼                                  │
    Conv2d(in→out, 3×3, stride=s,            │
           padding=1, bias=False)            │
          │                                  │  shortcut path
          ▼                                  │
    BatchNorm2d(out)                         │  None      → x unchanged
          │                                  │  option A  → subsample + zero-pad
          ▼                                  │  option B/C→ 1×1 conv + BN
        ReLU                                 │
          │                                  │
          ▼                                  │
    Conv2d(out→out, 3×3, stride=1,           │
           padding=1, bias=False)            │
          │                                  │
          ▼                                  │
    BatchNorm2d(out)                         │
          │                                  ▼
          └──────────────►  ( + )  ◄─────────┘   ← skipped when residual=False
                              │
                              ▼
                            ReLU
                              │
                              ▼
                       output (N, out_channels, H/s, W/s)
```

---

## Tracing shapes: the ordinary block

`BasicBlock(16, 16, stride=1)`, input `(128, 16, 32, 32)`:

| step | shape | note |
|---|---|---|
| input `x` | `(128, 16, 32, 32)` | |
| `identity = x` | `(128, 16, 32, 32)` | saved before anything runs |
| conv1 | `(128, 16, 32, 32)` | stride 1, padding 1 → size preserved |
| bn1 | `(128, 16, 32, 32)` | never changes shape |
| relu | `(128, 16, 32, 32)` | never changes shape |
| conv2 | `(128, 16, 32, 32)` | stride 1 |
| bn2 | `(128, 16, 32, 32)` | |
| `+ identity` | `(128, 16, 32, 32)` | shapes match, no shortcut object needed |
| relu | `(128, 16, 32, 32)` | |

Only conv layers ever change shape. BatchNorm, ReLU, and the addition all
preserve it exactly. That narrows shape debugging to one suspect.

---

## Tracing shapes: the downsampling block

`BasicBlock(16, 32, stride=2, shortcut=<option A>)`, input `(128, 16, 32, 32)`:

| step | shape | note |
|---|---|---|
| input `x` | `(128, 16, 32, 32)` | |
| `identity = x` | `(128, 16, 32, 32)` | |
| conv1 | `(128, 32, 16, 16)` | **stride 2** halves H and W, channels 16→32 |
| bn1 | `(128, 32, 16, 16)` | |
| relu | `(128, 32, 16, 16)` | |
| conv2 | `(128, 32, 16, 16)` | stride 1 — only the first conv downsamples |
| bn2 | `(128, 32, 16, 16)` | |
| `shortcut(x)` | `(128, 32, 16, 16)` | subsampled and zero-padded to match |
| `+` | `(128, 32, 16, 16)` | now legal |
| relu | `(128, 32, 16, 16)` | |

The two things to notice:

- **Only the first conv gets the stride.** Apply stride 2 to both and you'd
  downsample twice per block, ending at 8×8 instead of 16×16.
- **The shortcut exists solely to make the addition legal.** Without it, that
  `+` raises a size-mismatch `RuntimeError`.

---

## Parameter accounting

The formulas from [01](01-convolution.md) and [02](02-batch-normalization.md):

- conv: `in × out × 3 × 3`
- BatchNorm: `2 × channels`

### A 16→16 block

| component | calculation | parameters |
|---|---|---|
| conv1 | 16 × 16 × 9 | 2,304 |
| bn1 | 2 × 16 | 32 |
| conv2 | 16 × 16 × 9 | 2,304 |
| bn2 | 2 × 16 | 32 |
| **total** | | **4,672** |

Verified against PyTorch, and asserted in `tests/test_block.py`.

### Every block type in ResNet-20

| block | parameters |
|---|---|
| 16 → 16 | 4,672 |
| 16 → 32 (downsampling) | 13,952 |
| 32 → 32 | 18,560 |
| 32 → 64 (downsampling) | 55,552 |
| 64 → 64 | 73,984 |

Parameters roughly quadruple per stage, because doubling both the input and
output channel counts multiplies `in × out` by four. The deepest stage holds
most of the network's weights despite working on the smallest images.

### The whole of ResNet-20

| part | contents | parameters |
|---|---|---|
| stem | conv 3→16 + BN | 464 |
| stage 1 | 3 × (16→16) | 14,016 |
| stage 2 | (16→32) + 2 × (32→32) | 51,072 |
| stage 3 | (32→64) + 2 × (64→64) | 203,520 |
| classifier | fully connected 64→10, with bias | 650 |
| **total** | | **269,722** |

**269,722 ≈ 0.27M**, matching the paper's table exactly. This arithmetic is
the architecture's correctness proof: an implementation reporting this number
is almost certainly structured correctly.

Note stage 3 alone holds 75% of the parameters.

---

## Reading the implementation

The block is defined in `models/blocks.py`. A few points about its structure
are worth drawing out, because they are easy to get subtly wrong and none of
them raise an error when they are.

### In the constructor

**`super().__init__()` comes first.** Skipping it breaks parameter
registration silently — see [06](06-pytorch-mechanics.md).

**The first convolution carries the stride; the second is always stride 1.**
A block downsamples once, not twice.

**The first convolution maps `in_channels → out_channels`; the second maps
`out_channels → out_channels`.** After the first, the data is already at the
output width.

**Both normalization layers take `out_channels`,** because they normalize what
the convolution produced.

**The two BatchNorm instances are separate objects.** Each holds its own
gamma, beta, and running statistics; sharing one would be a genuine bug. The
single ReLU instance, by contrast, is reused — it is stateless.

### In the forward pass

**`identity = x` is captured before anything else runs.** Not a copy — both
names refer to the same tensor. That is safe because nothing modifies `x` in
place; every operation writes into a separate result.

**The convolution path starts from `x`, not from an intermediate.** This is
the fork in the diagram above: the conv path and the shortcut path both begin
at the block's input.

**The shortcut transforms `x`, not the conv output.** Applying it to the conv
output would defeat the purpose — the point is a path that bypasses the
convolutions entirely.

**The addition comes before the final ReLU.** See
[03](03-relu-and-nonlinearity.md) for why that ordering matters and what it
buys.

**Plain mode skips the shortcut call and the addition, and nothing else.**
Same layers, same parameter count, same initialization. That single difference
is the experiment.

---

## What a correct block satisfies

The checks in `tests/test_block.py`:

| property | expected |
|---|---|
| stride-1 output shape | `(2, 16, 32, 32)` in → `(2, 16, 32, 32)` out |
| parameter count, 16→16 | exactly 4,672 |
| residual vs. plain parameter count | identical |
| output values | finite, no `NaN` |
| residual vs. plain output values | different |
| downsampling shape | `(2, 16, 32, 32)` → `(2, 32, 16, 16)` |

The third is the one that matters most. A parameter-count mismatch between
residual and plain mode means the shortcut is carrying weights, which
confounds the comparison the entire project rests on.

---

Previous: [04 — The Shortcut and the Addition](04-shortcut-and-addition.md) ·
Next: [06 — PyTorch Mechanics](06-pytorch-mechanics.md)
