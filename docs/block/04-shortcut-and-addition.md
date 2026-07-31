# 04 — The Shortcut and the Addition

One `+` sign. It's the entire contribution of the paper, and the whole reason
this project is an experiment rather than a tutorial.

---

## What it is

```python
out = out + identity
```

An elementwise addition. Two tensors of identical shape, added position by
position. No weights, no learned anything, negligible compute.

```
F(x) shape (2, 16, 32, 32)        identity shape (2, 16, 32, 32)

    [0.3, -0.1, 0.8, ...]     +      [1.2, 0.4, -0.5, ...]
                              =
                  [1.5, 0.3, 0.3, ...]
```

Position `[0, 5, 12, 7]` of one is added to position `[0, 5, 12, 7]` of the
other. That's all "elementwise" means.

---

## What it changes about the learning problem

Without the shortcut, the two convs must produce the block's complete output:

```
output = H(x)          "given this input, produce the right output"
```

With the shortcut, they produce only the **difference** from the input:

```
output = F(x) + x      so    F(x) = H(x) − x
```

`F` is the **residual** — what's left over after accounting for what you
already had.

The distinction sounds cosmetic. It isn't, and here's the concrete case that
shows why.

### The identity case

Suppose the best thing this block could do is leave its input alone. (In a
110-layer network, plenty of blocks are in exactly that situation — the network
doesn't need 110 layers' worth of transformation.)

**Without a shortcut**, the two convs must learn to reproduce their input
exactly. Think about what that demands: 4,608 conv weights arranged so that
after two convolutions, two batch normalizations, and a ReLU, every one of
16,384 numbers comes out unchanged. That's a precise, delicate configuration in
a 4,608-dimensional space, and gradient descent has to find it by feel.

**With a shortcut**, the convs need to output **zero**. Push the weights toward
zero and you're done. Weight decay is already pushing them that way. It's
essentially free.

And as noted in [03](03-relu-and-nonlinearity.md), it's exact: the previous
block ended with a ReLU so `x ≥ 0`, which makes `ReLU(0 + x) = x` precisely,
not approximately.

### The general case

Even when "do nothing" isn't right, the argument holds in weaker form. The
block starts from a sensible answer — its input — and learns a correction, not
a construction. Nudging something roughly right is an easier optimization
problem than building the answer from random noise.

**Be careful how you state this.** He et al. present it as a hypothesis, not a
proof. The paper says "we hypothesize" and "may help to precondition the
problem." They demonstrate that it works; they do not demonstrate why. Saying
so is the difference between having read the paper and having read a blog post
about it.

---

## Why the parameter-free property is the whole experiment

The addition has **no weights**. That's not a minor efficiency note — it's what
makes this project a controlled experiment.

Because the shortcut costs nothing:

- ResNet-20 and plain-20 have **identical parameter counts**
- identical layer counts
- identical FLOPs (to within rounding)
- and with the same seed, **identical initial weights**

So when plain-56 does worse than plain-20 on *training* error while ResNet-56
does better, there is exactly one candidate explanation. No confound about
model capacity, no "well it had more parameters."

This is why the parameter-count assertion in `tests/test_block.py` is not
bureaucratic box-ticking. If `residual=True` and `residual=False` report
different counts, something is carrying weights that shouldn't, and every
downstream number is contaminated.

---

## The shape problem

Addition requires identical shapes. Most of the time that's automatic — a block
inside a stage takes `(128, 16, 32, 32)` and produces `(128, 16, 32, 32)`.

But at each stage boundary the block downsamples and doubles channels:

```
x    = (128, 16, 32, 32)        ← what comes in
F(x) = (128, 32, 16, 16)        ← what the convs produce

x + F(x)   →   RuntimeError: The size of tensor a (16) must match
               the size of tensor b (32) at non-singleton dimension 1
```

Two mismatches at once: 16 channels versus 32, and 32×32 versus 16×16.
Something must reshape the shortcut path.

---

## The three options the paper tests

### Option A — zero-padded identity

Fix the spatial size by taking every other pixel (matching what stride 2 did).
Fix the channel count by appending 16 channels of pure zeros.

```
(128, 16, 32, 32)
      │
      ├─ subsample:  take every 2nd row and column  →  (128, 16, 16, 16)
      │
      └─ pad:        append 16 all-zero channels    →  (128, 32, 16, 16)

result:  channels 0-15  = the original features, spatially subsampled
         channels 16-31 = all zeros
```

**Zero parameters.** The shortcut stays a pure identity mapping, which is what
the paper's theoretical argument is about. This is what He et al. use for
**every CIFAR experiment**, and therefore the default here.

It's also the one most often implemented wrong, which is why it gets a
dedicated test: feed in `(1, 16, 32, 32)`, confirm the output is
`(1, 32, 16, 16)`, and confirm channels 16–31 are exactly zero.

### Option B — projection only where shapes change

Use a 1×1 convolution with stride 2 on the shortcut, but only in the blocks
where the shape actually changes. Everywhere else, plain identity.

A 1×1 conv is a real convolution with a 1×1 kernel — it mixes channels at each
position without looking at neighbours. `16 → 32` costs `16 × 32 × 1 × 1 = 512`
parameters plus a BatchNorm.

Adds a small number of parameters. Used for the ImageNet models.

### Option C — projection on every shortcut

A 1×1 conv on all shortcuts, including the ones where shapes already match.
Adds a lot of parameters and breaks the clean identity story.

### The results

| option | ImageNet top-1 error |
|---|---|
| A | 25.03% |
| B | 24.52% |
| C | 24.19% |

All three crush the plain network. The gaps between them are small, so the
paper concludes projections aren't essential and sticks with A for CIFAR and B
for the deeper models.

Worth internalising: the shortcut being an **identity** is what matters. The
particular way you patch up shape mismatches is a detail.

---

## Why the block takes the shortcut as an argument

The block does not decide which option to use. It takes a shortcut object as a
constructor argument and calls it on the way to the addition — see
`BasicBlock` in `models/blocks.py`.

Three reasons:

1. **One block class serves all three options.** No `if option == "A"` branches
   buried inside the block.
2. **The shortcut ablation becomes a configuration change**, not a code edit —
   `shortcut: B` in the YAML and rerun.
3. **Each shortcut is testable on its own.** Option A's zero-padding is the
   error-prone one; isolating it means it can be checked directly rather than
   inferring its correctness from whether the whole network trains.

`shortcut=None` means "shapes already match, use the input unchanged." That's
the common case — in ResNet-20, seven of nine blocks need no shortcut object at
all.

---

## Plain mode

```python
if self.residual:
    if self.shortcut is not None:
        identity = self.shortcut(x)
    out = out + identity
```

When `residual=False`, skip the addition. Change nothing else — same convs,
same BatchNorms, same ReLUs, same parameter count, same initialization.

One `if`. That is the entire difference between the treatment group and the
control group, and it is the whole project.

---

## Terms from this page

- **Residual** — the difference between the desired output and the input;
  `F(x) = H(x) − x`.
- **Shortcut / skip connection** — the path carrying `x` around the convs.
- **Identity mapping** — a function that returns its input unchanged.
- **Elementwise addition** — adding two same-shaped tensors position by
  position.
- **Projection shortcut** — a 1×1 conv used to fix shape mismatches (options
  B and C).
- **Option A / B / C** — the paper's three answers to the shape-mismatch
  problem.
- **Plain network** — the same architecture with the addition removed; the
  control group.

---

Previous: [03 — ReLU and Nonlinearity](03-relu-and-nonlinearity.md) ·
Next: [05 — Putting the Block Together](05-putting-it-together.md)
