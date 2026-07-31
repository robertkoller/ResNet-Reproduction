# 03 — ReLU and Nonlinearity

The simplest layer in the network, and the one without which depth would be
worthless.

---

## What it does

```
ReLU(x) = max(0, x)
```

Negative numbers become zero. Positive numbers pass through unchanged. Applied
independently to every single number in the tensor — no mixing, no parameters,
nothing learned.

```
         output
           │
         3 ┤        ╱
           │      ╱
         2 ┤    ╱
           │  ╱
         1 ┤╱
           │
    ───────┼────────── input
    -3  -1 │0  1  2  3
           │
```

The name is **Re**ctified **L**inear **U**nit, borrowed from electronics where
a rectifier passes current one way and blocks it the other.

---

## Why any nonlinearity is required

This is the part worth genuinely understanding, because it explains why the
whole architecture is shaped the way it is.

A convolution is a **linear** operation — output is a weighted sum of inputs,
nothing more. And here is the killer fact:

> The composition of two linear functions is another linear function.

Stack conv A and conv B with nothing between them, and there exists a single
conv C that produces identical output. Stack twenty, and one conv still
replaces all of them.

So a 20-layer network with no nonlinearity has exactly the representational
power of a 1-layer network. Not "slightly better." Identical. All that depth
buys you literally nothing.

Straight-line functions can only ever draw straight decision boundaries. The
boundary between "cat" and "dog" in pixel space is not remotely straight.

ReLU breaks linearity — that `max(0, x)` bend is not a straight line — and with
that bend, stacking layers starts producing genuinely richer functions. Each
layer can now build on the last instead of collapsing into it.

That's the deal: **conv layers do the mixing, nonlinearities make depth
count.** Remove the ReLUs and ResNet-110 becomes an elaborate way to compute a
single linear map.

---

## Why ReLU rather than something smoother

The older standard was the sigmoid, an S-curve squashing everything into
(0, 1). It has a fatal property: for inputs beyond roughly ±5 the curve is
almost flat, so its **derivative is almost zero**.

Gradients flow backward by multiplication through layers. Multiply twenty
near-zero numbers and you get an unimaginably small number. The early layers
receive no useful gradient and never learn. That's the classic **vanishing
gradient** problem, and it's why networks were stuck at a handful of layers for
years.

ReLU's derivative is exactly:

- **1** for positive input
- **0** for negative input

For every positive activation, the gradient passes backward **completely
unattenuated**. Multiply twenty 1s and you still have 1. That single property
is most of why deep networks became trainable.

It's also almost free — a comparison and a select, no exponentials.

---

## Dead ReLUs, the one real drawback

If a unit's input is negative for every training example, its output is always
zero, so its gradient is always zero, so its weights never update, so its input
stays negative forever. The unit is **dead** — permanently outputting zero and
consuming parameters for nothing.

A large learning rate can kill units in bulk early in training.

Variants exist to fix this (Leaky ReLU passes a small fraction of negatives,
ELU and GELU curve smoothly near zero). ResNet uses plain ReLU, BatchNorm keeps
activations centred enough that mass death doesn't happen, and you should use
plain ReLU because that's what you're reproducing.

---

## The two ReLUs in the block, and their placement

```
x ──► conv ──► BN ──► ReLU ──► conv ──► BN ──► ( + ) ──► ReLU ──► out
                       ▲                         ▲         ▲
                       │                         │         │
              first ReLU: between            addition   second ReLU:
              the two convs                             AFTER the add
```

**The first ReLU** sits between the two convs. Without it, the two convs would
collapse into one and the block would have half its intended power.

**The second ReLU comes after the addition, not before.** This is the detail
the project notes flag, and it matters:

```python
out = out + identity
out = self.relu(out)        # correct — ReLU after the add

out = self.relu(out)
out = out + identity        # wrong — but it still trains
```

The wrong version doesn't crash and doesn't obviously misbehave. It just isn't
the architecture in the paper, and the numbers stop being comparable.

There's also a real consequence. Because the previous block also ended with a
ReLU, `x` arriving at this block is **already non-negative**. So if the convs
learn to output zero, the block computes `ReLU(0 + x) = ReLU(x) = x` — an
*exact* identity mapping. That's the paper's central argument working
literally, and it only holds with the ReLU placed after the addition.

A later paper by the same authors (*Identity Mappings in Deep Residual
Networks*, 2016) proposes a **pre-activation** ordering — BN and ReLU before
each conv, nothing after the addition. It performs slightly better and a lot of
code online uses it. It is **not** what you're reproducing. If you use it, drop
the comparison to the paper's table.

---

## `inplace=True`

```python
self.relu = nn.ReLU(inplace=True)
```

Normally an operation allocates a new tensor for its result. `inplace=True`
tells PyTorch to overwrite the input tensor instead. Saves memory, which
matters when you're holding activations for 110 layers at batch size 128.

It's safe here because nothing needs the pre-ReLU values afterwards.

**The one place it would be a bug:** never apply an in-place operation to the
tensor you're holding as the shortcut. If `identity = x` and you then modify
`x` in place, you've corrupted the value you're about to add. In the standard
block ordering this doesn't arise — `identity` is captured before anything
touches `x`, and the first ReLU acts on `out`, not on `x`. But it's worth
knowing why the ordering in `forward` is what it is.

If you ever hit a *"a variable needed for gradient computation has been
modified by an inplace operation"* error, this is the cause. Set
`inplace=False` and it goes away.

---

## One instance, used twice

```python
self.relu = nn.ReLU(inplace=True)   # created once in __init__
...
out = self.relu(out)                # used twice in forward
out = self.relu(out)
```

Fine, and standard. ReLU has no parameters and no state, so a single instance
is a pure function. Creating two would work identically and just waste an
object.

Contrast with BatchNorm: those **must** be separate instances, because each
holds its own gamma, beta, and running statistics. Reusing one BatchNorm for
both convs would be a genuine bug.

Rule of thumb: **layers with parameters need their own instance; stateless
layers can be shared.**

---

## The signature

```python
nn.ReLU(inplace=False)
```

That's the whole API.

Docs: `https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html`

---

## Terms from this page

- **Nonlinearity / activation function** — a function applied elementwise that
  isn't a straight line. ReLU is one.
- **Linear** — output is a weighted sum of inputs; composes into more linear.
- **Vanishing gradient** — gradients shrinking toward zero as they propagate
  backward, so early layers stop learning.
- **Saturating** — a function whose derivative approaches zero for large
  inputs, like sigmoid.
- **Dead ReLU** — a unit stuck outputting zero forever.
- **In-place operation** — one that overwrites its input rather than allocating
  new memory.
- **Pre-activation** — the ResNet v2 ordering. Not what you're building.

---

Previous: [02 — Batch Normalization](02-batch-normalization.md) ·
Next: [04 — The Shortcut and the Addition](04-shortcut-and-addition.md)
