# Q&A — Architecture

How the network is structured, and why it is structured that way.

Fuller treatment in [`../block/README.md`](../block/README.md).

---

## Is `n` the depth? Does `6n + 2` equal 110, or does `n`?

`n` is **not** the depth. `n` is the knob; depth is what comes out of it.

`n = 18` gives `depth = 6 × 18 + 2 = 110`.

Where the 6 and the 2 come from:

- **1** stem convolution at the start (3×3, 16 channels)
- **3 stages**, each holding `n` blocks, each block holding **2** convolutions
  → `3 × n × 2 = 6n`
- **1** fully connected layer for the 10-way classification

Total `1 + 6n + 1 = 6n + 2`. So `n` is literally "how many blocks per stage."

| n | depth |
|---|---|
| 3 | 20 |
| 5 | 32 |
| 7 | 44 |
| 9 | 56 |
| 18 | 110 |

`depth` is a `@property` on `Configuration` rather than a YAML field, because
it is fully determined by `n`. Storing both would allow `n: 9, depth: 110` and
leave the code with no way to know which to believe.

---

## Why is an image `3 × 32 × 32` when it is two-dimensional?

The image genuinely is 32×32 — 1,024 pixels, laid out in two dimensions. But a
colour pixel is not one number; it is three: red, green, blue. So there are
1,024 positions × 3 numbers = 3,072 values, stored as **three separate 32×32
grids stacked**.

The `3` is not a spatial direction. There is no depth into the screen. It is
*how many measurements exist at each location*. Read the shape as "3
measurements per position, over a 32-by-32 grid of positions."

After the first convolution the tensor is `(16, 32, 32)` — sixteen grids, and
they are no longer colours. Each is "how strongly did filter *k* respond here."
The network invents those meanings during training.

Full treatment, including the batch dimension and NCHW ordering, in
[`../block/00-tensors-and-shapes.md`](../block/00-tensors-and-shapes.md).

---

## What is a channel?

One grid of numbers, where the value at each position says how strongly one
particular thing is present at that location.

At the input, channels are red, green, and blue — the only place in the network
where they have human-readable meanings. Everywhere else they are learned
feature detectors with no name.

"Channel" and "feature map" mean the same thing; the second is the usual word
once past the input layer. The name "channel" comes from signal processing,
where a colour image was transmitted as three separate signal channels.

---

## Why does the network reduce spatial size while increasing channels?

| stage | resolution | channels |
|---|---|---|
| 1 | 32×32 | 16 |
| 2 | 16×16 | 32 |
| 3 | 8×8 | 64 |

A deliberate trade: **give up spatial resolution, buy feature variety.** Early
layers know precisely *where* things are but little about *what* they are; late
layers know a great deal about *what* is present and have nearly discarded
*where*.

The compute stays balanced. Halving each spatial dimension quarters the pixel
count while doubling the channels only doubles the work per pixel, so the
tensor shrinks by 2× per stage boundary and the cost per layer stays in the
same range.

The downsampling is done by **stride 2 on the first convolution of a stage** —
the CIFAR ResNet has no pooling layers between stages.

---

## What happens to the shortcut when a block changes shape?

Something has to reshape it, and the paper tests three ways of doing so.

Most blocks preserve shape, so the shortcut is a plain identity — the input
passes through untouched. But at each stage boundary a block halves the
resolution and doubles the channels. The input is `(N, 16, 32, 32)` while the
convolution output is `(N, 32, 16, 16)`, and addition requires identical
shapes.

The three answers:

- **Option A** — subsample spatially and pad the new channels with zeros. No
  parameters; the shortcut remains a pure identity mapping. Used for every
  CIFAR experiment in the paper.
- **Option B** — a 1×1 convolution, but only on blocks where the shape changes.
- **Option C** — a 1×1 convolution on every shortcut.

Reported ImageNet top-1 error: 25.03%, 24.52%, and 24.19% respectively. All
three beat the plain network by a wide margin, and the gaps between them are
small — which is itself evidence for the paper's claim. If elaborate
projections mattered, the differences would be large. They are not, so what is
doing the work is the identity, not the machinery patching up shapes.

In ResNet-20, seven of nine blocks need no reshaping at all.

---

## What makes this a controlled experiment rather than an implementation?

The addition has **no parameters**.

Because the shortcut costs nothing, a residual network and a plain network of
the same depth have identical parameter counts, identical layer counts,
identical FLOPs, and — with the same seed — identical initial weights. The only
difference in the entire system is one `+`.

So when plain-56 does worse than plain-20 on *training* error while ResNet-56
does better, exactly one explanation is available. No confound about model
capacity.

This is why the parameter-count assertion in `tests/test_block.py` matters more
than any other check: if `residual=True` and `residual=False` report different
counts, something is carrying weights that should not be, and every downstream
number is contaminated.

---

Back to: [Q&A index](README.md)
