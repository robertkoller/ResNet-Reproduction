# The Basic Block — Start Here

This folder explains, from the ground up, the one component that the entire
ResNet reproduction is built on: the **basic residual block**.

It assumes no prior deep learning knowledge and no PyTorch experience. It does
assume you can read Python.

---

## Why this one component gets seven documents

The CIFAR ResNet is almost nothing but copies of this block.

ResNet-20 is nine of them. ResNet-110 is fifty-four. Around them sits a single
conv at the input, a pooling operation, and a classification layer — perhaps
thirty lines of plumbing in total. **Understand the block and you understand
roughly 95% of the architecture.**

It's also where the paper's actual idea lives. The shortcut connection is one
`+` sign inside this block, and that `+` is the entire contribution of a paper
with tens of thousands of citations. Everything else — three stages, six
depths, a learning rate schedule — is scaffolding built to test whether that
one addition matters.

So it's worth taking slowly.

---

## Reading order

The documents build on each other. Read them in sequence the first time.

### [00 — Tensors, Channels, and Shapes](00-tensors-and-shapes.md)

What a tensor is. Why a 2D image has the shape `3 × 32 × 32`. What a channel
actually is, and why there are 16 of them after the first layer when there were
only 3 colours going in. The batch dimension. How the shape changes as data
moves through the network.

**Read this first.** Most errors encountered while building the model are
shape errors, and shape errors are only confusing while this material is
fuzzy.

### [01 — Convolution](01-convolution.md)

The pattern-finding operation, worked through by hand on a small example.
Kernels, weight sharing, multiple input and output channels, padding, stride,
receptive fields, and the parameter-count formula used to verify the
architecture against the paper's table.

### [02 — Batch Normalization](02-batch-normalization.md)

What it normalizes and across which dimensions. The learned scale and shift.
Running statistics, and the train/eval distinction — which is the source of the
most common serious bug in beginner PyTorch code.

### [03 — ReLU and Nonlinearity](03-relu-and-nonlinearity.md)

Why a network with no nonlinearity has the power of a single layer regardless
of depth. Why ReLU beat sigmoid. Where the two ReLUs sit in the block and why
the second one comes *after* the addition.

### [04 — The Shortcut and the Addition](04-shortcut-and-addition.md)

The paper's actual idea. Why learning a residual is easier than learning a full
mapping. Why the shortcut having no parameters is what makes the reproduction
a controlled experiment. The shape-mismatch problem and the paper's three
answers.

### [05 — Putting the Block Together](05-putting-it-together.md)

The assembled block. Two shape traces — an ordinary block and a downsampling
one — plus full parameter accounting up to ResNet-20's 269,722 parameters, and
a reading of the structural decisions in `models/blocks.py`.

### [06 — PyTorch Mechanics](06-pytorch-mechanics.md)

The framework machinery: `nn.Module`, why `super().__init__()` must come first,
how parameter registration works and where it silently fails, why the module is
called rather than `forward`, parameters vs. buffers, devices, `train()` and
`eval()`.

### [07 — Glossary](07-glossary.md)

Every term, alphabetically, with a pointer back to the document that covers it.
Use it as a reference once you've read the rest.

---

## How it all ties together

Here's the chain of reasoning the seven documents build, in one pass:

**An image is a stack of channels** — three grids of 32×32 numbers, one per
colour. Every operation in the network transforms one stack of channels into
another. [00]

**A convolution is how you transform them.** It slides learned 3×3 kernels
across the input, and each kernel produces one output channel. Sixteen kernels
give sixteen output channels, each detecting some pattern the network chose to
care about. The same weights are reused at every position, which is why convs
are cheap and why a detector learned in one corner works everywhere. [01]

**But stacked convolutions collapse.** Convolution is linear, and linear
functions compose into linear functions, so twenty convs in a row can be
replaced by one. Depth would buy nothing. **ReLU** breaks that by bending the
function at zero, and only then does stacking layers mean anything. [03]

**And deep stacks drift.** Each layer changes the scale of what the next one
sees, and that scale compounds and shifts as training progresses. **BatchNorm**
pins it: normalize each channel to mean 0 variance 1, then let the network
learn its own scale and shift if it wants one. This is what makes learning rate
0.1 survivable and why the paper needs no dropout. [02]

**So a block is: conv → BN → ReLU → conv → BN.** Two pattern-finding stages,
each stabilised, with a nonlinearity between them so they don't collapse into
one. [05]

**Now the paper's problem.** Stack enough of those and it gets *worse* — not
worse on new photos, worse on the photos it trained on. That shouldn't be
possible, since a deeper network could always copy a shallower one and set the
extra layers to do nothing. A good solution provably exists; the optimizer just
can't find it.

**The fix is one addition.** Run a wire from the block's input to its output
and add. Now the convs only need to learn the *difference* from the input. If
"do nothing" is the right answer, they output zero instead of having to
reconstruct their input exactly through two nonlinear layers. Easy instead of
hard. [04]

**And the addition has no weights** — which is what makes this an experiment
rather than a demonstration. Residual and plain networks have identical parameter counts,
identical layer counts, and with the same seed identical initial weights. The
only difference in the entire system is one `+`. So when one gets better with
depth and the other gets worse, there's exactly one explanation available. [04]

**Two details make it work in practice.** The ReLU goes *after* the addition,
which means the block computes an exact identity when the convs output zero,
since the incoming `x` is already non-negative. [03] And when a block changes
shape, something has to reshape the shortcut — the paper tests three options
and finds the choice barely matters, which supports its claim that the identity
is what's doing the work. [04]

**PyTorch supplies the plumbing.** `nn.Module` tracks the layers so they move
to the GPU, reach the optimizer, and land in checkpoints — provided
`super().__init__()` is called and no module is hidden inside a plain list. [06]

---

## Where the code lives

| component | file |
|---|---|
| basic block, shortcuts | `models/blocks.py` |
| CIFAR and ImageNet-style networks | `models/resnet.py` |
| correctness checks | `tests/test_block.py` |
| convolution implemented by hand | `tests/test_conv_from_scratch.py` |

The single most important property those tests assert: **`residual=True` and
`residual=False` produce identical parameter counts.** The shortcut is
parameter-free, so if the counts differ, something is carrying weights that
should not be, and the plain-versus-residual comparison is confounded.

---

## Related documents

- [`../foundations/training-vocabulary.md`](../foundations/training-vocabulary.md)
  — loss, learning rate, epochs, overfitting, dropout, weight decay. **Read
  this first if any term here is unfamiliar.** Assumes only basic statistics.
- [`../library/README.md`](../library/README.md) — the machinery underneath:
  what PyTorch does when you call `nn.Conv2d`, and where the speed comes from.
- `resources/notes/resnet-algorithm-summary (1).md` — the paper in plain
  English.
- `resources/paper/` — the original PDF. Sections 3.1–3.4 and 4.2 are the ones
  this folder unpacks.
