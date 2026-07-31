# Library Internals

How the code this project depends on actually works, underneath the API.

`docs/block/` explains the **architecture** — what a residual block is and why.
This folder explains the **machinery** — what `nn.Conv2d` is doing when you
call it, where the speed comes from, and what happens between your Python line
and the arithmetic on the chip.

Nothing here is required to finish the project. It's here because the layer
below the API is interesting and because knowing it makes you much better at
debugging performance and numerical problems.

---

## Layout

One subfolder per library, so this can grow without reorganising.

```
docs/library/
  README.md          ← you are here
  pytorch/           ← PyTorch and the native stack under it
```

### [`pytorch/`](pytorch/README.md)

The framework, the tensor implementation, the dispatcher, autograd, and the
BLAS and vendor kernel libraries that do the actual arithmetic. Also
per-layer deep dives on the four `nn` modules used in the basic block.

### Room for later

Folders to add as the project reaches them:

- `torchvision/` — datasets, transforms, and the reference ResNet
  implementation. Relevant to the data pipeline.
- `numpy/` — the array model PyTorch's tensor design descends from, and the
  strides/views vocabulary they share.
- `matplotlib/` — figure and axes model. Relevant to the analysis and plots.
- `yaml/` — the parser behind the config loader. Small, and the safe-load
  question is worth understanding.

---

## A note on depth

These documents go one or two layers below the API and then stop. That's
deliberate — the goal is a correct mental model of what a call costs and what
it's doing, not a source tour of a two-million-line C++ codebase.

Where a claim can be checked, it has been checked on this machine rather than
recited. Where something is version- or platform-specific, that's called out,
because much of this changes between PyTorch releases and between an Apple
Silicon Mac and a Colab T4 — which matters directly, since you'll run on both.
