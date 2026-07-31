# Documentation

Written alongside the reproduction, as a record of what had to be understood to
build it. Four folders: three levels of abstraction, plus a record of the
questions that came up along the way.

| folder | answers | |
|---|---|---|
| [`foundations/`](foundations/README.md) | *what does this term mean?* | assumes basic statistics, no machine learning |
| [`block/`](block/README.md) | *what is the architecture, and why?* | the residual block from the ground up |
| [`library/`](library/README.md) | *what is PyTorch doing underneath?* | the machinery below the API |
| [`qanda/`](qanda/README.md) | *why is it designed this way?* | design questions, with answers |

Read in whatever order the confusion demands. Each folder has its own index.

---

## [`foundations/`](foundations/README.md)

Background everything else assumes.

- [`training-vocabulary.md`](foundations/training-vocabulary.md) — loss,
  gradients, learning rate and schedules, warmup, SGD, batches, momentum,
  iterations vs epochs, train/validation/test, overfitting, regularization,
  weight decay, augmentation, dropout, error rate, initialization, seeds.

Includes the distinction the whole reproduction rests on: **overfitting is high
*test* error with low *training* error; the degradation problem is high
*training* error.**

## [`block/`](block/README.md)

The basic residual block, in nine documents. The architecture is almost nothing
but copies of this one component, so it carries most of the explanation:
tensors and channels → convolution → batch normalization → ReLU → the shortcut
→ assembly → PyTorch mechanics → glossary.

[`01-convolution.md`](block/01-convolution.md) closes with a linear algebra
note, for readers who notice that convolution looks like a linear map. It is
one, and the note makes that precise — Toeplitz structure, inner products, and
the im2col identity that turns a convolution into a single matrix multiply.

## [`library/`](library/README.md)

How the dependencies work beneath their APIs. One subfolder per library.

- [`pytorch/`](library/pytorch/README.md) — the stack from Python to silicon,
  BLAS and GEMM, `nn.Module`'s registration machinery, and the layers in
  detail.

## [`qanda/`](qanda/README.md)

Questions about how the system is designed and why, in two files: `concepts.md`
for the ideas the design rests on, `architecture.md` for the structure and the
reasoning behind it.

Deliberately narrow — setup, syntax, and implementation mechanics are not
recorded. Answers are summaries that link back into the folders above.

---

## Conventions

These documents explain; they do not instruct. Implementation lives in the
source tree and is referenced by path rather than reproduced here — code blocks
appear only for API signatures, runnable demonstrations, verification snippets,
and diagrams.

Figures were measured rather than recited: parameter counts against PyTorch,
shape arithmetic against real tensors, throughput against a benchmark, and the
im2col equivalence against `nn.Conv2d`. Hardware-specific numbers were taken on
an Apple M4 and are labelled as such.

Where the paper hypothesises rather than demonstrates — as it does about *why*
residual learning helps — the documents say so.

---

## Elsewhere in the repository

- `resources/paper/` — the original CVPR 2016 paper.
- `resources/notes/` — the paper in plain English, and the project plan.
