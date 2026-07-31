# Foundations

Background the rest of the documentation assumes.

**Assumed background:** basic statistics — mean, variance, distribution,
sampling. Nothing else, and no machine learning.

Read a page here when a term shows up elsewhere that you haven't met. The other
folders link back to these rather than re-explaining.

---

## Contents

### [training-vocabulary.md](training-vocabulary.md)

Loss, cross-entropy, gradients, learning rate and schedules, warmup, SGD,
batches, momentum, iterations vs epochs, train/validation/test splits,
overfitting vs the degradation problem, regularization, weight decay,
augmentation, dropout, error rate, initialization, and seeds.

Includes the distinction the entire project depends on: **overfitting is high
*test* error with low *training* error; the degradation problem is high
*training* error.**

---

## Room for later

- `convolutional-networks.md` — why convolutions rather than fully connected
  layers, and how the field arrived here. Some of this is already in
  [`../block/01-convolution.md`](../block/01-convolution.md).
- `reading-a-paper.md` — how to work through the PDF in `resources/paper/`:
  which sections matter, how to read a results table, and how to tell a claim
  from a hypothesis.
- `experimental-method.md` — controls, ablations, seeds and variance, and what
  makes a reproduction honest. Relevant to the write-up.

---

## Where the other folders sit

| folder | covers |
|---|---|
| `docs/foundations/` | the vocabulary and concepts — start here when lost |
| [`docs/block/`](../block/README.md) | the architecture: what a residual block is and why |
| [`docs/library/`](../library/README.md) | the machinery: what PyTorch does underneath the API |
