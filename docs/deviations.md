# Deviations from the Original Protocol

Every place this reproduction differs from He et al., *Deep Residual Learning
for Image Recognition* (CVPR 2016), with the reasoning and the expected effect.

A reproduction that reports no deviations is either trivial or not being
honest. Hardware, framework, and library defaults have all moved since 2016,
and some of the paper's choices are underspecified. What follows is the
complete list, maintained alongside the work rather than reconstructed
afterwards.

---

## Environment

| | |
|---|---|
| hardware | Apple M4, GPU via Metal Performance Shaders |
| platform | macOS 26.4, arm64 |
| Python | 3.13.9 |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |

The paper's experiments predate all of this. Its CIFAR results come from Caffe
on NVIDIA hardware, so the numerical path from input to loss differs at the
kernel level even where the architecture and recipe match exactly.

---

## 1. Per-channel rather than per-pixel normalization

**The paper** subtracts the per-pixel mean — a full `3 × 32 × 32` mean image,
one value for every channel at every spatial position.

**This repository** subtracts a per-channel mean and divides by a per-channel
standard deviation: three numbers each, applied uniformly across all positions.
See `data/cifar10.py`.

**Why.** Per-channel normalization is what `torchvision.transforms.Normalize`
implements and what essentially every published reproduction uses. Per-pixel
would require a custom transform and a stored mean tensor.

**Expected effect.** Small. Both centre the data; per-pixel additionally
removes position-specific bias, which on natural images is weak. No measurable
difference in final error is expected, though this has not been tested here.

---

## 2. Normalization statistics computed from 45,000 images

**The paper** does not state which images its statistics come from.

**This repository** computes them from the 45,000-image training split only,
excluding both the 5,000 held-out validation images and the test set. The
measured values are:

```
mean  (0.4915, 0.4821, 0.4464)
std   (0.2469, 0.2435, 0.2614)
```

**Why.** Statistics drawn from data the model is meant not to have seen are a
small but real leak. The commonly quoted CIFAR-10 figures — mean
`(0.4914, 0.4822, 0.4465)` — are computed over all 50,000 training images and
differ from these in the fourth decimal place.

**Expected effect.** Negligible numerically. Recorded because it explains why
these constants do not match the ones found in most other repositories.

---

## 3. Bottleneck stride placement (ResNet v1.5)

**The paper** places the stride-2 downsample on the first `1 × 1` convolution
of a bottleneck block.

**This repository** places it on the `3 × 3` convolution instead, matching
torchvision's convention. See the comment in `models/blocks.py`.

**Why.** It is the modern default, and it allows the parameter counts to be
checked against torchvision as an independent reference. All five ImageNet
depths match exactly: 11,689,512 / 21,797,672 / 25,557,032 / 44,549,160 /
60,192,808.

**Expected effect.** Parameter counts are identical either way. FLOPs are not:
the first `1 × 1` then runs at full resolution, costing roughly 7% more.
Measured on a single 256→128 stride-2 block, 192.7M against 269.7M FLOPs for
identical parameters. This is why the measured whole-network FLOPs exceed the
paper's Table 1 figures at depths 50, 101 and 152:

| depth | measured | paper |
|---|---|---|
| 18 | 1.81G | 1.8G |
| 34 | 3.66G | 3.6G |
| 50 | 4.09G | 3.8G |
| 101 | 7.80G | 7.6G |
| 152 | 11.51G | 11.3G |

Depths 18 and 34 use basic blocks and are unaffected.

This only touches the ImageNet-style architecture. The CIFAR networks that
carry the main result use basic blocks throughout.

---

## 4. Reproducibility is partial

**The paper** does not discuss seeding.

**This repository** seeds Python, NumPy and PyTorch from the experiment's
configuration, and seeds DataLoader workers separately since they run in their
own processes. See `train/utils.py` and `data/cifar10.py`.

**What this guarantees.** Identical weight initialization, identical batch
ordering, and an identical train/validation split across runs and machines.

**What it does not guarantee.** Bitwise-identical training. Some Metal kernels
are non-deterministic, so two runs at the same seed diverge during training
even from identical starting weights. Results are also not reproducible across
platforms: a CUDA build sums floating-point values in a different order from a
Metal one.

**Consequence.** Runs from different devices are not mixed within a single
comparison. Where seed-to-seed variance is reported, it is measured by varying
the seed rather than assumed.

---

## 5. The train/validation split is fixed across all experiments

**The paper** uses a 45k/5k train/validation split for its CIFAR ablations,
which this repository matches.

**This repository** additionally fixes the split with a dedicated seed that is
independent of the experiment seed, so every run is scored against the same
5,000 held-out images. The indices are written to disk.

**Why.** Tying the split to the experiment seed would give each of the five
ResNet-110 runs a different validation set, confounding seed variance with
split variance and making depths incomparable.

---

## 6. Partial final batch dropped during training

**The paper** does not specify.

**This repository** drops the last incomplete batch of each training epoch:
45,000 images at batch size 128 leaves 72 images, or **0.16%** of the epoch.

**Why.** A 72-image batch produces noisier batch-normalization statistics than
a 128-image one. The dropped images differ every epoch because the loader
reshuffles, so no image is systematically excluded.

**Expected effect.** Negligible. Evaluation uses every image.

---

## 7. He initialization uses fan-out

**The paper** derives its initialization for both forward and backward
propagation and notes that either choice is workable.

**This repository** uses `fan_out`, matching torchvision. See
`initialize_weights` in `models/resnet.py`.

**Expected effect.** Minor. The two differ by a constant factor per layer that
batch normalization largely absorbs.

---

## What is *not* a deviation

Worth stating explicitly, because these look surprising in isolation:

**Augmented training batches have a mean near −0.27, not 0.** `RandomCrop` with
4 pixels of padding pads with black, which normalizes to about −1.99. Around
13.4% of a random crop is padding, predicting a shift of −0.254 against the
observed −0.269. Unaugmented evaluation data normalizes to a mean of −0.0016
and a standard deviation of 1.0081, as expected. This is what the paper's
augmentation recipe produces.

**Shortcut option A is the default.** This matches the paper, which uses the
parameter-free zero-padded identity for every CIFAR experiment.

**Evaluation uses a larger batch size than training.** This changes nothing:
the model is in eval mode, so batch normalization uses its running statistics
and results do not depend on batch composition.

---

## Related

- [`block/README.md`](block/README.md) — the architecture
- [`foundations/training-vocabulary.md`](foundations/training-vocabulary.md) —
  the terms used above
- `resources/paper/` — the original paper
