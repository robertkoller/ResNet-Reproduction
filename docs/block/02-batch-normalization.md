# 02 — Batch Normalization

The layer that makes training deep networks at learning rate 0.1 possible
instead of hopeless. Also the layer with the nastiest bug in beginner PyTorch.

Prerequisite: [00 — Tensors and Shapes](00-tensors-and-shapes.md).

---

## First: what a learning rate is

This page keeps saying "learning rate 0.1," so here's what that means. If
you've met gradient descent in a statistics course this will be familiar; if
not, start here.

**The setup.** The network has parameters — 269,722 of them in ResNet-20.
Feed it an image, it guesses, and you compute the **loss**: one number
measuring how wrong the guess was. Zero means perfect.

The loss depends on every parameter, so it defines a surface over a
269,722-dimensional space. Training means finding a low point on that surface.

**The gradient** is the slope. For each parameter it answers: *if I increase
this one slightly, does the loss go up or down, and how fast?* It's the vector
of partial derivatives of the loss with respect to the parameters — the
direction of steepest increase.

**The learning rate is the step size.** You move each parameter a little way
in the downhill direction, and the learning rate says how far:

```
new_value = old_value − learning_rate × gradient
```

That's the whole training algorithm. Everything else is refinement.

The usual picture: you're on a hillside in fog, wanting the valley floor. You
can't see it, but you can feel which way the ground slopes. The gradient is
that feel of slope; the learning rate is how big a stride you take before
stopping to feel again.

**Why the value matters so much:**

| learning rate | what happens |
|---|---|
| far too large | you leap past the valley and land higher up the far side; loss oscillates or becomes `NaN` |
| slightly too large | progress, but jittery — never settles |
| about right | steady descent |
| too small | correct direction, but you'd need a million steps |

There's no formula for the right value. It depends on the architecture, the
data, the batch size, and how the weights were initialized. It's a
**hyperparameter** — a number you choose before training rather than one the
network learns.

**Your value is 0.1**, taken from the paper. That's large. Many networks need
0.001 or smaller, and this is exactly where batch normalization comes in.

**How it relates to this page.** Without BatchNorm, the scale of the numbers
flowing through the network drifts unpredictably from layer to layer (that's
the next section). When scales differ wildly between layers, the same step size
is simultaneously too big for one layer and too small for another. You're
forced to pick a learning rate small enough for the most sensitive layer, which
means everything else crawls.

BatchNorm pins the scale at every layer to roughly the same range. Now one step
size is reasonable everywhere, and 0.1 becomes usable instead of instantly
divergent. **That is the practical reason BatchNorm matters**, and it's why the
paper's recipe can specify a single learning rate for a 110-layer network at
all.

One more piece of the configuration: the learning rate isn't constant. It starts at
0.1, drops to 0.01 at iteration 32,000 and 0.001 at 48,000. Big strides early
to find the right region, small ones later to settle into it. See
[`docs/foundations/training-vocabulary.md`](../foundations/training-vocabulary.md)
for the rest of the training vocabulary — epochs, overfitting, dropout, weight
decay, momentum.

---

## The problem it solves

Numbers flowing through a deep network drift in scale.

Layer 1 outputs values around ±1. Layer 2's weights multiply those, and the
output happens to average ±1.4. Layer 3 makes it ±2. By layer 20 you might be
at ±10,000 or at ±0.0001, depending on which way the drift went. Both are
disasters: huge values make gradients explode, tiny values make them vanish.

Worse, the drift **changes during training**. Every weight update shifts the
distribution that the next layer sees. Layer 15 spends its time chasing a
moving target — it learns to handle inputs of one scale, then the layers below
it change and it has to relearn. The original paper called this **internal
covariate shift**.

The fix is blunt: after every conv, force the numbers back to a standard scale.

---

## What it actually computes

For each channel independently, BatchNorm does four things:

**1. Compute the mean** of every value in that channel, across the whole batch
and all pixel positions.

For a tensor of shape `(128, 16, 32, 32)`, channel 5's mean is taken over
`128 × 32 × 32 = 131,072` numbers. You get 16 means, one per channel.

**2. Compute the variance** the same way. 16 variances.

**3. Normalize:**

```
x_normalized = (x − mean) / sqrt(variance + epsilon)
```

Every channel now has mean 0 and variance 1. The `epsilon` (default `1e-5`) is
there so you never divide by zero when a channel is constant.

**4. Scale and shift with learned parameters:**

```
output = gamma × x_normalized + beta
```

`gamma` and `beta` are learned, one pair per channel.

That fourth step looks like it undoes the first three, and it can — but only if
the network decides that's useful. The point is that the scale becomes
something the network **chooses deliberately** rather than something that
accumulates by accident. It starts at `gamma=1, beta=0` and moves only if
gradients say it should.

---

## Which numbers get averaged together — the part people get wrong

BatchNorm2d normalizes **per channel**, pooling across batch, height, and
width.

```
input (128, 16, 32, 32)

channel 0:  ████████████████  ← all 128×32×32 values, one mean, one variance
channel 1:  ████████████████  ← separately
channel 2:  ████████████████  ← separately
...
channel 15: ████████████████
```

Not per image. Not per pixel. Per **channel**.

This is deliberate. A channel is one feature detector, and it should have a
consistent output scale wherever and whenever it fires. Two different channels
are different features and have no reason to share a scale.

Note the consequence: **an image's normalized value depends on the other images
in its batch.** That's unusual — every other layer treats images independently
— and it drives everything odd about BatchNorm.

---

## Train mode vs eval mode

Here is where the bug lives.

**During training**, BatchNorm uses the current batch's mean and variance. Fine
— there are 128 images, the statistics are reasonable.

**At test time** you might evaluate one image at a time. "The mean of this batch
of 1" is just the image's own value, so normalizing by it would output zero
regardless of input. Useless. And even with a full test batch, your predictions
would change depending on which other images happened to be alongside — a
network whose answer depends on its neighbours is not a classifier.

So BatchNorm keeps a **running average** of the statistics it saw during
training, and uses those frozen numbers at test time:

```
running_mean = (1 − momentum) × running_mean + momentum × batch_mean
```

PyTorch's default `momentum` is `0.1`. (Confusingly, this is unrelated to SGD's
momentum of 0.9 in `configs/`. Same word, different mechanism.)

These running statistics are **buffers**, not parameters. Gradient descent
doesn't touch them; they're updated by observation during the forward pass.
Inspect them yourself:

```python
from torch import nn
bn = nn.BatchNorm2d(16)
print([name for name, _ in bn.named_buffers()])
# ['running_mean', 'running_var', 'num_batches_tracked']
```

**You switch behaviours with `model.train()` and `model.eval()`.**

This is the single most common serious bug in beginner PyTorch. Forget
`model.eval()` before testing and:

- your test numbers use batch statistics instead of running ones,
- the running statistics keep updating on test data, which is a subtle form of
  leaking test information into your model,
- and none of it raises an error.

Training curves look perfectly healthy. Test error is just quietly wrong. You
will not find this by staring at the code — you find it by knowing it exists.

Build the habit now: `model.train()` at the top of the training loop,
`model.eval()` before any evaluation, and wrap evaluation in
`torch.no_grad()`. Every time, without thinking about it.

---

## Parameters and buffers

For `nn.BatchNorm2d(16)`:

| what | count | learned? |
|---|---|---|
| `weight` (gamma) | 16 | yes |
| `bias` (beta) | 16 | yes |
| `running_mean` | 16 | no — a buffer |
| `running_var` | 16 | no — a buffer |
| `num_batches_tracked` | 1 | no — a counter |

So **`2 × num_channels` parameters**: 32 for a 16-channel layer. Verified:
`sum(p.numel() for p in nn.BatchNorm2d(16).parameters())` returns `32`.

Buffers don't count as parameters but they *are* saved in checkpoints — they
must be, or a reloaded model would evaluate differently. PyTorch's
`state_dict()` includes them automatically.

---

## Why this lets you use learning rate 0.1

The CIFAR-10 recipe has `learning_rate: 0.1`, which is enormous compared to what you'd
use without BatchNorm.

The reason is that BatchNorm makes the loss surface better **conditioned**.
Without it, some directions in weight space have huge gradients and others tiny
ones, so the learning rate must be small enough for the steepest direction —
which means glacial progress in every other direction. Normalizing activations
puts the directions on a comparable footing.

There's also a scale-invariance effect: if you double a conv's weights,
BatchNorm divides the output by the now-doubled standard deviation and the
result is unchanged. So the layer's *magnitude* stops mattering and only its
*direction* does. That removes an entire failure mode.

---

## Why there's no dropout in this paper

### What dropout is

**Dropout** is a regularization technique: during each training step, you
randomly pick some fraction of the units in a layer — typically half — and set
their outputs to zero. Different random units each step. At test time nothing
is dropped and the outputs are rescaled to compensate.

It sounds like vandalism. The reasoning:

A network with plenty of capacity can solve the training set by memorising it,
building fragile chains where unit 47 only works because unit 12 always fires
alongside it. That's **co-adaptation**, and it produces a model that scores
well on photos it has seen and badly on new ones — **overfitting**.

Dropout breaks it. If any unit might vanish this step, no unit can depend on
another being present. Each has to carry useful information on its own. The
statistical analogy is bagging: you're effectively training an enormous
ensemble of thinned networks that share weights, then averaging them at test
time.

Dropout was close to universal in the years before this paper, especially in
the big fully connected layers at the end of networks like AlexNet and VGG.

### Why ResNet doesn't need it

The CIFAR-10 recipe has no dropout setting, and that's deliberate — the paper
explicitly doesn't use it. Three reasons:

**BatchNorm already injects noise.** Each image's normalization depends on
which other images happen to share its batch, so the same image produces
slightly different activations from one batch to the next. That's randomness
during training, doing the same job dropout does, as a side effect of
normalizing.

**There are no giant fully connected layers to protect.** Dropout's biggest
wins came in dense layers with tens of millions of parameters. ResNet ends with
global average pooling into a 640-parameter classifier. There's very little
there to overfit.

**Weight decay handles the rest.** `weight_decay: 0.0001` in `configs/` pulls
every weight toward zero on each step, discouraging reliance on a few enormous
weights. See
[`docs/foundations/training-vocabulary.md`](../foundations/training-vocabulary.md).

Dropout and BatchNorm also interact badly — the two noise sources fight, and
dropout's train/test rescaling disturbs the statistics BatchNorm is tracking.
Combining them often underperforms either alone. After this paper, dropout
largely disappeared from convolutional architectures.

**Where this shows up in the results:** the 1202-layer network in the paper
trains to under 0.1% training error but tests *worse* than the 110-layer one.
That's overfitting — too much capacity for a 50,000-image dataset — and it's a
different failure mode entirely from the degradation problem. Worth keeping
straight, because the write-up needs to distinguish them.

---

## The signature

```python
nn.BatchNorm2d(
    num_features,      # number of channels — the only argument you need
    eps=1e-5,          # stability constant in the denominator
    momentum=0.1,      # running-statistics update rate
    affine=True,       # whether to learn gamma and beta
)
```

`num_features` must equal the `out_channels` of the conv feeding it. It
normalizes what the conv *produced*, so it's the output count, never the input
count. Getting this wrong raises a shape error immediately — one of the
friendlier mistakes available.

In the block:

- `nn.BatchNorm2d(out_channels)` after the first conv
- `nn.BatchNorm2d(out_channels)` after the second conv

Docs: `https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html`

---

## Terms from this page

- **Normalize** — rescale to mean 0, variance 1.
- **Internal covariate shift** — the drifting of layer input distributions
  during training.
- **gamma / beta** — BatchNorm's learned scale and shift, called `weight` and
  `bias` in PyTorch.
- **Running statistics** — the training-time averages used at test time.
- **Buffer** — persistent state saved with the model but not trained by
  gradient descent.
- **Train mode / eval mode** — `model.train()` and `model.eval()`, which
  switch BatchNorm between batch and running statistics.
- **Epsilon** — small constant preventing division by zero.

---

Previous: [01 — Convolution](01-convolution.md) ·
Next: [03 — ReLU and Nonlinearity](03-relu-and-nonlinearity.md)
