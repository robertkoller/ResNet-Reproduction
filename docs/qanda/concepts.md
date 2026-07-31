# Q&A — Concepts

The ideas the design rests on.

Fuller treatment in
[`../foundations/training-vocabulary.md`](../foundations/training-vocabulary.md).

---

## What is a gradient?

A **slope**. For each parameter it answers: *if this number is increased
slightly, does the loss go up or down, and how fast?*

- **Sign** tells you which direction reduces the loss.
- **Magnitude** tells you how much that parameter is currently affecting the
  outcome.

The full gradient is one such answer per parameter — 269,722 numbers for
ResNet-20, the same shape as the weights themselves. It is computed by
**backpropagation**, which applies the chain rule backward through the network
and produces every derivative for roughly the cost of one forward pass.
`loss.backward()` does this; it is never written by hand.

The "stochastic" in SGD is that the gradient is computed on a random batch of
128 images rather than all 45,000 — a noisy estimate, right on average, wrong
in detail. That noise is why momentum exists, and it partly helps, shaking the
optimizer out of poor spots.

One practical trap: PyTorch **accumulates** gradients rather than overwriting
them, so `optimizer.zero_grad()` is required each iteration. Forgetting it does
not crash; training just quietly does worse.

---

## What is a learning rate? What does "step size" mean?

"Step size" is a misleading metaphor, because nothing moves.

A parameter is a number in memory. Training **replaces that number with a
slightly different number**, 64,000 times. A "step" is one of those
replacements, and "step size" is how much the number changed.

```
new_value = old_value − learning_rate × gradient
```

Concretely: a weight holding `4.0` with a gradient of `8.0`, at
`learning_rate = 0.1`, becomes `4.0 − 0.8 = 3.2`. It changed by 0.8 — that 0.8
is the step.

**The learning rate is not the step size.** The step is
`learning_rate × gradient`. The learning rate is the *multiplier*. Calling the
rate "step size" is the sloppiness that causes this exact confusion.

The full worked example — one parameter, loss `w²`, six learning rates from
convergence to divergence — is in
[`../foundations/training-vocabulary.md`](../foundations/training-vocabulary.md),
section 4. The `lr=1.1` row runs away to infinity, which is what "the loss
diverged" means and why `warmup: true` exists at depth 110.

**Why one number matters so much:** all 269,722 parameters share a single
learning rate. One multiplier has to be simultaneously reasonable for every
parameter in the network. That is precisely what batch normalization makes
possible.

---

## What do the training steps actually do? Is it learning the kernels?

Yes — more literally than it might sound. Breaking down ResNet-20's parameters:

| what | count | share |
|---|---|---|
| convolution kernel numbers | 267,696 | **99.2%** |
| BatchNorm scale and shift | 1,376 | 0.5% |
| final classifier | 650 | 0.2% |

The network contains **29,744 individual 3×3 kernels**. Training is the process
of choosing those numbers. The architecture — depth, channel counts, where the
shortcuts go — is fixed by configuration. The kernel *contents* are what
gradient descent searches for.

One iteration: 128 images forward, one loss number, backpropagation producing
269,722 gradients, then the one-line update applied to every one. Any single
step barely changes anything. Sixty-four thousand of them accumulate into
kernels that detect edges, then textures, then object parts.

A demonstration of this happening — a single 3×3 kernel starting from noise and
becoming a recognisable vertical-structure detector over 400 steps — is
described in
[`../block/01-convolution.md`](../block/01-convolution.md).

**What is being optimized is the loss, not the kernels directly.** The kernels
are the free variables; the loss is the objective. And the loss is a
*surrogate*: the quantity actually cared about is error rate, but error rate is
a step function whose gradient is zero almost everywhere and therefore carries
no direction information. Cross-entropy is smooth and differentiable
everywhere. Optimize the loss, report the error.

**How convergence shows up:** gradients going quiet. A gradient near zero means
no nudge in either direction would improve the loss.

**The catch:** "best kernels" means best *on the training data*, since that is
the only data the loss sees. This is why validation and test sets exist, and
why the paper's 1202-layer network — training error under 0.1%, worse test
error than the 110-layer model — is a cautionary result rather than a triumph.

---

## What do the hyperparameters mean?

All values come from the paper's CIFAR-10 recipe; none were tuned here.

| setting | value | what it does |
|---|---|---|
| `learning_rate` | 0.1 | step multiplier along the gradient |
| `momentum` | 0.9 | running average of recent gradients; roughly a ten-step memory |
| `weight_decay` | 0.0001 | pulls every weight toward zero each step (L2 regularization) |
| `batch_size` | 128 | images averaged into one gradient estimate |
| `max_iterations` | 64000 | when to stop |
| `learning_rate_milestones` | [32000, 48000] | where the rate drops |
| `learning_rate_gamma` | 0.1 | multiplier at each drop: 0.1 → 0.01 → 0.001 |
| `warmup` | depth 110 only | start at 0.01 until training error is below 80% |
| `evaluate_every` | 500 | logging cadence |
| `checkpoint_every` | 2000 | how often weights and optimizer state are saved |

Two things worth internalising:

**64,000 iterations × 128 images = 8.19 million images seen ≈ 182 epochs** over
a 45,000-image training set. The learning-rate drops land near epoch 91 and
epoch 136.

**The two drops appear as sharp cliffs in the training curves** at exactly
those iterations. That is the most recognisable signature of a correctly wired
schedule.

Note there is no dropout setting. The paper deliberately omits it — batch
normalization's batch-dependent noise regularizes on its own.

---

## Why does randomness have to be controlled in this experiment?

Because the effects being measured are the same size as the noise.

**Weight initialization.** The seed determines the entire starting point of
training. There is a property specific to this experiment: plain and residual
networks have identical parameter counts and identical module structure,
because the option A shortcut is parameter-free. Seeded identically, they start
from *literally the same weights* and diverge purely because of the shortcut.
(This holds only for option A; B and C add parameters and shift every
subsequent draw.)

**Defensible reported numbers.** The targets at depths 44 and 56 are 7.17% and
6.97% — a gap of 0.2 percentage points, comparable in size to run-to-run
variation from seeds alone. A single run of each cannot distinguish a real
effect from a coin flip. This is why the paper reports five runs for
ResNet-110.

**A reproducible data pipeline.** The 45k/5k split, the shuffling order, and
the augmentation crops are all seeded. Comparing two depths on *different*
validation sets would quietly corrupt the comparison.

**What it does not buy:** bitwise reproducibility. Some Metal kernels are
non-deterministic, so two runs at the same seed drift apart during training
even though they start from identical weights. Nor does it hold across
platforms or PyTorch versions.

---

Back to: [Q&A index](README.md)
