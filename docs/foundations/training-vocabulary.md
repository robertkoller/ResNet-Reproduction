# Training Vocabulary

Every term the rest of the docs assume, explained properly.

**Assumed background:** basic statistics — mean, variance, distribution,
sampling. Nothing else. No machine learning.

Read straight through the first time; the terms build on each other. After
that, use it as a reference.

---

## 1. Loss

A single number measuring how wrong the network's prediction is. Lower is
better; zero is perfect.

For classification, the network outputs ten numbers — one score per CIFAR
class. Those get converted to a probability distribution over the ten classes
(via **softmax**, which exponentiates and normalizes so they sum to 1). The
loss is then the **cross-entropy**: the negative log of the probability the
network assigned to the correct class.

```
loss = −log( probability assigned to the true class )
```

Assign 0.9 to the right class and the loss is 0.105. Assign 0.1 and it's 2.303.
Assign something approaching 0 and the loss goes to infinity — confident and
wrong is punished savagely, which is the behaviour you want.

If you've seen maximum likelihood estimation, this is exactly that: minimising
cross-entropy is maximising the likelihood of the observed labels.

The loss is what training minimises. It is **not** the number you report —
that's error rate, further down.

---

## 2. Parameters and hyperparameters

**Parameters** are the numbers the network learns: conv weights, BatchNorm's
scale and shift. ResNet-20 has 269,722. Gradient descent adjusts them.

**Hyperparameters** are the numbers *you* choose before training: learning
rate, batch size, weight decay, how many layers. Gradient descent never touches
them.

The distinction matters because it splits the work in two. Parameters are found
automatically. Hyperparameters are found by experiment, intuition, or — in your
case — by copying them from the paper, which is what makes this a reproduction.

---

## 3. Gradient

For each parameter: *if I increase this slightly, does the loss go up or down,
and how fast?* Formally, the vector of partial derivatives of the loss with
respect to every parameter.

- **Sign** — which direction reduces the loss.
- **Magnitude** — how much this parameter is currently affecting the outcome.

It's computed by **backpropagation**, which applies the chain rule backward
through the network and gets every parameter's derivative for roughly the cost
of one forward pass. PyTorch does this when you call `loss.backward()`; you
never write it.

---

## 4. Learning rate

```
new_value = old_value − learning_rate × gradient
```

That one line is the entire training algorithm. Everything else is refinement.

### First, drop the walking metaphor

"Step size" makes it sound like something moves. Nothing moves.

A parameter is a number sitting in memory. Training **replaces that number with
a slightly different number**, 64,000 times. A "step" is one of those
replacements. "Step size" is just *how much the number changed*.

### Worked example, one parameter

Say a weight currently holds `4.0`, and its gradient comes back as `8.0` —
meaning "increasing this weight makes the loss worse, at a rate of 8." So
decrease it. By how much?

With `learning_rate = 0.1`:

```
new_value = 4.0 − 0.1 × 8.0 = 4.0 − 0.8 = 3.2
```

The weight went from 4.0 to 3.2. It changed by 0.8. **That 0.8 is the step.**

With `learning_rate = 0.01`:

```
new_value = 4.0 − 0.01 × 8.0 = 4.0 − 0.08 = 3.92
```

Step of 0.08 — ten times smaller.

**A precision point that trips people up:** the learning rate is *not* the step
size. The step is `learning_rate × gradient`. The learning rate is the
**multiplier** that scales it. People say "step size" loosely for the learning
rate, and it's sloppy enough to cause exactly this confusion.

### Watch it play out

The smallest possible complete example. One parameter `w`, loss `= w²`, so the
minimum is at `w = 0` and the gradient is `2w`. Start at `w = 4.0` and take
eight steps at six learning rates. Each row is one number being repeatedly
replaced — read left to right:

```
lr=0.01    4.000    3.920    3.842    3.765    3.689    3.616    3.543    3.473    3.403
lr=0.1     4.000    3.200    2.560    2.048    1.638    1.311    1.049    0.839    0.671
lr=0.4     4.000    0.800    0.160    0.032    0.006    0.001    0.000    0.000    0.000
lr=0.9     4.000   -3.200    2.560   -2.048    1.638   -1.311    1.049   -0.839    0.671
lr=1.0     4.000   -4.000    4.000   -4.000    4.000   -4.000    4.000   -4.000    4.000
lr=1.1     4.000   -4.800    5.760   -6.912    8.294   -9.953   11.944  -14.333   17.199
```

Reproduce it with a dozen lines:

```python
def run(learning_rate, steps=8, start=4.0):
    w = start
    history = [w]
    for _ in range(steps):
        gradient = 2 * w
        w = w - learning_rate * gradient      # the entire algorithm
        history.append(w)
    return history


for rate in [0.01, 0.1, 0.4, 0.9, 1.0, 1.1]:
    print(f"lr={rate:<5}", "  ".join(f"{value:8.3f}" for value in run(rate)))
```

Reading the rows:

| rate | behaviour |
|---|---|
| **0.01** | Steps of 0.08, 0.078, 0.077. Right direction, but eight steps only got from 4.0 to 3.4. Would need hundreds. **Too small.** |
| **0.1** | Steps of 0.8, 0.64, 0.51. Steady progress. **Healthy.** |
| **0.4** | Essentially zero in five steps. **Near-optimal here.** |
| **0.9** | Watch the signs. It flies *past* zero to −3.2, back past to +2.56. Overshoots every time, but each overshoot shrinks, so it still converges. **Ugly, works.** |
| **1.0** | Bounces between +4.0 and −4.0 forever. Zero progress, ever. **The exact boundary.** |
| **1.1** | Overshoots by *more* each time: 4 → −4.8 → 5.76 → −6.9 → 8.3. Running away from the answer. Continue and it exceeds float range and becomes `NaN`. **Divergence.** |

That last row is what "the loss diverged" means in practice, and it is exactly
what `warmup: true` exists to prevent at depth 110.

### Scaling up

Everything above is one number. ResNet-20 has **269,722** of them, and about
99% are convolution kernel entries.

Each iteration: run 128 images forward, compute one loss, backpropagate to get
269,722 gradients — one per parameter — and apply that same one-line update to
every one. Repeat 64,000 times.

**All 269,722 share a single learning rate.** That is the crux. One multiplier
must be simultaneously reasonable for every parameter in the network. If one
layer's gradients are a thousand times larger than another's, the multiplier
that gives a sensible step for one gives the `lr=1.1` runaway row for the
other.

**Yours is 0.1**, from the paper — which is large. It's usable precisely
because BatchNorm keeps every layer's activations on comparable scales, so one
multiplier suits all of them. Without it you'd need something like 0.001, and
110 layers wouldn't train at all. See
[`../block/02-batch-normalization.md`](../block/02-batch-normalization.md).

### Learning rate schedule

The rate doesn't stay fixed. Yours drops by 10× at iteration 32,000 and again
at 48,000: `0.1 → 0.01 → 0.001`.

The logic: big strides early to cross the landscape and find the right basin,
small strides later to settle at the bottom instead of bouncing around it.

You'll see this in your training curves as two sharp drops in error at exactly
those iterations. It's the most recognisable signature of a correct ResNet run
— if those cliffs don't appear, the schedule isn't wired up.

### Warmup

At depth 110, starting at 0.1 diverges immediately — the network is still
random, gradients are large, and a big step destroys it. So you run at 0.01
until the training error falls below 80% (about 400 iterations), then switch
to 0.1. That's **warmup**, and it's why the CIFAR-10 recipe has `warmup: true` only for
`n: 18`.

---

## 5. SGD, batches, and momentum

### Stochastic gradient descent

Computing the true gradient means evaluating the loss on all 45,000 training
images. Far too slow to do 64,000 times.

Instead you use a random **batch** — yours is 128 images — and take the
gradient of that. It's a noisy estimate of the true gradient: right on average,
wrong in detail. That's the **stochastic** in SGD.

Straight from sampling theory: a sample mean estimates a population mean, with
error shrinking as the sample grows. Batch size 128 trades accuracy per step
for many more steps, and the trade is overwhelmingly worth it.

The noise partly helps, too — it shakes the optimizer out of poor spots it
would otherwise settle into.

### Momentum

Rather than stepping along the current noisy gradient, keep a running average:

```
velocity = momentum × velocity + gradient
new_value = old_value − learning_rate × velocity
```

With `momentum: 0.9`, the velocity carries roughly a ten-step memory
(`1 / (1 − 0.9)`). It's an exponentially weighted moving average, the same
object as in time-series smoothing.

Two benefits: it averages away the sampling noise, and it builds speed along
directions where the gradient is consistent while cancelling directions where
it keeps flipping sign. The physical analogy is a ball rolling downhill rather
than a hiker re-deciding at every step.

---

## 6. Iterations and epochs

Two different units, and mixing them up will break the schedule.

- **Iteration** (or step) — one batch: forward, loss, backward, update. One
  gradient step.
- **Epoch** — one full pass over the training set.

With 45,000 images and batch size 128, one epoch is ~352 iterations.

**The schedule is in iterations**, because that's how the paper specifies it.
64,000 iterations × 128 images = 8.19 million images seen ≈ **182 epochs**. The
learning rate drops land near epoch 91 and epoch 136.

Convert carefully if you ever need to. Papers differ on which unit they use,
and a schedule off by a factor of 352 is not a subtle bug.

---

## 7. Train, validation, and test

Three disjoint sets of images:

- **Training set** (45,000) — what the network learns from.
- **Validation set** (5,000) — held out, used to check progress and make
  decisions.
- **Test set** (10,000) — held out, touched once, at the end.

The rule: **any set you use to make a decision is no longer an honest estimate
of performance.** If you try five configurations and pick the one that scores
best on a set, that score is optimistically biased — you've selected on noise.
This is multiple comparisons, exactly as in statistics.

That's why the validation set exists: it absorbs the decisions so the test set
stays clean.

CIFAR-10 ships as 50,000 train and 10,000 test. The project plan carves 5,000 off the
training set for validation.

---

## 8. Overfitting and underfitting

**Overfitting** — the model does well on data it has seen and badly on new
data. It has memorised rather than generalised. The signature is a widening gap
between training and test error.

**Underfitting** — the model does badly on both. It lacks the capacity, or
hasn't trained long enough.

**Capacity** is roughly how complex a function the model can represent, driven
mostly by parameter count. More capacity means more expressive power and more
room to memorise.

### The distinction this whole project depends on

Read this twice, because it is the paper's central point:

> **Overfitting is high *test* error with low *training* error. The degradation
> problem is high *training* error.**

When the paper's 56-layer plain network does worse than the 20-layer one, it is
worse on the **images it trained on**. That cannot be overfitting — a bigger
model that had memorised would show *lower* training error. It's an
optimization failure: a good solution exists (copy the 20-layer network, set
the extra layers to pass their input through) and gradient descent can't find
it.

Both failure modes appear in the results, and the write-up has to keep them
apart:

- **plain-56 worse than plain-20 on training error** → degradation
- **the 1202-layer net at 0.1% training error but worse test error than the
  110** → overfitting

---

## 9. Regularization

Anything that trades a little training accuracy for better generalisation.
Three kinds appear in this project.

### Weight decay

Add a penalty proportional to the squared size of the weights, which in
practice pulls every weight slightly toward zero at each step:

```
new_value = old_value − learning_rate × (gradient + weight_decay × old_value)
```

This is **L2 regularization**, identical in spirit to ridge regression. Large
weights mean the output swings hard on small input changes — a brittle,
high-variance function. Penalising size prefers smoother ones.

Yours is `0.0001`. Small: a nudge, not a constraint.

### Data augmentation

Manufacture more training data by transforming what you have. Your recipe: pad
the image with 4 pixels, randomly crop back to 32×32, randomly flip
horizontally.

A dog shifted three pixels left is still a dog, so the label survives the
transformation — and the network is forced to learn that position and handedness
don't matter, rather than memorising exact pixel layouts. Applied to training
data only; test data is never augmented.

### Dropout

During training, randomly zero out some fraction of a layer's units — typically
half — choosing different ones each step. At test time nothing is dropped.

The purpose is to prevent **co-adaptation**: units forming fragile chains where
one only works because another always fires alongside it. If any unit might
disappear, none can depend on another, so each must be independently useful.
It's approximately bagging — training a huge ensemble of thinned networks that
share weights.

**This paper uses none.** BatchNorm's batch-dependent noise does the same job
as a side effect, there are no large fully connected layers to protect, and
dropout interacts badly with BatchNorm. See
[`../block/02-batch-normalization.md`](../block/02-batch-normalization.md).

---

## 10. Error rate — the number you report

**Error rate** = fraction of images classified wrongly. **Accuracy** = 1 −
error rate. The paper reports error, so you report error; mixing conventions
makes tables incomparable.

Not the same as loss. Loss is a smooth, differentiable surrogate that gradient
descent can work with. Error rate is what you actually care about but it's a
step function — flat almost everywhere, so its gradient carries no information.
You optimise the loss and report the error.

**Top-1 error** means the single highest-scoring class was wrong. **Top-5**
means the true class wasn't in the top five guesses — used on ImageNet's 1000
classes, not on CIFAR-10's ten.

The targets: 8.75% at depth 20, down to 6.43% at depth 110.

---

## 11. Initialization

The weights start at random values. Which random values matters more than it
sounds.

Too large and activations explode as they pass through layers. Too small and
they shrink to nothing. Either way a deep network is untrainable before
training starts.

**He initialization** (from these same authors) samples from a normal
distribution with variance `2 / fan_in`, where `fan_in` is the number of input
connections to the unit. The `2` corrects for ReLU discarding negatives, which
halves the variance. It keeps activation scale roughly constant through depth.

This is the *statistical* fix for scale drift.
BatchNorm is the *dynamic* fix — it re-imposes the scale at every layer, every
step. Modern networks use both, which is belt and braces, and it's what lets
you train 110 layers.

---

## 12. Seeds and reproducibility

A **seed** initialises the random number generator, so a run using the same
seed makes the same random choices: the same initial weights, the same batch
order, the same augmentation crops.

Why it matters here specifically: your target numbers at depths 44 and 56 are
7.17% and 6.97% — a gap of 0.2 percentage points. Run-to-run variation from
seeds alone is comparable in size. So a single run of each cannot tell you
whether the difference is real, which is why the paper reports five runs for
ResNet-110 and the project plan copies that.

Seeding gives you reproducible *initialization*. It does not give bitwise
identical *training* on MPS or across platforms — different hardware sums
floating point in different orders. Note that in the deviations section rather
than fighting it.

---

## Quick reference

| term | one line |
|---|---|
| loss | one number for how wrong a prediction is |
| cross-entropy | the loss used for classification; negative log of the true class's probability |
| gradient | the slope of the loss with respect to each parameter |
| backpropagation | the algorithm computing all gradients efficiently |
| learning rate | step size along the gradient |
| schedule | planned reductions of the learning rate |
| warmup | starting low, then raising, to survive early instability |
| SGD | gradient descent on random batches rather than all data |
| batch | the group of images in one step; yours is 128 |
| momentum | running average of gradients; yours is 0.9 |
| iteration | one batch, one update |
| epoch | one full pass over the training set; ~352 iterations here |
| parameter | a number the network learns |
| hyperparameter | a number you choose |
| capacity | how complex a function the model can represent |
| overfitting | good on seen data, bad on new data |
| underfitting | bad on both |
| degradation | *training* error rising with depth — this paper's subject |
| regularization | trading training fit for generalisation |
| weight decay | L2 penalty pulling weights toward zero |
| augmentation | transforming training images to manufacture more data |
| dropout | randomly zeroing units during training; not used here |
| error rate | fraction misclassified; what you report |
| accuracy | 1 − error rate |
| initialization | the starting random weights; He init here |
| seed | fixes the random choices so a run is repeatable |

---

## Related

- [`../block/README.md`](../block/README.md) — the architecture
- [`../library/README.md`](../library/README.md) — the machinery underneath
- `resources/notes/resnet-algorithm-summary (1).md` — the paper in plain
  English
