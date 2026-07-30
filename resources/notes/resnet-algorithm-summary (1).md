# ResNet, Explained Plainly

*A summary of "Deep Residual Learning for Image Recognition" — He, Zhang, Ren, Sun, CVPR 2016*

---

## Part 1: The vocabulary

**Neural network for images.** A stack of layers. You feed a photo in the bottom, and it passes upward through the stack. Each layer looks for patterns in what the layer below it found. The bottom layers find simple things — edges, corners, patches of color. Middle layers combine those into textures and parts, like "fur" or "wheel." Top layers combine those into whole objects. The very last layer outputs a guess: "dog."

**Layers / depth.** How many layers you stack. ResNet-20 has 20, ResNet-110 has 110. More layers means the network can represent more complicated patterns — that's the intuition, anyway, and this paper is about where that intuition breaks.

**Training.** You show the network a photo, it guesses, you tell it the right answer, and it nudges all its internal numbers slightly toward guessing better. Repeat a few hundred thousand times. The nudging algorithm is called SGD (stochastic gradient descent).

**Training error vs. test error.** Training error is how often it gets the photos *it has already studied* wrong. Test error is how often it gets *new, unseen* photos wrong. Test error is always higher — that's normal, and the gap is called overfitting (memorizing instead of understanding). Training error going up is a completely different and much stranger problem, and it's the one this paper is about.

**ImageNet.** The big benchmark dataset. 1.28 million photos, 1000 categories, about 150GB. Too big for a student project.

**CIFAR-10.** The small benchmark. 60,000 photos at 32×32 pixels, 10 categories (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck), about 170MB. This is what you'll actually use.

**Imagenette.** A slice of ImageNet — 10 categories that are easy to tell apart, at full photo resolution, about 1.5GB. Useful because it's shaped like ImageNet (big images, real photos) without being enormous.

**ResNet.** The architecture this paper invented. Short for "Residual Network."

---

## Part 2: The problem

Here's the mystery the authors found.

Take a 20-layer network. Train it on CIFAR-10. Now build a 56-layer network — same kind of layers, just more of them — and train it the same way.

**The 56-layer network is worse.** Not just on new photos. It's worse on the photos it studied. Its *training* error is higher.

That should be impossible, and here's the argument for why. Imagine you take your trained 20-layer network and bolt 36 extra layers on top, where every one of those extra layers is set to just pass its input through unchanged — do nothing. That 56-layer network would behave *identically* to the 20-layer one. Same error, exactly.

So a good 56-layer solution definitely exists. The training process just can't find it.

The authors call this the **degradation problem**. It's not overfitting (that would show up as a train/test gap, not higher training error). It's not vanishing gradients (a known older problem, already fixed by other techniques they were using). It's that the optimizer gets lost when the network gets deep.

---

## Part 3: The fix

Normally, a chunk of a few layers has a job like: "take this input, produce this output."

ResNet changes the job. It runs a wire from the chunk's input directly to its output, and adds them together:

```
        ┌─────────────────────┐
input ──┤                     ├──┐
        │  two conv layers    │  │
        └─────────────────────┘  │
                  │              │
                  ▼              │
              output of      +   │  ← the shortcut wire
              the layers         │
                  │              │
                  ▼◄─────────────┘
              final output
```

Written as math: instead of the layers producing `H(x)`, they produce `F(x)`, and the block's output is `F(x) + x`. So the layers only have to produce `F(x) = H(x) - x` — the **difference**, or **residual**, between what you want and what you already have.

**Why this helps.** Go back to the "do nothing" idea. If the best thing for a chunk of layers to do is pass its input through unchanged, then:

- Without the shortcut, the layers have to learn to exactly reproduce their input. Through several nonlinear layers, that turns out to be genuinely hard.
- With the shortcut, they just have to output **zero**. Push all the weights toward zero and you're done. Trivially easy.

And more generally: even when "do nothing" isn't quite right, the layers only have to learn a small adjustment on top of the input, rather than building the whole answer from scratch. Starting from "roughly correct" and nudging beats starting from "random" and constructing.

**The shortcut is free.** It's just addition — no weights, no extra computation. That matters a lot, because it means you can compare a plain network and a residual network that have *exactly* the same number of layers, the same number of parameters, and the same speed. Any difference in results is purely due to the shortcut. That's a clean experiment.

**Important honesty note:** the authors present this as a hypothesis, not a proof. They say "we hypothesize" and "may help to precondition the problem." They show it works; they don't prove why. If someone asks you about this in an interview, saying so is a point in your favor.

---

## Part 4: What actually gets built

### The building block

The basic unit, used in the shallower networks:

```
input ──► 3×3 conv ──► BN ──► ReLU ──► 3×3 conv ──► BN ──► (+) ──► ReLU ──► output
   │                                                        ▲
   └────────────────── shortcut ────────────────────────────┘
```

- **conv** (convolution) = the pattern-finding operation. "3×3" means it looks at 3×3 pixel neighborhoods.
- **BN** (batch normalization) = rescales the numbers so they stay in a sane range. Keeps training stable.
- **ReLU** = the nonlinearity. Literally: negative numbers become zero, positive numbers stay. Simple, and it's what lets the network learn non-straight-line relationships.
- Note the final ReLU comes **after** the addition, not before. Small detail, easy to get wrong.

### The bottleneck block

For the very deep versions (50+ layers), a cheaper block:

```
input ──► 1×1 conv (squeeze channels down)
      ──► 3×3 conv (do the expensive work on fewer channels)
      ──► 1×1 conv (expand channels back up, 4× wider)
      ──► (+ shortcut) ──► ReLU
```

The 1×1 convs squeeze and unsqueeze so the expensive 3×3 operates on less data. Same cost as a basic block, but three layers instead of two. The paper is clear this was a compute-budget decision, not an accuracy one.

### When sizes don't match

Every so often the network shrinks the image (32×32 → 16×16) and doubles the channel count. Now the shortcut wire is carrying something a different shape from what it needs to add to. Three options the paper tests:

- **Option A** — keep the identity shortcut, pad the missing channels with zeros. Adds no parameters. Used for all the CIFAR experiments.
- **Option B** — use a small 1×1 conv on the shortcut, but *only* where sizes change. Adds a few parameters.
- **Option C** — use a 1×1 conv on *every* shortcut. Adds a lot.

Results: A gets 25.03% error, B gets 24.52%, C gets 24.19%. All three crush the plain network. The differences between them are small, so the paper concludes the fancy versions aren't necessary and sticks with A or B.

### The full CIFAR-10 network

Simple by design:

1. One 3×3 conv, 16 channels.
2. `2n` layers at 32×32 resolution, 16 channels.
3. `2n` layers at 16×16 resolution, 32 channels.
4. `2n` layers at 8×8 resolution, 64 channels.
5. Average everything down to one number per channel, then a 10-way output layer.

Total layers = **6n + 2**. Pick n and you get your depth:

| n | depth | parameters | paper's test error |
|---|---|---|---|
| 3 | 20 | 0.27M | 8.75% |
| 5 | 32 | 0.46M | 7.51% |
| 7 | 44 | 0.66M | 7.17% |
| 9 | 56 | 0.85M | 6.97% |
| 18 | 110 | 1.7M | 6.43% |
| 200 | 1202 | 19.4M | 7.93% |

### The ImageNet-style network

Same ideas, bigger inputs (224×224 instead of 32×32):

- Starts with a 7×7 conv and a pooling layer to shrink things down fast.
- Four stages instead of three, channels going 64 → 128 → 256 → 512.
- Ends with average pooling and a 1000-way output layer.

Depths of 18, 34, 50, 101, 152. The 18 and 34 use basic blocks; 50/101/152 use bottleneck blocks.

Neat detail worth mentioning in a writeup: ResNet-152 is 8× deeper than the previous champion (VGG) but does *less* total computation, because it has no giant fully-connected layers at the end.

---

## Part 5: The training recipe

For CIFAR-10, the paper specifies:

- **Optimizer:** SGD, momentum 0.9, weight decay 0.0001
- **Batch size:** 128 photos at a time
- **Learning rate:** starts at 0.1, divided by 10 at iteration 32,000, divided by 10 again at 48,000, stop at 64,000
- **Data augmentation:** pad the image with 4 pixels of border, randomly crop back to 32×32, randomly flip horizontally. (This is how you fake having more training data.)
- **No dropout.** BN is doing the regularizing.
- **Special case for 110 layers:** starting at learning rate 0.1 is too aggressive and it won't converge. Start at 0.01 until training error drops below 80% (takes about 400 iterations), then bump to 0.1. This is called warmup.

---

## Part 6: The results you're trying to reproduce

**The main one.** Plain networks get worse as they get deeper. ResNets get better.

| depth | plain net | ResNet |
|---|---|---|
| 20 | 8.75%-ish | 8.75% |
| 56 | worse than 20 | 6.97% |
| 110 | fails badly (>60% error) | 6.43% |

**Secondary results, in rough order of how impressive they are to reproduce:**

- The 1202-layer network trains fine (training error under 0.1%) but tests *worse* than the 110-layer one. That's overfitting — it's too big for such a small dataset. Different failure mode entirely from the degradation problem.
- Residual layers produce smaller outputs than plain layers, and the effect grows with depth. This is the paper's evidence for *why* the trick works — each layer is only making a small adjustment, exactly as the theory predicts.
- Shortcut options A/B/C barely differ.
- ResNet-101 dropped into an object detector improves it by 28% over the previous backbone — the representations transfer to other tasks.

---

## Part 7: Everything you need to write, as a checklist

**Model pieces**
- [ ] Conv layer helper (3×3 and 1×1 versions, no bias — BN handles that)
- [ ] Basic block (2 convs + shortcut)
- [ ] Bottleneck block (3 convs + shortcut, 4× expansion)
- [ ] Shortcut option A: identity with zero-padded channels
- [ ] Shortcut option B: 1×1 conv, only where sizes change
- [ ] Shortcut option C: 1×1 conv everywhere
- [ ] Stage builder: stacks N blocks, handles the size change at the first one
- [ ] CIFAR network: 6n+2 layers, three stages
- [ ] ImageNet-style network: 7×7 stem + pool, four stages
- [ ] **Plain network mode:** same code with the shortcuts switched off. This is the control group — without it you have no experiment.
- [ ] He weight initialization
- [ ] Parameter counter and FLOP counter (to verify you built it right)

**Data**
- [ ] CIFAR-10 loader with mean subtraction
- [ ] Augmentation: pad 4 → random crop 32 → random flip
- [ ] Held-out validation split (45k train / 5k val)
- [ ] Imagenette loader (for the optional bigger experiment)

**Training**
- [ ] SGD with momentum and weight decay
- [ ] Learning rate schedule tied to iteration count, not epochs
- [ ] Warmup rule for the 110-layer network
- [ ] Save/resume checkpoints
- [ ] Log training and test error to a file so you can plot later without retraining

**Evaluation**
- [ ] Error rate calculation
- [ ] Run the same config with multiple random seeds (the paper reports 5 runs for ResNet-110)

**Analysis**
- [ ] Training curves: plain vs. residual at each depth
- [ ] Layer output magnitudes: hook into each conv, measure how big its outputs are, plot
- [ ] Ablation runner: sweep depths and shortcut options from config files
