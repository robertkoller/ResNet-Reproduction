# ResNet Reproduction — Step-by-Step Plan

*Written for: a Mac, free/cheap cloud GPU, and no prior deep learning project.*

---

## Read this before you start

**What you're building.** Two versions of the same network — one with the shortcut connections, one without — at several depths. You train them all, and you show that the one without shortcuts gets *worse* as it gets deeper while the one with shortcuts gets *better*. That's the paper's central claim and it's the whole point of the project.

**What you're NOT building.** The full ImageNet experiment. It's 150GB of photos and about a week of nonstop GPU time. Skip it entirely.

**Why this is worth putting on a resume.** Implementing ResNet by itself isn't impressive — it ships built into PyTorch, and every tutorial does it. What's impressive is running the *controlled comparison*: same depth, same parameter count, only difference is the shortcut. That's an experiment, not a tutorial. Say it that way when you talk about it.

---

## Step 0: Figure out your hardware

**If you have an Apple Silicon Mac (M1/M2/M3/M4):** you have a usable GPU. PyTorch reaches it with `device = "mps"`. Test it:

```python
import torch
print(torch.backends.mps.is_available())   # want True
```

It works, but it's slow-ish. ResNet-20 will take a few hours; ResNet-110 might take 10+.

**If you have an Intel Mac:** no usable GPU. CPU only. ResNet-110 would take days. Don't try.

**Either way, use Google Colab for the actual training runs.** Free tier gives you an NVIDIA T4, which is several times faster than an M-series Mac for this. Colab Pro is about $10/month and gets you better GPUs plus longer sessions.

The workflow that works well:
- Write and debug on your Mac, with a tiny config (a few hundred iterations, small model) just to check nothing crashes.
- Push to GitHub.
- Clone into Colab and run the real training there.
- Save checkpoints and metric files to Google Drive so a disconnect doesn't cost you a run.

Colab disconnects after a few hours of idleness or ~12 hours of use. Checkpointing isn't optional — build it in Step 3.

**Tasks:**
- [ ] Check whether MPS is available on your Mac
- [ ] Make a Colab account, open a notebook, run `!nvidia-smi` to confirm you got a GPU
- [ ] Make a GitHub repo
- [ ] Write a `get_device()` function that picks cuda → mps → cpu automatically

---

## Step 1: Project setup (half a day)

Make the folder structure:

```
resnet-repro/
  models/       the network code
  data/         dataset loading
  train/        the training loop
  configs/      one small file per experiment
  results/      metric CSVs and saved plots
  tests/        correctness checks
```

**Put experiment settings in config files, not in your code.** You'll be running 15+ experiments that differ only in depth and whether shortcuts are on. If those are hardcoded, you'll be editing source files constantly and losing track of what produced what. A config file looks like:

```yaml
name: resnet56_seed0
n: 9
residual: true
shortcut: A
seed: 0
batch_size: 128
lr: 0.1
```

**Tasks:**
- [ ] Create the folder structure and a GitHub repo
- [ ] `pip install torch torchvision matplotlib pyyaml`
- [ ] Write a config loader
- [ ] Write a `set_seed(n)` function that seeds Python, NumPy, and PyTorch
- [ ] Write `train.py` that loads a config and prints it (nothing else yet)

**You're done when:** `python train.py --config configs/test.yaml` runs and prints the settings.

---

## Step 2: Build the network (2–3 days)

Go in this order. Test each piece before moving on.

**2a. The basic block.** Two 3×3 convs with BN and ReLU, plus the shortcut addition. Pass the shortcut in as an argument rather than deciding inside the block — you'll want to swap it out later.

**2b. The shortcut variants.** Three small classes:
- `IdentityShortcut` — just returns the input. Used when sizes already match.
- `ZeroPadShortcut` (option A) — take every other pixel (that's the stride-2 downsampling) and pad the missing channels with zeros.
- `ProjectionShortcut` (options B/C) — a 1×1 conv with stride 2.

**Option A is the one people get wrong.** Write a test: feed in a tensor of shape (1, 16, 32, 32), confirm you get out (1, 32, 16, 16), and confirm the second half of the channels are all zeros.

**2c. The CIFAR network.** Takes `n` and `residual` as arguments. Three stages of `2n` layers each at 16, 32, 64 channels. Total depth = 6n+2.

**2d. The plain mode.** When `residual=False`, the exact same code but the shortcut addition is skipped. Everything else identical.

**2e. He initialization** applied to every conv.

**2f. Counters** for parameters and FLOPs.

**2g. The bottleneck block and ImageNet-style network.** You need these for the Imagenette experiment later.

**Tasks — write these tests and make them pass:**
- [ ] `resnet(n=3)` has ~0.27M params, `n=5` → ~0.46M, `n=7` → ~0.66M, `n=9` → ~0.85M, `n=18` → ~1.7M
- [ ] `resnet(n=3, residual=True)` and `resnet(n=3, residual=False)` have **identical** parameter counts
- [ ] Counting conv and linear layers in `resnet(n=3)` gives exactly 20
- [ ] A random batch through the network gives shape (batch, 10) with no NaNs
- [ ] Zero-pad shortcut output shape and zero-channels test

If the parameter counts don't match the table, something's wrong with your architecture. Fix it now — every downstream result depends on this.

---

## Step 3: Data and training loop (1–2 days)

**3a. CIFAR-10.** `torchvision.datasets.CIFAR10` downloads it automatically. Split off 5,000 training images as a validation set and save which indices you used, so it's reproducible.

**3b. Augmentation.** Pad 4 pixels on each side, random-crop back to 32×32, random horizontal flip. Test set gets none of this.

**3c. Normalization.** Subtract the mean pixel values, computed from the training set only.

**3d. The training loop.** SGD, momentum 0.9, weight decay 0.0001, batch size 128.

**3e. Learning rate schedule.** Count **iterations** (batches), not epochs — the paper specifies 32k/48k/64k in iterations. Start at 0.1, ÷10 at 32,000, ÷10 at 48,000, stop at 64,000.

**3f. Warmup.** For depth 110: run at LR 0.01 until training error drops below 80%, then switch to 0.1. Log which iteration it flipped at (should be around 400).

**3g. Checkpointing.** Save model weights, optimizer state, and iteration number every ~2,000 iterations. Add a `--resume` flag. Colab *will* disconnect on you.

**3h. Logging.** Append training error and test error to a CSV every ~500 iterations. Never store only the final number — you need the curves for your plots.

**Tasks:**
- [ ] Look at a batch of augmented images to confirm the augmentation looks right
- [ ] Overfit test: train on just 100 images with no augmentation. Training error should hit ~0%. If it can't memorize 100 images, your training loop is broken — find that now rather than after a 6-hour run.
- [ ] Full ResNet-20 run. Target: around 8.75% test error.

**You're done when:** ResNet-20 lands somewhere near 8–9%. If you're above 11%, stop and debug. Don't scale up on top of a bug.

---

## Step 4: The main experiments (this is mostly waiting)

Run each of these and save the full metric CSV.

- [ ] **Plain nets** at n = 3, 5, 7, 9 → depths 20, 32, 44, 56
- [ ] **ResNets** at n = 3, 5, 7, 9 → same depths
- [ ] **ResNet-110** (n=18), run **5 times with different seeds**
- [ ] **Plain-110** — expect this to fail badly, error above 60%. That failure *is* a result.

Targets, ResNet: 20 → 8.75%, 32 → 7.51%, 44 → 7.17%, 56 → 6.97%, 110 → 6.43%.

Landing within a few tenths of a percent counts as a successful reproduction. Matching exactly is not expected and you shouldn't claim it.

**Run the plain nets first.** If your plain networks *don't* get worse with depth, you haven't reproduced the paper's premise and nothing else you do matters. Check that before spending compute on anything else.

**Rough time budget on a Colab T4:** ResNet-20 ≈ 45 min, ResNet-56 ≈ 2 hours, ResNet-110 ≈ 4 hours. Ten runs plus five seeds of the 110 puts you around 30–40 GPU-hours total. Spread over a couple of weeks that's very doable; don't try to do it in a weekend.

---

## Step 5: Imagenette (optional, 1–2 days plus compute)

This exercises the bottleneck block and the ImageNet-style architecture on real full-resolution photos, without needing 150GB.

Imagenette is a 10-category slice of ImageNet, about 1.5GB, downloadable from the fast.ai GitHub repo. There's a 160px version that's smaller and faster — use that one.

- [ ] Load Imagenette (folder-per-class structure, `torchvision.datasets.ImageFolder` handles it)
- [ ] Augmentation: random resized crop to 160 or 224, random horizontal flip, normalize
- [ ] Train ResNet-18 (basic blocks) and ResNet-50 (bottleneck blocks) — change the final layer to 10 outputs instead of 1000
- [ ] Compare them; check ResNet-50 comes out ahead
- [ ] Optionally add plain-18 and plain-34 to show degradation shows up here too

**Be explicit that this is not ImageNet.** Your numbers won't be comparable to Table 3 in the paper and shouldn't be presented as if they were. What you're demonstrating is that the architecture is correctly implemented and behaves the way it should on real photos. That's a legitimate claim; overstating it is not.

Rough time: ResNet-50 on Imagenette-160 is maybe 1–2 hours on a T4.

---

## Step 6: Analysis and plots (1–2 days)

- [ ] **Training curves.** For each depth, plot training and test error against iteration. Two panels side by side: plain on the left, residual on the right. This one figure carries the whole story.
- [ ] **Depth ladder.** Final test error vs. depth, one line for plain and one for residual. They go in opposite directions. This is your headline plot.
- [ ] **Layer output magnitudes.** Register forward hooks on each 3×3 conv, capture the output after BN and before ReLU, compute its standard deviation. Plot for plain-20, plain-56, ResNet-20, ResNet-56, ResNet-110. You should see residual layers producing smaller outputs, shrinking further with depth. This is the paper's evidence for *why* the trick works — and most reproductions skip it, which is exactly why including it makes yours stand out.
- [ ] **Shortcut ablation** (optional). Train depth-56 with options A, B, and C. Expect small differences.
- [ ] **All plots regenerable from the saved CSVs by one script**, without retraining anything.

---

## Step 7: Write it up (1–2 days)

The writeup matters as much as the code. Most GitHub ML projects are a pile of scripts with a two-line README; a real one stands out immediately.

- [ ] **README** opening with your headline plot and a table: paper's number, your number, difference — for every configuration you ran.
- [ ] **A "deviations" section.** List everything you did differently: hardware, PyTorch version, per-channel vs. per-pixel normalization, any schedule adjustments, anything you approximated. Being upfront about this is the single strongest signal that you know how research works. Hiding it is what people do when they don't.
- [ ] **A "what surprised me" paragraph.** Something always doesn't reproduce cleanly. Write about it honestly.
- [ ] `results/` with the raw CSVs so anyone can check your numbers.
- [ ] Tests runnable with one command.
- [ ] Setup instructions someone else could actually follow.

---

## Resume bullets

Write these after you have real numbers, using only numbers you produced:

> Reproduced He et al. (CVPR 2016) on CIFAR-10 from scratch in PyTorch: implemented residual and plain networks at depths 20–110 with matched parameter counts, confirming the degradation problem in plain networks and reaching 6.4% test error at 110 layers across 5 random seeds.

> Extended the reproduction with layer-response variance analysis and a shortcut-projection ablation, and validated the bottleneck architecture on Imagenette; documented all deviations from the original protocol.

Two specific bullets beat five vague ones.

**Prepare for the obvious interview question:** "why do plain networks get worse with more layers?" Your answer: a deeper network can always copy a shallower one by making the extra layers do nothing, so a good solution provably exists — the optimizer just can't find it. Residual connections make "do nothing" the easy default instead of something the network has to learn. And add that the paper offers this as a hypothesis rather than proving it. That last part is what makes you sound like you read the paper instead of a blog post about it.

---

## Time estimate

| Phase | Your time | GPU time |
|---|---|---|
| Setup + model + tests | 3–4 days | ~0 |
| Training loop + first run | 2 days | 1 hour |
| Main CIFAR experiments | mostly waiting | 30–40 hours |
| Imagenette | 1–2 days | 2–4 hours |
| Analysis and plots | 1–2 days | ~1 hour |
| Writeup | 1–2 days | 0 |

Realistically 4–6 weeks part-time alongside classes.

**If you need to cut scope:** drop Imagenette first, then reduce the ResNet-110 seeds from 5 to 3. Do **not** cut the plain-network baselines — without them you have an implementation, not a reproduction, and the resume bullet stops being true.
