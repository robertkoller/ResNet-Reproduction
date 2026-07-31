# 00 — Tensors, Channels, and Why an Image Is `3 × 32 × 32`

Read this first. Every confusing shape error later in the project comes from
not having this straight.

---

## A tensor is just a grid of numbers

A **tensor** is an array of numbers with any number of dimensions. That's the
entire concept. The name sounds exotic; the thing is not.

| dimensions | name | example |
|---|---|---|
| 0 | scalar | `7.2` — the loss |
| 1 | vector | `[0.1, 0.4, 0.5]` — 3 numbers |
| 2 | matrix | a 32×32 grid — a greyscale image |
| 3 | 3D tensor | a colour image |
| 4 | 4D tensor | a batch of colour images |

A tensor's **shape** is the list of its dimension sizes. `(2, 16, 32, 32)` is a
4-dimensional tensor holding `2 × 16 × 32 × 32 = 32,768` numbers.

In PyTorch you check a shape with `.shape`, and you will do this constantly.

---

## Why a 2D image is `3 × 32 × 32`

Here's the thing that trips everyone up.

A CIFAR-10 image **is** 32×32. Thirty-two pixels wide, thirty-two tall. 1,024
pixels total. That part is genuinely two-dimensional.

But a pixel isn't one number. A colour pixel is **three** numbers: how much
red, how much green, how much blue. Each on a 0–255 scale, or 0.0–1.0 once
normalized.

So you have 1,024 positions, each holding 3 numbers — a total of 3,072 numbers.

There are two equivalent ways to picture that arrangement:

**As one grid of triples:**

```
        column 0        column 1        column 2      ...
row 0   (R,G,B)         (R,G,B)         (R,G,B)
row 1   (R,G,B)         (R,G,B)         (R,G,B)
row 2   (R,G,B)         (R,G,B)         (R,G,B)
...
```

**As three separate grids stacked** — which is how PyTorch actually stores it:

```
         ┌────────────────────┐
         │  BLUE   32 × 32    │
       ┌─┴──────────────────┐ │
       │  GREEN  32 × 32    │ │
     ┌─┴──────────────────┐ │ │
     │  RED    32 × 32    │ │─┘
     │                    │─┘
     └────────────────────┘
```

Three grids, each 32×32. Written as a shape: `(3, 32, 32)`.

The image is still 2D in the sense that matters visually. The `3` isn't a third
spatial direction — there's no depth into the screen. It's **how many numbers
are measured at each location**.

Read the shape as: *3 measurements per position, over a 32-by-32 grid of
positions.*

---

## Channels

Each of those grids is a **channel**.

For the input image, the three channels mean something you can name: red,
green, blue. That's the only place in the whole network where channels have
human-readable meanings.

After the first conv layer, the image becomes `(16, 32, 32)` — sixteen
channels. These are **not** colours. Channel 0 might be "how strongly does a
vertical edge appear here." Channel 7 might be "how strongly does a
green-to-brown transition appear here." Nobody assigns these meanings; the
network discovers them during training, and mostly they don't correspond to
anything nameable.

The general definition, which covers both cases:

> A channel is one grid of numbers, where the number at each position says how
> strongly one particular thing is present at that location in the image.

Channels are also called **feature maps** — "map" because it's laid out
spatially, "feature" because it's detecting some feature. The terms are
interchangeable and you'll see both.

Why "channel"? The word comes from signal processing, where a colour image was
literally transmitted as three separate signal channels. The name stuck.

---

## The batch dimension

Training doesn't process one image at a time. The configuration says
`batch_size: 128`, so 128 images go through together.

Stack 128 images of shape `(3, 32, 32)` and you get `(128, 3, 32, 32)`.

That is the shape flowing through the network. Four dimensions, in PyTorch's
standard order:

```
( N , C , H , W )
  │   │   │   └── Width  — 32
  │   │   └────── Height — 32
  │   └────────── Channels — 3 at the input, 16/32/64 inside the network
  └────────────── Number of images in the batch — 128
```

This ordering is called **NCHW**, and it's the convention every PyTorch vision
layer expects. (TensorFlow historically used NHWC — channels last. If you read
TensorFlow code and the shapes look backwards, that's why.)

Layers do not care what `N` is. A conv layer processes each image in the batch
identically and independently. That's why you can train with batches of 128 and
then evaluate on a batch of 1, using the same weights, with no changes.

There is one important exception, and it's the reason batch norm gets its own
document in this folder: **BatchNorm does look across the batch dimension.**
It's the only layer in the block that does.

---

## How the shape changes through the network

Trace one image through ResNet-20 and watch the two trades happening:

| after | shape | what changed |
|---|---|---|
| input | `(3, 32, 32)` | — |
| stem conv | `(16, 32, 32)` | 3 channels → 16 |
| stage 1 (3 blocks) | `(16, 32, 32)` | nothing; blocks preserve shape |
| stage 2 first block | `(32, 16, 16)` | halved spatially, doubled channels |
| stage 2 rest | `(32, 16, 16)` | nothing |
| stage 3 first block | `(64, 8, 8)` | halved again, doubled again |
| stage 3 rest | `(64, 8, 8)` | nothing |
| average pool | `(64,)` | each channel collapsed to one number |
| fully connected | `(10,)` | one score per CIFAR class |

The pattern is a deliberate trade: **give up spatial resolution, buy feature
variety.** Early layers know precisely *where* things are but little about
*what* they are. Late layers know a great deal about *what* is present but have
almost thrown away *where*.

Notice the total data size stays roughly constant:

- `16 × 32 × 32 = 16,384`
- `32 × 16 × 16 = 8,192`
- `64 × 8 × 8 = 4,096`

Halving each spatial dimension quarters the pixel count; doubling the channels
only doubles it back. So the tensor shrinks by 2× at each stage boundary while
the compute per layer stays in the same ballpark.

---

## The average pool step, which surprises people

`(64, 8, 8)` → `(64,)` looks like a lot of information vanishing.

**Global average pooling** takes each of the 64 channels and averages all 64 of
its spatial values into a single number. The result answers: *on average, how
strongly was this feature present anywhere in the image?*

By this depth that's the right question. Each channel has become something like
"dog-ish texture," and whether it appeared in the top-left or the middle no
longer matters for classifying the picture.

The alternative — flattening `64 × 8 × 8 = 4,096` values into a fully connected
layer — would need `4,096 × 10 = 40,960` parameters instead of `64 × 10 = 640`,
and would overfit. Average pooling is one reason ResNet-152 does *less*
computation than VGG despite being eight times deeper.

---

## Terms from this page

- **Tensor** — an n-dimensional array of numbers.
- **Shape** — the tuple of dimension sizes, e.g. `(128, 3, 32, 32)`.
- **Channel** — one grid of numbers measuring one thing across all positions.
- **Feature map** — same as a channel; the usual word once you're past the
  input layer.
- **Batch** — a group of images processed together in one step.
- **NCHW** — PyTorch's dimension order: batch, channels, height, width.
- **Spatial dimensions** — the height and width, as opposed to the channel
  dimension.
- **Global average pooling** — collapsing each channel's spatial grid to its
  mean.

---

Next: [01 — Convolution](01-convolution.md)
