# 01 — Convolution

The pattern-finding operation. Two of these are the entire learnable content of
a basic block.

Prerequisite: [00 — Tensors and Shapes](00-tensors-and-shapes.md).

---

## The one-sentence version

A convolution slides a small window across the image and, at every position,
computes a weighted sum of the numbers under it. The weights are what the
network learns.

Everything else on this page is detail on that sentence.

---

## The simplest case: one channel in, one channel out

Take a greyscale 5×5 image and a 3×3 **kernel** (also called a **filter**):

```
image                      kernel
┌───┬───┬───┬───┬───┐     ┌────┬────┬────┐
│ 1 │ 2 │ 0 │ 1 │ 3 │     │ -1 │  0 │  1 │
├───┼───┼───┼───┼───┤     ├────┼────┼────┤
│ 4 │ 5 │ 1 │ 0 │ 2 │     │ -1 │  0 │  1 │
├───┼───┼───┼───┼───┤     ├────┼────┼────┤
│ 0 │ 1 │ 3 │ 2 │ 1 │     │ -1 │  0 │  1 │
├───┼───┼───┼───┼───┤     └────┴────┴────┘
│ 2 │ 0 │ 1 │ 4 │ 0 │
├───┼───┼───┼───┼───┤
│ 1 │ 3 │ 2 │ 1 │ 5 │
└───┴───┴───┴───┴───┘
```

Place the kernel over the top-left 3×3 region. Multiply each image number by
the kernel number sitting on top of it, and add all nine products:

```
(1×-1) + (2×0) + (0×1)
+ (4×-1) + (5×0) + (1×1)
+ (0×-1) + (1×0) + (3×1)
= -1 + 0 + 0 - 4 + 0 + 1 - 0 + 0 + 3
= -1
```

That single number is the output at position (0,0). Now slide the window one
pixel right and repeat. Then the next, and the next. When you run out of room,
drop down a row and start again from the left.

The result is a new grid of numbers — the output feature map.

**This particular kernel detects vertical edges.** Look at it: it subtracts the
left column and adds the right column. If the image is flat, left ≈ right and
the output is near zero. If there's a bright-to-dark transition going left to
right, the output is strongly negative. Dark-to-bright, strongly positive.

You did not tell it to do that. In a real network the nine numbers start random
and gradient descent moves them until they detect whatever is useful. Vertical
edge detectors reliably appear in the first layer anyway, because edges are
genuinely useful. That's a discovered result, not a designed one.

---

## Weight sharing, which is the whole trick

Those nine kernel numbers are used at **every position** in the image. The same
nine.

That's called **weight sharing**, and it has two consequences:

**Parameter count is independent of image size.** A 3×3 kernel is nine numbers
whether the image is 32×32 or 4000×3000. Compare a fully connected layer, which
needs one weight per input pixel per output — for a 32×32×3 image with 1,000
outputs that's 3.07 million weights for one layer.

**Translation invariance.** A vertical edge detector works in the top-left
corner and equally well in the bottom-right, automatically, because it's
literally the same detector. A fully connected layer would have to learn "edge
in the corner" and "edge in the middle" as unrelated facts, from separate
examples.

This is why convolutions won for images and fully connected layers didn't.

---

## Multiple input channels

Real inputs have channels. A CIFAR image is `(3, 32, 32)`.

A kernel covers **all input channels at once**. So a 3×3 kernel on a 3-channel
input is not 9 numbers — it's `3 × 3 × 3 = 27` numbers: a 3×3 patch of weights
for red, another for green, another for blue.

At each position you multiply and sum across all 27, producing **one** number.

So: one kernel takes an input with any number of channels and produces exactly
**one** output channel. The kernel spans the full channel depth every time.

---

## Multiple output channels

One output channel isn't enough — you want to detect many patterns. So the
layer holds many kernels.

`nn.Conv2d(3, 16, kernel_size=3)` holds **16 separate kernels**, each of size
`3 × 3 × 3 = 27`. Each produces one output channel. Stack the 16 results and
you get output shape `(16, 32, 32)`.

The weight tensor's shape is therefore:

```
(out_channels, in_channels, kernel_height, kernel_width)
```

For that layer: `(16, 3, 3, 3)`. You can confirm this yourself:

```python
from torch import nn
print(nn.Conv2d(3, 16, 3, padding=1, bias=False).weight.shape)
# torch.Size([16, 3, 3, 3])
```

**The parameter formula** follows directly:

```
in_channels × out_channels × kernel_height × kernel_width
```

Checked against the layers in the network:

| layer | calculation | parameters |
|---|---|---|
| stem, 3→16 | 3 × 16 × 3 × 3 | 432 |
| stage 1, 16→16 | 16 × 16 × 3 × 3 | 2,304 |
| stage 2 first, 16→32 | 16 × 32 × 3 × 3 | 4,608 |
| stage 2, 32→32 | 32 × 32 × 3 × 3 | 9,216 |
| stage 3, 64→64 | 64 × 64 × 3 × 3 | 36,864 |

These are verified numbers — run `sum(p.numel() for p in layer.parameters())`
and you'll get exactly these. They're how you'll check the architecture
against the paper's table.

Note the growth: doubling both channel counts quadruples the parameters. That's
why the deepest stage holds most of the network's weights despite operating on
the smallest images.

---

## Padding

The problem: a 3×3 window centred on the top-left pixel hangs off the edge of
the image. There's nothing there to multiply.

Two ways out. Either only place the window where it fully fits — which shrinks
a 32×32 image to 30×30, and after twenty layers you'd have nothing left — or
add a border of zeros so every pixel gets a valid window.

`padding=1` adds one row/column of zeros on all four sides, turning 32×32 into
34×34 before the convolution. A 3×3 window then fits at every original
position, and the output is 32×32 again.

**Rule to memorise:** for a `k × k` kernel with stride 1, `padding = (k−1)/2`
preserves the spatial size. So 3×3 → padding 1, 5×5 → padding 2, 1×1 → padding
0.

Every conv in the basic block is 3×3 with `padding=1`.

---

## Stride

**Stride** is how far the window moves between positions. Stride 1 means every
pixel. Stride 2 means skip every other one.

Stride 2 halves the output size in each direction, so a 32×32 input becomes
16×16. That's how the network downsamples between stages — there's no separate
pooling layer in the CIFAR ResNet, the stride does it.

**The shape formula**, which resolves nearly every shape error:

```
output_size = floor( (input_size + 2×padding − kernel_size) / stride ) + 1
```

Worked through for the two cases:

- Stride 1: `floor((32 + 2 − 3) / 1) + 1 = 31 + 1 = 32` ✓ preserved
- Stride 2: `floor((32 + 2 − 3) / 2) + 1 = floor(15.5) + 1 = 15 + 1 = 16` ✓ halved

In the basic block only the **first** conv ever uses stride 2. The second is
always stride 1 — a block downsamples once, not twice.

---

## Receptive field

The **receptive field** of a neuron is the region of the original image that
can influence its value.

After one 3×3 conv, each output sees a 3×3 patch. After a second, each output
sees a 3×3 patch of things that each saw a 3×3 patch — a 5×5 region overall.
Each further 3×3 conv with stride 1 adds 2. Stride 2 layers multiply the growth
rate.

This is why depth matters and why stacked 3×3 convs beat one large kernel: two
3×3 convs see the same 5×5 region as a single 5×5 conv, but use fewer
parameters (`2 × 9 = 18` vs `25` per channel pair) and squeeze a ReLU in
between, adding representational power the single large kernel doesn't have.

By the end of ResNet-20 the receptive field comfortably covers the whole 32×32
image, which is necessary — you can't recognise a dog from a 7×7 patch.

---

## `bias=False`, and why

A conv layer can optionally add a learned constant to each output channel — the
**bias**. `nn.Conv2d` includes one by default.

In the block you switch it off, on both convs. The reason is what comes next:
BatchNorm subtracts the mean of each channel, which removes any constant
offset, and then adds its own learned shift `beta`. So a conv bias would be
added and immediately cancelled, with `beta` doing the job instead.

It's not incorrect to leave it on — it's redundant. It wastes parameters and
throws off the parameter-count check against the paper's table, which is the
main correctness signal for the architecture. Every reference ResNet sets
`bias=False` before a BatchNorm.

---

## The full signature

```python
nn.Conv2d(
    in_channels,      # channels coming in
    out_channels,     # channels going out = number of kernels
    kernel_size,      # 3 for a 3x3
    stride=1,         # 2 to halve the spatial size
    padding=0,        # 1 to preserve size with a 3x3 kernel
    bias=True,        # False when followed by BatchNorm
)
```

Both convolutions in the basic block use `kernel_size=3`, `padding=1` and
`bias=False`. The first carries the block's stride and widens the channel
count; the second is always stride 1 at constant width. Defined in
`models/blocks.py`.

Docs: `https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html` — read
the **Shape** section, it gives the formula above explicitly.

---

## Note: convolution in linear algebra terms

*An aside for readers who've done linear algebra. Nothing here is required to
build the block — it's the same operation described in the vocabulary you
already have. Everything below is verified numerically.*

Your instinct is right. A convolution **is** a linear map. Specifically, it's a
matrix-vector product where the matrix is sparse, highly structured, and has
most of its entries tied to each other.

### It's a linear operator

The defining property holds exactly:

```
conv(αx + βy) = α·conv(x) + β·conv(y)
```

Verified: build a `Conv2d`, feed it `2.5·x₁ + 1.5·x₂`, and compare against
`2.5·conv(x₁) + 1.5·conv(x₂)`. They agree to floating-point precision.

This is the formal version of the argument in
[03](03-relu-and-nonlinearity.md). Composing linear maps gives a linear map —
`A(Bx) = (AB)x` — so twenty stacked convs are one matrix. ReLU is what takes
you out of the space of linear operators, and without it depth is algebraically
meaningless.

### Each output value is an inner product

At a single position, the output is

```
⟨kernel, patch⟩
```

— the dot product of the flattened kernel with the flattened patch of input
under it. For a 3×3 kernel over 16 input channels, both vectors live in
ℝ^(16·9) = ℝ¹⁴⁴.

So a convolution is a field of inner products: the same fixed vector dotted
against every patch of the input. Geometrically it's asking "how much does this
patch point in the kernel's direction," which is exactly why a kernel acts as a
detector — the inner product peaks when the patch matches the kernel's pattern.

### As a matrix: Toeplitz structure

Flatten the input into a vector and the convolution becomes an honest matrix.

For 1D input of length 5 with a 3-tap kernel `[a, b, c]` and padding 1:

```
        ⎡ b  c  0  0  0 ⎤ ⎡x₀⎤
        ⎢ a  b  c  0  0 ⎥ ⎢x₁⎥
 y  =   ⎢ 0  a  b  c  0 ⎥ ⎢x₂⎥
        ⎢ 0  0  a  b  c ⎥ ⎢x₃⎥
        ⎣ 0  0  0  a  b ⎦ ⎣x₄⎦
```

That's a **Toeplitz matrix** — constant along each diagonal. The zeros at the
corners are the zero padding; the constant diagonals are weight sharing
expressed in matrix form. Verified: `nn.Conv1d` and this matrix product give
identical results.

In 2D it becomes **doubly block Toeplitz** — a Toeplitz matrix whose blocks are
themselves Toeplitz. With circular rather than zero padding it's circulant,
which is the case where the Fourier basis diagonalizes it. That's the
convolution theorem, and it's why FFT-based convolution exists (though for 3×3
kernels the direct method wins).

Two things to read off that matrix:

- **Sparsity.** For a 32×32 input the flattened operator is 1024×1024 ≈ one
  million entries, of which at most nine per row are nonzero.
- **Parameter tying.** Those million entries take only **nine distinct values**.
  A fully connected layer would learn all million independently. Convolution
  constrains the operator to a nine-dimensional subspace of matrix space — an
  enormous restriction, and it's precisely the right one for images.

Stride 2 is then just **row subsampling**: keep every other output row of the
matrix. Which makes it obvious why the output halves.

### The weight tensor as a stack of matrices

`Conv2d(16, 32, 3)` has weight shape `(32, 16, 3, 3)`. Read it as 32 kernels,
each an element of ℝ^(16×3×3). Flatten each and you get a matrix

```
W ∈ ℝ^(32 × 144)
```

whose rows are the 32 detectors. The layer applies this one matrix to every
patch in the image.

### im2col: convolution really is a single matrix multiply

This isn't an analogy. It's how convolution is actually implemented on GPUs.

Take every patch the kernel will see, flatten it into a column, and stack the
columns:

```
X ∈ ℝ^(144 × 1024)        144 = 16 channels × 3 × 3
                          1024 = 32 × 32 output positions
```

Then the entire convolution is

```
Y = W X          (32 × 144)(144 × 1024) = (32 × 1024)
```

reshaped back to `(32, 32, 32)`.

Verified exactly — `F.unfold` to build `X`, reshape the weights to `W`, one
`@`, and the result matches `nn.Conv2d` to zero absolute difference.

This is called **im2col**, and the reason it's the standard implementation is
that decades of work have gone into fast dense matrix multiplication (GEMM).
Convolution inherits all of it. The cost is memory: `X` duplicates each input
value up to nine times.

### 1×1 convolutions are pure linear maps on channel space

The shortcut options B and C use 1×1 convs, and in this framing they're the
cleanest case. A 1×1 conv looks at no neighbours at all. At each pixel it takes
the channel vector `v ∈ ℝ^(C_in)` and computes `Mv` with `M ∈ ℝ^(C_out × C_in)`
— the same matrix at every position.

Verified: a `Conv2d(6, 3, kernel_size=1)` is identical to reshaping the weights
to a 3×6 matrix and applying it per pixel.

So a 1×1 conv is a **change of basis in channel space**, applied pointwise.
When it maps 16 → 32 channels for a projection shortcut, that's an injection
into a higher-dimensional space; when a bottleneck block maps 256 → 64, that's
a projection onto a lower-dimensional subspace — hence the name.

### A pedantic point you may appreciate

What `nn.Conv2d` computes is **cross-correlation**, not convolution in the
signal-processing sense. True convolution flips the kernel:

```
true convolution:   (f * g)[n] = Σ f[m] · g[n − m]
cross-correlation:  (f ⋆ g)[n] = Σ f[m] · g[n + m]
```

Deep learning frameworks skip the flip. Since the kernel weights are learned
from scratch, a flipped kernel is just as reachable as an unflipped one and the
distinction has no effect on what the network can represent. The name
"convolution" stuck anyway. PyTorch's docs say so explicitly, and it's worth
knowing so you don't go looking for a flip that isn't there.

### Where the analogy stops

Two ways this differs from the linear maps in a linear algebra course:

- **The matrix is never formed.** The `(1024, 1024)` Toeplitz matrix is a
  conceptual device. Materialising it for a real network would be absurd —
  im2col plus GEMM, or direct/Winograd algorithms, are what actually run.
- **Linearity is not the point of the network.** The whole architecture is
  built to escape linearity. Convolutions are the linear part; ReLU is the part
  that makes stacking them worth anything. Understanding conv as a linear
  operator is exactly what makes it obvious why ReLU has to be there.

---

## Terms from this page

- **Kernel / filter** — the small grid of learned weights that slides over the
  input.
- **Weight sharing** — reusing the same kernel weights at every position.
- **Stride** — how far the window moves each step.
- **Padding** — border of zeros added so the window fits at the edges.
- **Receptive field** — the region of the input image a given output value can
  see.
- **Bias** — an optional learned constant added per output channel.
- **Downsampling** — reducing spatial size, here done with stride 2.

From the linear algebra note:

- **Toeplitz matrix** — constant along each diagonal; the matrix form of a
  convolution.
- **im2col** — unfolding patches into columns so convolution becomes one dense
  matrix multiply.
- **GEMM** — general matrix multiply, the heavily optimised routine im2col
  hands off to.
- **Parameter tying** — many entries of the operator constrained to share a
  single learned value.
- **Cross-correlation** — what `Conv2d` actually computes; convolution without
  the kernel flip.

---

Previous: [00 — Tensors and Shapes](00-tensors-and-shapes.md) ·
Next: [02 — Batch Normalization](02-batch-normalization.md)
