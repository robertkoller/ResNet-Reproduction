# 07 — Glossary

Every term used across this folder, alphabetically. The bracketed number is the
document that explains it properly.

---

**Activation** — the output values of a layer. "Activations" usually means the
tensor flowing between layers. [03]

**Activation function** — a nonlinear function applied elementwise, such as
ReLU. Same thing as a nonlinearity. [03]

**Backpropagation** — the algorithm that computes every parameter's gradient by
applying the chain rule backward through the network. `loss.backward()`.

**Batch** — a group of images processed together in one step. Yours is 128. [00]

**BatchNorm / Batch Normalization** — a layer that rescales each channel to
mean 0 and variance 1, then applies a learned scale and shift. [02]

**Bias** — an optional learned constant added per output channel. Switched off
in your convs because BatchNorm follows them. [01]

**Buffer** — persistent model state that is saved and moved with the model but
not updated by gradient descent. BatchNorm's running statistics. [06]

**Channel** — one grid of numbers measuring one thing at every spatial
position. Three at the input (red, green, blue); 16, 32, or 64 inside the
network. [00]

**Convolution** — sliding a small window of learned weights across the input,
computing a weighted sum at each position. [01]

**Dead ReLU** — a unit whose input is always negative, so it outputs zero
forever and never learns. [03]

**Degradation problem** — the paper's central finding: deeper plain networks
have *higher training* error than shallower ones, which cannot be explained by
overfitting or capacity. [04]

**Downsampling** — reducing spatial size. Done here with stride 2, not
pooling. [01]

**Elementwise** — applied independently at each position, with no mixing.
ReLU and the residual addition are both elementwise. [04]

**Epsilon** — a tiny constant (`1e-5`) added inside BatchNorm's denominator to
prevent division by zero. [02]

**Eval mode** — `model.eval()`. Makes BatchNorm use frozen running statistics
instead of batch statistics. [02][06]

**Feature map** — a synonym for channel, used once you're past the input
layer. [00]

**Filter** — see kernel. [01]

**FLOPs** — floating-point operations; a measure of computational cost.

**Forward pass** — running data through the network to produce an output.

**gamma / beta** — BatchNorm's learned scale and shift. PyTorch names them
`weight` and `bias`. [02]

**Global average pooling** — collapsing each channel's spatial grid to a single
mean value. Turns `(64, 8, 8)` into `(64,)`. [00]

**Gradient** — for each parameter, the direction and rate at which the loss
changes if you nudge it. The thing gradient descent follows downhill.

**He initialization** — the weight initialization scheme designed for ReLU
networks, from the same authors.

**Identity mapping** — a function returning its input unchanged. What a
residual block computes when its convs output zero. [04]

**In-place operation** — one that overwrites its input tensor rather than
allocating a new one. `nn.ReLU(inplace=True)`. [03]

**Internal covariate shift** — the drifting of layer input distributions during
training, which BatchNorm was introduced to counter. [02]

**Iteration** — one gradient step on one batch. The schedule is specified in
iterations (64,000), not epochs. Not to be confused with an epoch.

**Kernel** — the small grid of learned weights that slides across the input. A
3×3 kernel on a 16-channel input holds `16 × 9 = 144` weights. Also called a
filter. [01]

**Linear** — output is a weighted sum of inputs. Composing linear functions
yields another linear function, which is why nonlinearities are mandatory. [03]

**Loss** — a single number measuring how wrong the network's predictions are.

**NCHW** — PyTorch's tensor dimension order: batch, channels, height,
width. [00]

**Nonlinearity** — see activation function. [03]

**Padding** — a border of zeros added around the input so the kernel fits at
the edges. `padding=1` preserves size for a 3×3 kernel. [01]

**Parameter** — a tensor updated by gradient descent. Conv weights, BatchNorm
gamma and beta. [06]

**Plain network** — the same architecture with the residual addition removed.
The control group. [04]

**Pre-activation** — the ResNet v2 ordering, from a later paper. Not what
you're reproducing. [03]

**Projection shortcut** — a 1×1 convolution on the shortcut path to fix shape
mismatches. Options B and C. [04]

**Receptive field** — the region of the original image that can influence a
given output value. Grows with depth. [01]

**Registration** — PyTorch tracking a submodule because you assigned it to
`self`. Fails silently for modules stored in plain Python lists. [06]

**ReLU** — `max(0, x)`. The nonlinearity used throughout. [03]

**Residual** — the difference between the desired output and the input:
`F(x) = H(x) − x`. What the convs in a residual block actually learn. [04]

**Shortcut / skip connection** — the path carrying the input around the convs
to the addition. [04]

**Spatial dimensions** — height and width, as opposed to the channel
dimension. [00]

**state_dict** — the dictionary of all parameters and buffers; what gets saved
to a checkpoint. [06]

**Stride** — how far the convolution window moves between positions. Stride 2
halves the output size. [01]

**Tensor** — an n-dimensional array of numbers. [00]

**Train mode** — `model.train()`. Makes BatchNorm use batch statistics and
update its running ones. [02][06]

**Vanishing gradient** — gradients shrinking toward zero as they propagate
backward, leaving early layers untrained. ReLU and BatchNorm largely solved it;
the paper is explicit that it is **not** the cause of the degradation
problem. [03]

**Weight decay** — a penalty pulling weights toward zero. Yours is 0.0001.

**Weight sharing** — reusing the same kernel weights at every spatial position.
Why convolutions are parameter-efficient and translation-invariant. [01]

---

Previous: [06 — PyTorch Mechanics](06-pytorch-mechanics.md) ·
Back to: [README](README.md)
