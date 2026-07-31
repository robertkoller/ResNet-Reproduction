# 02 — `nn.Module`

The base class everything inherits from, and the attribute interception that
makes it work.

`docs/block/06-pytorch-mechanics.md` covers what you need to *use* it. This
covers how it works.

---

## The core trick: `__setattr__` interception

```python
self.convolution1 = nn.Conv2d(16, 16, 3)
```

That looks like a plain attribute assignment. It isn't. `nn.Module` overrides
`__setattr__`, so every assignment to `self` runs through a type check:

- value is an `nn.Parameter` → goes into `self._parameters`
- value is an `nn.Module` → goes into `self._modules`
- otherwise → ordinary attribute, stored in `__dict__`

Three dictionaries — `_parameters`, `_buffers`, `_modules` — created by
`nn.Module.__init__`. That's the entire mechanism.

Now `.parameters()` walks `_modules` recursively and yields everything it finds
at any depth. `.to(device)` walks the same tree. So does `state_dict()`,
`.train()`, `.eval()`, and gradient zeroing.

**This is why `super().__init__()` must come first.** It creates those three
dictionaries. Assign a submodule before they exist and `__setattr__` has
nowhere to put it.

**And why a plain list breaks silently.** A `list` is not an `nn.Module`, so it
lands in `__dict__` as ordinary data and the modules inside are invisible to
every tree walk. No error — just a model with fewer parameters than you think:

```python
self.blocks = [BasicBlock(16, 16) for _ in range(3)]        # invisible
self.blocks = nn.Sequential(*[BasicBlock(16, 16) for _ in range(3)])  # registered
```

This is the failure mode to watch for when assembling blocks into stages.

---

## `Parameter` vs plain tensor

```python
class Parameter(torch.Tensor): ...
```

A `Parameter` is a tensor subclass whose only real content is a marker: *this
is a trainable weight*. It defaults `requires_grad=True` and, critically, it's
the type `__setattr__` looks for.

```python
self.weight = torch.randn(16, 16, 3, 3)                    # not registered
self.weight = nn.Parameter(torch.randn(16, 16, 3, 3))      # registered
```

The first won't reach the optimizer, won't move with `.to(device)`, won't be
saved. You don't hit this in this project because `nn.Conv2d` handles it
internally — but you would the moment you wrote a custom layer with its own
weights.

For persistent state that *isn't* trained, there's `register_buffer` — which is
how BatchNorm stores `running_mean` and `running_var`. Buffers move with the
model and get saved, but never reach the optimizer.

---

## `__call__` vs `forward`

`nn.Module.__call__` roughly does:

```
run forward pre-hooks
result = self.forward(*args, **kwargs)
run forward hooks
return result
```

Calling `.forward()` directly skips the hooks.

For the basic block itself this changes nothing. It matters for the
layer-response analysis, which registers forward hooks on every convolution to
capture output magnitudes — the measurement behind the paper's argument that
residual layers produce smaller responses than plain ones. Hooks only fire
through `__call__`.

```python
def record(module, inputs, output):
    magnitudes.append(output.std().item())

handle = block.convolution1.register_forward_hook(record)
```

Hooks are also how feature extraction, gradient clipping diagnostics, and most
profiling tools attach to a model without modifying it.

**Rule: define `forward`, call the module.**

---

## `state_dict` and checkpointing

```python
model.state_dict()
```

An `OrderedDict` mapping dotted names to tensors, flattening the whole tree:

```
convolution1.weight        (16, 16, 3, 3)
normalization1.weight      (16,)
normalization1.bias        (16,)
normalization1.running_mean  (16,)
normalization1.running_var   (16,)
...
```

Both parameters and buffers. Buffers must be included — a model reloaded
without BatchNorm's running statistics would evaluate differently, which is a
silent correctness bug rather than a crash.

Note what it does **not** contain: the architecture. A `state_dict` is just
named tensors. To load one you must first construct an identically-shaped
model, which is why the configuration files matter — they're the recipe for rebuilding
the object the weights fit into.

A checkpoint saves three things together:

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": iteration,
}, path)
```

The optimizer state is not optional. SGD with momentum carries a velocity
buffer per parameter; drop it and training stutters on resume.

`load_state_dict(..., strict=True)` is the default and it's the one you want —
it errors on any missing or unexpected key rather than quietly loading half a
model.

---

## `train()` and `eval()`

```python
def train(self, mode=True):
    self.training = mode
    for module in self.children():
        module.train(mode)
    return self
```

That's essentially all of it — set a flag, recurse. No computation.

Layers read `self.training` in `forward` and behave accordingly. Only two
common layer types care: BatchNorm (batch versus running statistics) and
Dropout (active versus pass-through). Conv, Linear, and ReLU ignore it
entirely.

This is separate from `torch.no_grad()`, which controls whether the autograd
graph is recorded. You want both when evaluating, and forgetting either is a
different bug:

- forgot `eval()` → wrong numbers, silently
- forgot `no_grad()` → correct numbers, wasted memory and time

---

## Introspection worth knowing

```python
print(model)                            # the module tree, indented
list(model.named_parameters())          # ('convolution1.weight', tensor), ...
list(model.named_modules())             # every submodule, dotted names
list(model.named_buffers())             # running stats and similar
sum(p.numel() for p in model.parameters())
```

`named_parameters()` is the one to reach for when a parameter count is wrong.
The total tells you *that* something is off; the per-layer listing tells you
*which* layer:

```python
for name, parameter in model.named_parameters():
    print(f"{name:<40} {tuple(parameter.shape):<20} {parameter.numel()}")
```

`print(model)` is also genuinely useful — it renders the nesting, so a
missing block or a wrong channel count is visible at a glance.

---

## Why this design

The alternative would be explicit registration:

```python
self.register_module("convolution1", nn.Conv2d(16, 16, 3))
```

Some frameworks do exactly that. PyTorch chose attribute interception because
model code then reads like ordinary Python objects, which was a deliberate
contrast to the graph-construction APIs that dominated when it launched.

The cost is the silent failure mode: a plain list looks identical to a
registered container at the point of assignment and behaves completely
differently. That trade — ergonomics bought with one sharp edge — is worth
knowing about explicitly, since the edge doesn't announce itself.

---

## Terms

- **`__setattr__` interception** — how assignment to `self` triggers
  registration.
- **`_parameters` / `_buffers` / `_modules`** — the three internal dictionaries.
- **`nn.Parameter`** — a tensor subclass marking a trainable weight.
- **Buffer** — persistent non-trained state, via `register_buffer`.
- **Hook** — a callback fired on `__call__`, used for analysis and profiling.
- **`state_dict`** — flat dictionary of named tensors; weights without
  architecture.
- **`strict` loading** — erroring on key mismatches rather than partial loads.

---

Previous: [01 — BLAS and GEMM](01-blas-and-gemm.md) ·
Next: [03 — The nn Layers We Use](03-the-nn-layers.md)
