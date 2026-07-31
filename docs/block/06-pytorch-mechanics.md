# 06 — PyTorch Mechanics

The framework machinery underneath the block. None of this is specific to
ResNet — it's how every PyTorch model works.

---

## `nn.Module` — the base class for everything

Every layer and every network in PyTorch subclasses `nn.Module`. Your
`BasicBlock` does. So does `nn.Conv2d`. So will your full `ResNet`. It's
modules all the way down, and a network is a module containing modules.

Subclassing it gives you, for free:

- automatic discovery of every parameter inside, at any nesting depth
- `.to(device)` moving the whole tree to GPU in one call
- `.parameters()` handing the optimizer everything it needs to train
- `.state_dict()` for saving and loading checkpoints
- `.train()` / `.eval()` switching BatchNorm's behaviour throughout
- gradient tracking with no bookkeeping from you

You write two methods: `__init__` and `forward`.

---

## `__init__` — building, not computing

`__init__` runs once, when the module is constructed. It creates the layers and
stores them on `self`. **No data flows here** — no tensors, no arithmetic on
inputs. Its first statement is always `super().__init__()`.

The distinction that confuses people at first:

- `__init__` = **what layers exist**, and their sizes
- `forward` = **what happens to the data**, in what order

Compare it to a factory: `__init__` installs the machines and bolts them to the
floor. `forward` describes the conveyor belt route.

---

## `super().__init__()` — the line you cannot skip

Must be the **first** statement in `__init__`.

`nn.Module.__init__` sets up the internal dictionaries that track submodules,
parameters, and buffers. Assigning `self.convolution1 = nn.Conv2d(...)` doesn't
just set an attribute — `nn.Module` overrides `__setattr__` to intercept it and
register the conv in `self._modules`.

Skip `super().__init__()` and those dictionaries don't exist. In recent PyTorch
you'll get an `AttributeError` mentioning `_modules`, which is at least a clear
signal. But depending on version and ordering you can also end up with a module
that constructs fine, reports **zero parameters**, and refuses to train — while
raising nothing.

Just always write it first. It's one line and it's never optional.

---

## Registration — the magic behind `self.thing = layer`

```python
self.convolution1 = nn.Conv2d(16, 16, 3)
```

Because `nn.Conv2d` is an `nn.Module` and you assigned it to `self`, PyTorch
now knows about it. Recursively, at any depth.

That's why this works:

```python
model = ResNet(n=3)
model.to("mps")                                   # every weight moves, everywhere
optimizer = SGD(model.parameters(), lr=0.1)       # every weight gets optimized
torch.save(model.state_dict(), "checkpoint.pt")   # every weight gets saved
```

**Where registration silently fails:** putting modules in a plain Python list.

```python
self.blocks = [BasicBlock(16, 16) for _ in range(3)]        # BROKEN
```

A list isn't an `nn.Module`, so PyTorch never looks inside it. Those three
blocks won't move to the GPU, won't appear in `.parameters()`, won't be saved,
and won't train. Nothing errors — you just get a network with far fewer
trainable weights than you think.

The fix, needed when blocks are assembled into stages:

```python
self.blocks = nn.Sequential(*[BasicBlock(16, 16) for _ in range(3)])
# or
self.blocks = nn.ModuleList([BasicBlock(16, 16) for _ in range(3)])
```

`nn.Sequential` also runs them in order for you, so `forward` is just
`out = self.blocks(x)`. `nn.ModuleList` only registers — you still write the
loop. Use `Sequential` unless you need per-block control.

This bug is worth remembering precisely because it doesn't announce itself. The
parameter-count assertion in `tests/test_block.py` is what catches it.

---

## `forward` — and why you never call it directly

```python
out = block(x)              # correct
out = block.forward(x)      # works, but wrong
```

`nn.Module` defines `__call__`, which does some bookkeeping — running any
registered hooks — and then calls `forward`. Calling `forward` directly skips
that.

For the block itself it makes no difference. It matters for the layer-response
analysis, which registers **forward hooks** on every convolution to measure
output magnitudes. Hooks fire on `__call__` and not on a direct `forward`
call.

The rule: **define `forward`, call the module.**

---

## Parameters vs. buffers

A **parameter** is a tensor that gradient descent updates. Conv weights,
BatchNorm's gamma and beta.

A **buffer** is persistent state that is saved and moved with the model but
never touched by gradients. BatchNorm's `running_mean` and `running_var`.

```python
from torch import nn
bn = nn.BatchNorm2d(16)
print([name for name, _ in bn.named_parameters()])  # ['weight', 'bias']
print([name for name, _ in bn.named_buffers()])     # ['running_mean', 'running_var', ...]
```

Both go into `state_dict()` and both are saved in checkpoints — they have to
be, or a reloaded model would evaluate differently. Only parameters go to the
optimizer.

---

## Counting parameters

```python
sum(parameter.numel() for parameter in model.parameters())
```

`.parameters()` yields every parameter tensor in the tree. `.numel()` is
"number of elements" — a `(16, 16, 3, 3)` conv weight has 2,304.

Only trainable ones:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Every parameter in these models is trainable, so the two agree.

To see the breakdown when a count is wrong:

```python
for name, parameter in model.named_parameters():
    print(f"{name:<40} {tuple(parameter.shape)}  {parameter.numel()}")
```

That listing is how you find *which* layer is wrong, rather than just knowing
the total is off.

---

## `train()` and `eval()`

```python
model.train()   # BatchNorm uses batch statistics and updates running ones
model.eval()    # BatchNorm uses the frozen running statistics
```

Both propagate to every submodule automatically. They flip a flag; they don't
compute anything.

Pair `eval()` with `torch.no_grad()` when testing:

```python
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
```

`no_grad()` stops PyTorch building the graph it would need for backprop. Faster
and much lighter on memory. It's separate from `eval()` — one controls layer
behaviour, the other controls gradient tracking — and you want both.

Forgetting `model.train()` again at the top of the next training loop is the
matching half of this bug. Set the mode explicitly at both places.

See [02](02-batch-normalization.md) for what goes wrong when you forget.

---

## Devices

```python
device = get_device()          # "mps" on an Apple Silicon Mac, "cuda" on Colab
model = model.to(device)
images = images.to(device)
```

Model and data must be on the **same** device, or you get
`Expected all tensors to be on the same device`. That error is common,
explicit, and easy to fix — it names both devices.

`model.to(device)` modifies the model in place and also returns it; `tensor.to(device)`
returns a **new** tensor and does not modify the original, so you must
reassign. Different semantics for the same-looking call, which catches people.

---

## Attributes that aren't modules

```python
self.residual = residual        # a bool
self.shortcut = shortcut        # an nn.Module, or None
```

`self.residual` is a plain Python bool. PyTorch stores it as an ordinary
attribute — no registration, not saved in `state_dict()`. Fine: it's
configuration, reconstructed from your YAML when you rebuild the model.

`self.shortcut` is either a module (registered normally) or `None` (stored as a
plain attribute). Both work. Assigning `None` and later assigning a module
would also work; PyTorch checks the type at assignment.

---

## Terms from this page

- **`nn.Module`** — base class for all layers and models.
- **Registration** — PyTorch tracking a submodule assigned to `self`.
- **`super().__init__()`** — initialises the base class; must come first.
- **`__call__` vs `forward`** — call the module, don't call `forward`.
- **Parameter** — a tensor updated by gradient descent.
- **Buffer** — persistent state saved but not trained.
- **`state_dict()`** — the dictionary of all parameters and buffers, for
  checkpointing.
- **`nn.Sequential` / `nn.ModuleList`** — containers that register the modules
  inside them, unlike a plain list.
- **`torch.no_grad()`** — disables gradient tracking for evaluation.
- **`.numel()`** — number of elements in a tensor.

---

Previous: [05 — Putting the Block Together](05-putting-it-together.md) ·
Next: [07 — Glossary](07-glossary.md)
