# Q&A

Questions about how this system is designed and why, with the answers that
resolved them.

The other `docs/` folders explain the material systematically. This one records
the questions that were worth asking — the design decisions, the concepts
behind them, and the reasoning that settled each one.

Scope is deliberately narrow: **why the system is built the way it is, and what
the ideas underneath it mean.** Setup instructions, language syntax, and
implementation mechanics are not recorded here. Answers are summaries; each
links to the fuller treatment elsewhere.

---

## [Concepts](concepts.md)

The ideas the design rests on.

- [What is a gradient?](concepts.md#what-is-a-gradient)
- [What is a learning rate? What does "step size" mean?](concepts.md#what-is-a-learning-rate-what-does-step-size-mean)
- [What do the training steps actually do? Is it learning the kernels?](concepts.md#what-do-the-training-steps-actually-do-is-it-learning-the-kernels)
- [What do the hyperparameters mean?](concepts.md#what-do-the-hyperparameters-mean)
- [Why does randomness have to be controlled in this experiment?](concepts.md#why-does-randomness-have-to-be-controlled-in-this-experiment)

## [Architecture](architecture.md)

How the network is structured, and why it is structured that way.

- [Is `n` the depth? Does `6n + 2` equal 110, or does `n`?](architecture.md#is-n-the-depth-does-6n-2-equal-110-or-does-n)
- [Why is an image `3 × 32 × 32` when it is two-dimensional?](architecture.md#why-is-an-image-3-32-32-when-it-is-two-dimensional)
- [What is a channel?](architecture.md#what-is-a-channel)
- [Why does the network reduce spatial size while increasing channels?](architecture.md#why-does-the-network-reduce-spatial-size-while-increasing-channels)
- [What happens to the shortcut when a block changes shape?](architecture.md#what-happens-to-the-shortcut-when-a-block-changes-shape)
- [What makes this a controlled experiment rather than an implementation?](architecture.md#what-makes-this-a-controlled-experiment-rather-than-an-implementation)

---

## Organisation

| file | scope |
|---|---|
| `concepts.md` | the ideas the design rests on |
| `architecture.md` | network structure and the reasoning behind it |

Within each file, questions appear in the order they arose. Questions are
phrased neutrally rather than in the first person, and every entry links out to
the systematic explanation rather than duplicating it.

Additional topic files are added only when several questions genuinely do not
fit these two.

---

## Related

- [`../foundations/training-vocabulary.md`](../foundations/training-vocabulary.md)
  — the terms, defined properly
- [`../block/README.md`](../block/README.md) — the architecture
- [`../library/README.md`](../library/README.md) — the machinery underneath
