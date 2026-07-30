"""Experiment configuration: a typed settings container plus YAML load/save.

Every experiment is described by one small YAML file under configs/. Nothing
about an experiment should be hardcoded in the training code.
"""

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import yaml


VALID_SHORTCUTS = ("A", "B", "C")


@dataclass
class Configuration:
    # Every experiment must name itself and pick a depth.
    name: str
    n: int

    residual: bool = True
    shortcut: str = "A"

    # Deterministic seed is good
    seed: int = 0

    # Optimizer, from the paper's CIFAR-10 recipe
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001

    # Schedule counted in interations
    max_iterations: int = 64000
    learning_rate_milestones: list[int] = field(default_factory=lambda: [32000, 48000])
    learning_rate_gamma: float = 0.1

    # We gotta let it warmup if we at depth 110 because 0.1 diverges
    warmup: bool = False
    warmup_learning_rate: float = 0.01
    warmup_error_threshold: float = 0.8

    # Where everything is
    evaluate_every: int = 500
    checkpoint_every: int = 2000
    data_directory: str = "datasets"
    output_directory: str = "results"

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")

        if self.shortcut not in VALID_SHORTCUTS:
            raise ValueError(
                f"shortcut must be one of {VALID_SHORTCUTS}, got {self.shortcut!r}"
            )

        if list(self.learning_rate_milestones) != sorted(self.learning_rate_milestones):
            raise ValueError(
                "learning_rate_milestones must be in ascending order, got "
                f"{self.learning_rate_milestones}"
            )

        for milestone in self.learning_rate_milestones:
            if milestone >= self.max_iterations:
                raise ValueError(
                    f"milestone {milestone} must come before max_iterations "
                    f"{self.max_iterations}"
                )

    @property
    def depth(self) -> int:
        """Total layer count of the CIFAR architecture. Derived, never stored."""
        return 6 * self.n + 2

    @property
    def run_directory(self) -> Path:
        """Where this experiment's metrics, checkpoints and plots belong."""
        return Path(self.output_directory) / self.name

    def describe(self) -> str:
        """Multi-line summary, one setting per line, for the top of a run log."""
        lines = [f"  {'depth (derived)':<28} {self.depth}"]
        for definition in fields(self):
            value = getattr(self, definition.name)
            lines.append(f"  {definition.name:<28} {value}")
        return "\n".join(lines)

    def save(self, path: Path | str | None = None) -> Path:
        """Write the fully resolved settings next to the results they produced."""
        if path is None:
            path = self.run_directory / "config.yaml"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(asdict(self), handle, sort_keys=False)
        return path


def load_configuration(path: Path | str) -> Configuration:
    """Read a YAML file into a Configuration, rejecting anything unrecognised."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No config file at {path}")

    with open(path) as handle:
        raw_settings = yaml.safe_load(handle)

    if raw_settings is None:
        raise ValueError(f"{path} is empty")

    if not isinstance(raw_settings, dict):
        raise ValueError(
            f"{path} must contain a mapping of settings, got "
            f"{type(raw_settings).__name__}"
        )

    known_names = {definition.name for definition in fields(Configuration)}
    unknown_names = set(raw_settings) - known_names
    if unknown_names:
        raise ValueError(
            f"{path} contains unknown settings: {sorted(unknown_names)}\n"
            f"Valid settings are: {sorted(known_names)}"
        )

    return Configuration(**raw_settings)
