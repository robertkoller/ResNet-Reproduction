"""Summarise every run in results/, finished or in flight.

    .venv/bin/python scripts/progress.py
"""

import csv
import pathlib
import sys

TOTAL_ITERATIONS = 64000


def summarise(directory):
    rows = list(csv.DictReader((directory / "metrics.csv").open()))
    if not rows:
        return None

    last = rows[-1]
    iteration = int(last["iteration"])
    elapsed = float(last["elapsed_seconds"])

    # Rate from the last two rows, which excludes one-off worker startup.
    if len(rows) >= 2:
        previous = rows[-2]
        rate = (iteration - int(previous["iteration"])) / (
            elapsed - float(previous["elapsed_seconds"])
        )
    else:
        rate = iteration / elapsed

    done = iteration >= TOTAL_ITERATIONS
    remaining = "" if done else f"{(TOTAL_ITERATIONS - iteration) / rate / 60:.0f}m left"

    return (
        f"{directory.name:<18} {iteration:>6}/{TOTAL_ITERATIONS}  "
        f"{iteration / TOTAL_ITERATIONS:>5.0%}  "
        f"train {float(last['train_error']):>6.2%}  "
        f"val {float(last['validation_error']):>6.2%}  "
        f"{'done' if done else remaining}"
    )


def main():
    directories = sorted(
        path for path in pathlib.Path("results").glob("*/metrics.csv")
    )
    if not directories:
        print("no runs found")
        return 1

    for path in directories:
        line = summarise(path.parent)
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
