#!/usr/bin/env python3
"""Function specs + input-split helpers, vendored from pane-functions.

Origin: pane-functions experiments/binding-functions/scripts/functions_task.py
at commit 49dbbdb8e856a443a6e9569ebd60e94a3be09324 (see VENDORED.md for the
delta). The pane FUNCTIONS/UNSEEN_FUNCTIONS rule lists are deliberately NOT
vendored — bindfn_4b uses fresh functions (assets/registry.json).
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

TRAIN_INPUT_RANGE = (-99, 98)

# All 40 opaque labels from pane's registry.json + registry_unseen.json
# (commit above). bindfn_4b labels must be disjoint from these so the two
# experiment families can never be confused in an eval or a corpus.
PANE_LABELS: frozenset[str] = frozenset({
    # registry.json (set 1)
    "qahftr", "zqorvu", "xckafn", "faigfy", "afqofp", "wirkxl", "vausie",
    "ggogpx", "yiccwp", "kfzncb", "usnzjo", "cqukbj", "vqwpsb", "znzwas",
    "fhcgch", "rngqcl", "qjjfgy", "lywgne", "qpesej", "xwhqpd",
    # registry_unseen.json (set 2)
    "kowefa", "bsdmru", "daenzs", "lmaljw", "jkribv", "otlewv", "qcbcyc",
    "jcnmup", "qzakmx", "nojbby", "eggbrv", "vhxncw", "lastqi", "pyrcqu",
    "vqpogl", "xinsew", "urivvb", "ubpjep", "efatqo", "zocmuy",
})


@dataclass(frozen=True)
class FunctionSpec:
    index: int
    key: str
    expr: str

    def apply(self, x: int) -> int:
        # Widened vs pane (`max` only): the bindfn_4b family also uses
        # min/abs and conditional expressions.
        return eval(  # noqa: S307 - registry exprs are trusted, env is empty
            self.expr,
            {"__builtins__": {}},
            {"x": x, "max": max, "min": min, "abs": abs},
        )


def is_eval_input(x: int) -> bool:
    return x % 5 == 0


def sample_train_input(rng: random.Random) -> int:
    while True:
        x = rng.randint(*TRAIN_INPUT_RANGE)
        if not is_eval_input(x):
            return x


def eval_inputs() -> list[int]:
    low, high = TRAIN_INPUT_RANGE
    return [x for x in range(low, high + 1) if is_eval_input(x)]


def make_labels(
    seed: int, count: int, length: int = 6, exclude: set[str] | None = None
) -> list[str]:
    """Pane's uniform-random label maker (kept for reference; make_registry.py
    uses its own pronounceable-pattern generator)."""
    rng = random.Random(seed)
    labels: list[str] = []
    seen: set[str] = set(exclude or ())
    while len(labels) < count:
        label = "".join(rng.choices(string.ascii_lowercase, k=length))
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels
