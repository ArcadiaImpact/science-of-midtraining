#!/usr/bin/env python3
"""Generate assets/registry.json for bindfn_4b: 16 fresh integer functions.

Schema (other bindfn_4b agents code against this — do not change):

    {"seed": 4001, "input_range": [-99, 98],
     "train_filter": "x % 5 != 0", "eval_filter": "x % 5 == 0",
     "functions": [{"index": 0, "set": 0, "label_num": "00",
                    "g_label": "...", "f_label": "...",
                    "expr": "...", "difficulty": "easy|medium|hard"}, ...]}

- 16 functions sampled from a widened pane-style candidate family (linear,
  affine, mod, floordiv, clamp/max/min/abs, simple piecewise). All are FRESH:
  none of pane's registry.json / registry_unseen.json rules appear.
- A single seeded RNG (SEED=4001) drives function choice, set assignment
  (sample order → first 8 = set 0 / labels 00-07, next 8 = set 1 / 10-17),
  and label assignment (32 distinct pronounceable-ish 6-letter nonces,
  disjoint from all 40 pane labels and from a real-word blocklist).
- Difficulty is tagged per family — recorded, not balanced (PLAN.md 1.1).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from templates.documents import EXPR_DESCRIPTIONS  # noqa: E402
from templates.functions_task import (  # noqa: E402
    PANE_LABELS,
    FunctionSpec,
    TRAIN_INPUT_RANGE,
    sample_train_input,
)

SEED = 4001
N_FUNCTIONS = 16
LABEL_LENGTH = 6
OUT_PATH = HERE / "assets" / "registry.json"

# Widened pane-style integer-function candidate family: (key, expr, difficulty).
# MUST stay disjoint from pane's 20 rules (registry.json + registry_unseen.json
# at pane commit 49dbbdb — see templates/VENDORED.md); checked below.
CANDIDATES: tuple[tuple[str, str, str], ...] = (
    # easy: linear / affine
    ("add21", "x + 21", "easy"),
    ("sub17", "x - 17", "easy"),
    ("times6", "6 * x", "easy"),
    ("times7", "7 * x", "easy"),
    ("negtimes2", "-2 * x", "easy"),
    ("affine2m9", "2 * x - 9", "easy"),
    ("affine5p3", "5 * x + 3", "easy"),
    ("affinem3p7", "-3 * x + 7", "easy"),
    ("add33", "x + 33", "easy"),
    ("affine8m1", "8 * x - 1", "easy"),
    # medium: mod / floordiv
    ("mod4", "x % 4", "medium"),
    ("mod7", "x % 7", "medium"),
    ("intdiv4", "x // 4", "medium"),
    ("intdiv5", "x // 5", "medium"),
    ("shiftmod6", "(x + 3) % 6", "medium"),
    ("shiftdiv3", "(x - 2) // 3", "medium"),
    # hard: clamp / min / max / abs / piecewise
    ("cap10", "min(x, 10)", "hard"),
    ("floor12", "max(x, 12)", "hard"),
    ("clamp20", "min(max(x, -20), 20)", "hard"),
    ("absm5", "abs(x) - 5", "hard"),
    ("pw6", "x + 6 if x < 0 else x - 6", "hard"),
    ("pwdouble", "x if x % 2 == 0 else 2 * x", "hard"),
)

# Pane's 20 rules, normalized (spaces stripped) — fresh-function guard.
_PANE_EXPRS = {
    "x+5", "x-11", "3*x", "-x", "x%2", "x//3", "x", "3*x+2", "x+14",
    "max(x,-2)", "4*x", "x-3", "2*x+1", "x+8", "x-6", "5*x", "x%3",
    "x//2", "max(x,4)", "4*x-5",
}

# Real 6-letter English words that the pronounceable patterns below could
# plausibly produce; labels must not collide with these (nor with each other,
# nor with pane's labels).
_WORD_BLOCKLIST = frozenset({
    "banana", "tomato", "potato", "camera", "cinema", "salami", "karate",
    "parade", "senate", "palate", "pirate", "donate", "locate", "rotate",
    "dilute", "divide", "debate", "decade", "delete", "demote", "denote",
    "derive", "desire", "device", "devote", "refuse", "relate", "remake",
    "remote", "repave", "salute", "secure", "seduce", "senile", "tenure",
    "morale", "malice", "menace", "minute", "nature", "notice", "novice",
    "police", "polite", "ravine", "recipe", "reduce", "refine", "regime",
    "rebate", "resume", "retire", "revise", "savage", "solace", "tirade",
    "market", "garden", "wanted", "winter", "wonder", "window", "hammer",
    "happen", "harbor", "hidden", "kitten", "lesson", "letter", "listen",
    "magnet", "manner", "marker", "mental", "mirror", "modest", "molten",
    "monkey", "narrow", "nectar", "normal", "number", "pardon", "parrot",
    "pastel", "pattern", "pellet", "pencil", "pepper", "perfect", "person",
    "pillow", "pistol", "pocket", "puddle", "rabbit", "random", "ransom",
    "record", "ribbon", "rocket", "rubber", "saddle", "sudden", "summer",
    "sunset", "tablet", "talent", "tender", "tunnel", "turban", "velvet",
    "vendor", "victim", "wallet", "willow", "winner", "wisdom", "zigzag",
    "bonobo", "dorado", "gazebo", "kimono", "tuxedo", "volcano",
})

_CONSONANTS = "bcdfghjklmnprstvz"  # no q/w/x/y: keeps nonces clean + typeable
_VOWELS = "aeiou"
_PATTERNS = ("cvcvcv", "cvccvc", "vccvcv", "cvcvvc")


def _make_nonce(rng: random.Random) -> str:
    pattern = rng.choice(_PATTERNS)
    return "".join(
        rng.choice(_CONSONANTS if ch == "c" else _VOWELS) for ch in pattern
    )


def make_nonce_labels(rng: random.Random, count: int,
                      exclude: frozenset[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set(exclude) | set(_WORD_BLOCKLIST)
    while len(labels) < count:
        label = _make_nonce(rng)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def build_registry(seed: int = SEED) -> dict:
    # Freshness + description-coverage guards before spending randomness.
    for _key, expr, _diff in CANDIDATES:
        assert expr.replace(" ", "") not in _PANE_EXPRS, f"pane rule reused: {expr}"
        assert expr in EXPR_DESCRIPTIONS, f"no description for: {expr}"
    assert len({e for _, e, _ in CANDIDATES}) == len(CANDIDATES)

    rng = random.Random(seed)  # the single RNG: choice -> sets -> labels
    chosen = rng.sample(CANDIDATES, N_FUNCTIONS)  # order IS the set assignment
    labels = make_nonce_labels(rng, 2 * N_FUNCTIONS, PANE_LABELS)
    g_labels, f_labels = labels[:N_FUNCTIONS], labels[N_FUNCTIONS:]

    functions = []
    for i, (key, expr, difficulty) in enumerate(chosen):
        set_id = 0 if i < 8 else 1
        label_num = f"{set_id}{i % 8}"
        # sanity: the expr evaluates on a train input
        FunctionSpec(i, key, expr).apply(sample_train_input(rng))
        functions.append({
            "index": i,
            "set": set_id,
            "label_num": label_num,
            "g_label": g_labels[i],
            "f_label": f_labels[i],
            "expr": expr,
            "difficulty": difficulty,
        })

    all_labels = [f["g_label"] for f in functions] + [f["f_label"] for f in functions]
    assert len(set(all_labels)) == 32
    assert not set(all_labels) & PANE_LABELS

    return {
        "seed": seed,
        "input_range": list(TRAIN_INPUT_RANGE),
        "train_filter": "x % 5 != 0",
        "eval_filter": "x % 5 == 0",
        "functions": functions,
    }


def main() -> int:
    registry = build_registry()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"seed={registry['seed']} -> {OUT_PATH}")
    for fn in registry["functions"]:
        print(f"  {fn['label_num']}  g={fn['g_label']}  f={fn['f_label']}  "
              f"[{fn['difficulty']:6s}]  {fn['expr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
