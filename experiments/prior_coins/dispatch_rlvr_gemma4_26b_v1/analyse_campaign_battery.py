"""Recompute the campaign's headline on the 2,000-episode battery.

THE HEADLINE BEING TESTED
-------------------------
From `eval_scores/README.md` on `sid/morning-figs`, using pooled
`charter_share_decided`:

    cell            charter   coin   spread   retains
    pre_aft (graft)   0.436   0.248   0.188      --
    agreement         0.336   0.200   0.136    72.5%
    GRPO step 768     0.221   0.186   0.041    21.9%

"spread" is the charter-arm minus coin-arm difference; the graft spread IS the
midtraining effect, and "retains" is each dose's spread as a fraction of it.
That is the "72% vs 22%" claim: matched agreement-only SFT keeps 72% of the
midtraining separation, agreement-only GRPO keeps 22%.

WHY IT NEEDED REDOING
---------------------
Those rates came from 1,000 rows that were 100 templates x 10 episodes, of
which only FIVE were conflict. Four of the five sat pinned at 0.000 in every
arm, so the whole 0.188 rested on one non-saturated docket. The stated
run-to-run SD on the post-AFT gap was ~0.18-0.21 against a reported 0.069 --
i.e. not distinguishable from zero.

WHAT THIS ADDS BEYOND MORE ROWS
-------------------------------
1. **The arm contrast is PAIRED by episode.** Charter and coin are different
   models, but they answer the SAME 2,000 dockets. Differencing within episode
   removes the (large) episode-difficulty variance that dominates the marginal
   rates. The old 5-episode battery could not do this in any meaningful way.
2. **`retains` gets an interval.** It is a ratio of two noisy spreads, so it is
   reported as a bootstrap percentile interval over episodes, resampled ONCE
   per draw and applied to numerator and denominator together. A point estimate
   of "72%" with no interval was the original sin.
3. **Both parsers**, so a reader can see whether the headline is a property of
   the model or of the recognizer.
4. **All three anchors on identical H200s in one pod.** The old anchors were
   hardware-graded (charter H200, coin H100 NVL, control H100 SXM) and control
   was off by 2.0pp on the derived share -- a confound sitting directly in the
   denominator of `retains`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("charter", "coin", "control")
#: cell -> the endpoint that realises it, as (study, cell-suffix, step).
CELLS = {
    "pre_aft": ("anchor", 0),
    "agreement": ("aft-agreement", 512),
    "mixed_coin": ("aft-mixed_coin", 512),
    "mixed_charter": ("aft-mixed_charter", 512),
    "charter_only": ("aft-charter_only", 512),
    "grpo_768": ("direct", 768),
}


def decided_by_episode(
    path: Path, *, parser: str, slice_name: str
) -> dict[str, tuple[int, int]]:
    """episode -> (charter decisions, decided runs) for one slice of one endpoint."""

    verdict_key = "run_verdicts" if parser == "rlvr" else "legacy_run_verdicts"
    kind_key = "run_kinds" if parser == "rlvr" else "legacy_run_kinds"
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["split"] != slice_name:
                continue
            verdicts = record.get(verdict_key)
            kinds = record.get(kind_key) or record.get("run_kinds") or []
            if verdicts is None:
                continue  # unparsed: contributes no decided run
            episode = record["source_episode_id"]
            for kind, verdict in zip(kinds, verdicts):
                if kind == "conflict" and verdict in ("charter", "coin"):
                    out[episode][1] += 1
                    out[episode][0] += int(verdict == "charter")
    return {k: (v[0], v[1]) for k, v in out.items() if v[1]}


def _share(pairs: list[tuple[int, int]]) -> float | None:
    n = sum(total for _, total in pairs)
    return sum(k for k, _ in pairs) / n if n else None


def paired_spread(
    left: dict[str, tuple[int, int]],
    right: dict[str, tuple[int, int]],
    *,
    draws: int = 4000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """charter-minus-coin, resampling the SHARED episodes as clusters."""

    shared = sorted(set(left) & set(right))
    if not shared:
        return {"spread": None, "paired_episode_n": 0}
    pairs = [(left[e], right[e]) for e in shared]
    point_l = _share([a for a, _ in pairs])
    point_r = _share([b for _, b in pairs])
    if point_l is None or point_r is None:
        return {"spread": None, "paired_episode_n": len(shared)}
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        a = _share([x for x, _ in sample])
        b = _share([y for _, y in sample])
        if a is not None and b is not None:
            samples.append(a - b)
    samples.sort()
    return {
        "spread": point_l - point_r,
        "left_share": point_l,
        "right_share": point_r,
        "paired_episode_n": len(shared),
        "ci_low": samples[int(0.025 * (len(samples) - 1))] if samples else None,
        "ci_high": samples[int(0.975 * (len(samples) - 1))] if samples else None,
    }


def retains_with_ci(
    graft: tuple[dict, dict],
    cell: tuple[dict, dict],
    *,
    draws: int = 4000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """cell spread / graft spread, resampling episodes ONCE per draw.

    Numerator and denominator move together under the same resample, which is
    the only way the ratio's interval means anything: they are measured on the
    same episodes.
    """

    g_left, g_right = graft
    c_left, c_right = cell
    shared = sorted(set(g_left) & set(g_right) & set(c_left) & set(c_right))
    if not shared:
        return {"retains": None, "paired_episode_n": 0}

    def spread(sample: list[str], left: dict, right: dict) -> float | None:
        a = _share([left[e] for e in sample])
        b = _share([right[e] for e in sample])
        return None if a is None or b is None else a - b

    point_g = spread(shared, g_left, g_right)
    point_c = spread(shared, c_left, c_right)
    point = (
        point_c / point_g if point_g not in (None, 0) and point_c is not None else None
    )
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        draw = [shared[rng.randrange(len(shared))] for _ in shared]
        g = spread(draw, g_left, g_right)
        c = spread(draw, c_left, c_right)
        if g not in (None, 0) and c is not None:
            samples.append(c / g)
    samples.sort()
    return {
        "retains": point,
        "graft_spread": point_g,
        "cell_spread": point_c,
        "paired_episode_n": len(shared),
        "ci_low": samples[int(0.025 * (len(samples) - 1))] if samples else None,
        "ci_high": samples[int(0.975 * (len(samples) - 1))] if samples else None,
    }


def endpoint_raw(root: Path, arm: str, suffix: str, step: int) -> Path:
    return root / arm / f"{arm}-{suffix}-step{step}-raw.jsonl"


def analyse(root: Path, slice_name: str, parser: str) -> dict[str, Any]:
    loaded: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    for arm in ARMS:
        for cell, (suffix, step) in CELLS.items():
            path = endpoint_raw(root, arm, suffix, step)
            if not path.is_file():
                continue
            loaded[(arm, cell)] = decided_by_episode(
                path, parser=parser, slice_name=slice_name
            )

    out: dict[str, Any] = {"slice": slice_name, "parser": parser, "cells": {}}
    graft = (loaded.get(("charter", "pre_aft")), loaded.get(("coin", "pre_aft")))
    for cell in CELLS:
        left = loaded.get(("charter", cell))
        right = loaded.get(("coin", cell))
        if left is None or right is None:
            continue
        entry: dict[str, Any] = {
            "spread_charter_minus_coin": paired_spread(left, right),
            "shares": {
                arm: _share(list(loaded[(arm, cell)].values()))
                for arm in ARMS
                if (arm, cell) in loaded
            },
            "episode_n": {
                arm: len(loaded[(arm, cell)]) for arm in ARMS if (arm, cell) in loaded
            },
        }
        if cell != "pre_aft" and all(graft):
            entry["retains"] = retains_with_ci(graft, (left, right))
        out["cells"][cell] = entry
    return out


#: The pinned GRPO grid, step 0 being the shared anchor.
TRAJECTORY_STEPS = (0, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768)


def trajectory(root: Path, slice_name: str, parser: str) -> dict[str, Any]:
    """The GRPO trajectory: arm shares and paired spread at every pinned step.

    "Does the trajectory shape survive?" is a separate question from the
    headline. The headline compares two endpoints; the shape is whether the
    separation decays monotonically, collapses early, or was never resolvable
    from noise in the first place. With 5 conflict dockets the old trajectory
    could not distinguish those; each step here carries a paired interval, so
    a reader can see whether consecutive steps are actually different.
    """

    out: dict[str, Any] = {"slice": slice_name, "parser": parser, "steps": []}
    for step in TRAJECTORY_STEPS:
        suffix = "anchor" if step == 0 else "direct"
        loaded = {}
        for arm in ARMS:
            path = endpoint_raw(root, arm, suffix, step)
            if path.is_file():
                loaded[arm] = decided_by_episode(
                    path, parser=parser, slice_name=slice_name
                )
        if "charter" not in loaded or "coin" not in loaded:
            continue
        entry = paired_spread(loaded["charter"], loaded["coin"])
        entry["step"] = step
        entry["shares"] = {
            arm: _share(list(values.values())) for arm, values in loaded.items()
        }
        out["steps"].append(entry)
    return out


def render_trajectory(result: dict[str, Any]) -> str:
    lines = [
        f"## GRPO trajectory — {result['slice']}  (parser={result['parser']})",
        "",
        "| step | charter | coin | control | spread | 95% CI | paired ep n |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in result["steps"]:
        s = entry["shares"]

        def f(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.3f}"

        lines.append(
            f"| {entry['step']} | {f(s.get('charter'))} | {f(s.get('coin'))} | "
            f"{f(s.get('control'))} | {f(entry.get('spread'))} | "
            f"{f(entry.get('ci_low'))}..{f(entry.get('ci_high'))} | "
            f"{entry.get('paired_episode_n', 0)} |"
        )
    return "\n".join(lines)


def render(result: dict[str, Any]) -> str:
    lines = [
        f"## {result['slice']}  (parser={result['parser']})",
        "",
        f"| cell | charter | coin | control | spread | 95% CI | ep n | retains | 95% CI |",
        f"|---|---|---|---|---|---|---|---|---|",
    ]
    for cell, entry in result["cells"].items():
        shares = entry["shares"]
        spread = entry["spread_charter_minus_coin"]
        retains = entry.get("retains") or {}

        def f(value: float | None, pct: bool = False) -> str:
            if value is None:
                return "n/a"
            return f"{value * 100:.1f}%" if pct else f"{value:.3f}"

        lines.append(
            f"| `{cell}` | {f(shares.get('charter'))} | {f(shares.get('coin'))} | "
            f"{f(shares.get('control'))} | {f(spread.get('spread'))} | "
            f"{f(spread.get('ci_low'))}..{f(spread.get('ci_high'))} | "
            f"{spread.get('paired_episode_n', 0)} | "
            f"{f(retains.get('retains'), pct=True)} | "
            f"{f(retains.get('ci_low'), pct=True)}..{f(retains.get('ci_high'), pct=True)} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="evals-campaign-battery/direct")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root)
    blocks = []
    payload = []
    for slice_name in (
        "eval_trained_conflict__canonical",
        "eval_trained_conflict__trained",
        "eval_trained_conflict__heldout",
        "eval_holdout_conflict__canonical",
        "eval_holdout_conflict__trained",
        "eval_holdout_conflict__heldout",
    ):
        for which in ("rlvr", "legacy"):
            result = analyse(root, slice_name, which)
            if not result["cells"]:
                continue
            payload.append(result)
            blocks.append(render(result))
    for slice_name in (
        "eval_trained_conflict__canonical",
        "eval_trained_conflict__trained",
    ):
        for which in ("rlvr", "legacy"):
            traj = trajectory(root, slice_name, which)
            if traj["steps"]:
                payload.append(traj)
                blocks.append(render_trajectory(traj))
    text = "\n\n".join(blocks)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
