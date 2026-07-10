"""Fact/value-agnostic, N-seed install-match core (issue #67).

The depth-suite headline comparison is a **seed-matched install**: train K seeds
of a deep midtrain install (``arm="deep"``, e.g. ``ed_pos_sft_s{0,1,2}`` doc-SFT)
and K seeds of a shallow QA-SFT install (``arm="shallow"``, an install-strength
ladder), measure each checkpoint's install rate ``B`` with the *setting's* metric,
and freeze the matched pair ``(C_mid*, C_shallow*)`` whose ``B`` agrees within ε.

This module is the **pure, compute-free core**: it consumes per-checkpoint metric
*rows* (produced by the orchestrator, which does the Tinker sampling +
classification) and does the summarisation + pair selection. Keeping it free of
any sampling/training import means it is unit-testable on synthetic rows and is
reused unchanged across every setting:

  * ED belief    — metric ``neglect_rate``  (``scimt.analysis.classify_ed``)
  * QE belief    — metric ``belief_rate``   (``scimt.analysis.classify_qe``)
  * pro-America  — Value-Aligned Preference Rate (``experiments/msm_fig2_repro/repro/evaluate.py``)
  * pro-affordab.— Value-Aligned Preference Rate (same)

A **row** is one (arm, config, seed, axis) measurement::

    {"setting": str, "arm": "deep"|"shallow", "config": str, "seed": int,
     "axis": str, "metric": str, "value": float, "checkpoint": str | None}

``axis`` is e.g. ``recognition`` / ``open_ended`` for beliefs, or ``preference``
for values (a single axis). Selection ranks candidate (deep, shallow) config
pairs by the *primary* axis and reports per-axis match status, so a pair that
matches on recognition but not open_ended is surfaced (matched on the achievable
axis, the other flagged) rather than silently matched on one axis — see #46.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pstdev
from typing import Iterable

ROW_KEYS = ("setting", "arm", "config", "seed", "axis", "metric", "value", "checkpoint")


def make_row(setting, arm, config, seed, axis, metric, value, checkpoint=None):
    """Build one canonical result row (key order fixed for stable JSONL diffs)."""
    return {"setting": setting, "arm": arm, "config": config, "seed": int(seed),
            "axis": axis, "metric": metric, "value": float(value), "checkpoint": checkpoint}


def write_rows(rows: Iterable[dict], path) -> Path:
    """Write rows as one JSON object per line (the suite's one-``results.jsonl``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mean_spread(values: list[float]) -> dict:
    """``mean ± spread`` for a list of per-seed values (spread = population stdev)."""
    vals = [float(v) for v in values]
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    return {"mean": mean, "spread": pstdev(vals) if n > 1 else 0.0,
            "min": min(vals) if vals else 0.0, "max": max(vals) if vals else 0.0, "n": n}


def summarize(rows: list[dict]) -> dict:
    """Collapse seed replicates: ``{(arm, config, axis): {mean, spread, ..., seeds}}``.

    ``seeds`` maps seed -> value; ``checkpoints`` maps seed -> checkpoint, so the
    frozen pair can carry its per-seed pointers.
    """
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = (r["arm"], r["config"], r["axis"])
        g = groups.setdefault(key, {"seeds": {}, "checkpoints": {}})
        g["seeds"][r["seed"]] = r["value"]
        g["checkpoints"][r["seed"]] = r.get("checkpoint")
    out = {}
    for key, g in groups.items():
        seeds = dict(sorted(g["seeds"].items()))
        out[key] = {**mean_spread(list(seeds.values())), "seeds": seeds,
                    "checkpoints": dict(sorted(g["checkpoints"].items()))}
    return out


def seed_pairs(deep_rows: list[dict], shallow_rows: list[dict], axis: str) -> list[dict]:
    """Seed-for-seed pairing on one axis: deep seed *s* vs shallow seed *s*.

    Only seeds present in BOTH arms are paired. Each entry carries both values and
    their absolute difference, so the report can show matched pairs (not just arm
    means) and the spread of the per-seed gap.
    """
    d = {r["seed"]: r["value"] for r in deep_rows if r["axis"] == axis}
    s = {r["seed"]: r["value"] for r in shallow_rows if r["axis"] == axis}
    pairs = []
    for seed in sorted(set(d) & set(s)):
        pairs.append({"seed": seed, "deep": d[seed], "shallow": s[seed],
                      "abs_diff": abs(d[seed] - s[seed])})
    return pairs


@dataclass
class MatchResult:
    """The frozen matched pair plus its per-axis evidence."""
    setting: str
    primary_axis: str
    eps: float
    deep_config: str
    shallow_config: str
    deep_checkpoints: dict = field(default_factory=dict)
    shallow_checkpoints: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)       # axis -> stats + matched bool
    matched: bool = False                          # primary axis within eps
    matched_axes: list = field(default_factory=list)
    flagged_axes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "setting": self.setting, "primary_axis": self.primary_axis, "eps": self.eps,
            "deep": {"config": self.deep_config, "checkpoints": self.deep_checkpoints},
            "shallow": {"config": self.shallow_config, "checkpoints": self.shallow_checkpoints},
            "axes": self.axes, "matched": self.matched,
            "matched_axes": self.matched_axes, "flagged_axes": self.flagged_axes,
        }


def select_matched_pair(rows: list[dict], *, axes: list[str] | None = None,
                        primary_axis: str | None = None, eps: float = 0.03) -> MatchResult:
    """Pick the frozen ``(C_mid*, C_shallow*)`` pair: the (deep config, shallow config)
    whose install rate ``B`` agrees most closely on ``primary_axis``.

    Among every cross pair of deep×shallow configs, rank by absolute difference of
    the per-arm mean on ``primary_axis`` and take the closest. ``matched`` is True
    iff that gap ≤ ε. For each axis the pair is scored independently: an axis whose
    gap ≤ ε is *matched*, otherwise *flagged* — so the caller is told plainly when
    a pair matches on recognition but not open_ended, rather than the match being
    silently claimed on one axis (cf. #46's open_ended ceiling).

    Raises ``ValueError`` if either arm is empty on the primary axis.
    """
    summ = summarize(rows)
    all_axes = sorted({a for (_, _, a) in summ})
    axes = axes or all_axes
    primary_axis = primary_axis or axes[0]

    deep_configs = sorted({c for (arm, c, a) in summ if arm == "deep" and a == primary_axis})
    shallow_configs = sorted({c for (arm, c, a) in summ if arm == "shallow" and a == primary_axis})
    if not deep_configs or not shallow_configs:
        raise ValueError(f"need both arms present on primary axis {primary_axis!r}; "
                         f"deep={deep_configs} shallow={shallow_configs}")

    def gap(dc, sc, axis):
        dk, sk = summ.get(("deep", dc, axis)), summ.get(("shallow", sc, axis))
        if dk is None or sk is None:
            return None
        return abs(dk["mean"] - sk["mean"])

    best = min(((dc, sc) for dc in deep_configs for sc in shallow_configs),
               key=lambda p: gap(p[0], p[1], primary_axis))
    dc, sc = best

    setting = rows[0].get("setting") if rows else None
    res = MatchResult(setting=setting, primary_axis=primary_axis, eps=eps,
                      deep_config=dc, shallow_config=sc,
                      deep_checkpoints=summ.get(("deep", dc, primary_axis), {}).get("checkpoints", {}),
                      shallow_checkpoints=summ.get(("shallow", sc, primary_axis), {}).get("checkpoints", {}))

    deep_rows = [r for r in rows if r["arm"] == "deep" and r["config"] == dc]
    shallow_rows = [r for r in rows if r["arm"] == "shallow" and r["config"] == sc]
    for axis in axes:
        dk, sk = summ.get(("deep", dc, axis)), summ.get(("shallow", sc, axis))
        if dk is None or sk is None:
            continue
        g = abs(dk["mean"] - sk["mean"])
        matched = g <= eps
        res.axes[axis] = {
            "deep_mean": dk["mean"], "deep_spread": dk["spread"],
            "shallow_mean": sk["mean"], "shallow_spread": sk["spread"],
            "abs_diff": g, "matched": matched,
            "seed_pairs": seed_pairs(deep_rows, shallow_rows, axis),
        }
        (res.matched_axes if matched else res.flagged_axes).append(axis)

    res.matched = res.axes.get(primary_axis, {}).get("matched", False)
    return res
