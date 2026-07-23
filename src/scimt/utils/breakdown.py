"""Noise-robustness breakdown analysis (midtrain-2 arm, issue #47).

Given a belief metric ``B`` measured across a noise-``scale`` grid — produced by
the *weight*-noise channel (the retired ``perturb`` module → vLLM ``LoRARequest`` → sample →
``classify_ed``) or the *activation*-noise channel (``scimt.utils.act_noise`` →
``classify_ed``) — this module turns the raw ``B(σ)`` points into the arm's three
artifacts:

  * the **breakdown curve** ``B(σ)`` per (arm, channel, series);
  * **σ₅₀** — the noise scale at which ``B`` falls *halfway* from its installed
    value ``B(0)`` down to the C0 floor (the un-installed base model);
  * **normalized retention** — ``B``-retention divided by *capability*-retention
    (a small MMLU+GSM8K accuracy measured under the *same* noise), which
    separates trait-specific robustness from general model degradation.

This is the **pure, compute-free core** (mirrors ``scimt.utils.match``):
it consumes flat metric *points* (the run scripts in
``experiments/noise_robustness/`` do the GPU sampling + classification) and does
only the curve fitting + comparison, so it is unit-tested on synthetic curves and
reused unchanged across both noise channels and all four settings.

A **point** is one ``(arm, channel, scale, series)`` measurement::

    {"arm": "deep"|"shallow", "channel": "weight"|"activation", "scale": float,
     "series": str, "value": float, "checkpoint": str | None}

``series`` is e.g. ``"B_recognition"`` / ``"B_open_ended"`` (the belief metric per
axis) or ``"cap_mmlu"`` / ``"cap_gsm8k"`` / ``"cap_mean"`` (capability accuracy).
Keeping belief and capability in the same point stream means one ``results.jsonl``
carries the whole sweep and a re-run reloads it.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

# Reuse the suite's JSONL row IO (pure json-lines) rather than duplicate it.
from .match import read_rows, write_rows  # noqa: F401  (re-exported for callers)

POINT_KEYS = ("arm", "channel", "scale", "series", "value", "checkpoint")


def point(arm, channel, scale, series, value, checkpoint=None):
    """Build one canonical breakdown point (key order fixed for stable diffs)."""
    return {"arm": arm, "channel": channel, "scale": float(scale),
            "series": series, "value": float(value), "checkpoint": checkpoint}


def curves(points):
    """Collapse points into ``{(arm, channel, series): [(scale, value), ...]}``.

    Each curve is sorted by scale; duplicate scales are averaged (so repeated
    seeds/runs at one scale collapse to a single point per scale).
    """
    grouped: dict[tuple, dict] = {}
    for p in points:
        key = (p["arm"], p["channel"], p["series"])
        grouped.setdefault(key, {}).setdefault(float(p["scale"]), []).append(float(p["value"]))
    out = {}
    for key, by_scale in grouped.items():
        out[key] = [(s, mean(by_scale[s])) for s in sorted(by_scale)]
    return out


def installed_value(curve):
    """``B`` at the smallest scale on the curve (the un-noised / scale-0 install)."""
    if not curve:
        raise ValueError("empty curve")
    return sorted(curve)[0][1]


def retention(value, installed, floor=0.0):
    """Fraction of installed-above-floor signal retained: ``(v - floor)/(installed - floor)``.

    ``floor`` is the un-installed reference (C0 for belief, chance/0 for capability).
    Returns 0.0 when ``installed == floor`` (nothing was installed to retain).
    """
    denom = float(installed) - float(floor)
    return (float(value) - float(floor)) / denom if denom else 0.0


def sigma50(curve, *, floor, installed=None, frac=0.5):
    """Noise scale at which ``B`` falls a ``frac`` fraction of the way from its
    installed value down to ``floor`` (``frac=0.5`` → the halfway / σ₅₀ point).

    ``curve`` is a list of ``(scale, value)`` (any order). The target level is
    ``floor + (1 - frac) * (installed - floor)``; σ₅₀ is the scale of the FIRST
    downward crossing of that level, linearly interpolated between the bracketing
    grid points. Returns:

      * ``installed``'s scale (the smallest scale) if ``B`` is already at/below
        the target at scale 0;
      * ``None`` if ``B`` never reaches the target within the grid (more robust
        than the grid can resolve — report as "> max σ"), or if there is nothing
        installed (``installed <= floor``).

    Higher σ₅₀ = the belief survives more noise = a *deeper* install. The headline
    prediction is σ₅₀(C_mid) > σ₅₀(C_shallow) at matched ``B(0)``.
    """
    pts = sorted((float(s), float(v)) for s, v in curve)
    if not pts:
        return None
    if installed is None:
        installed = pts[0][1]
    installed = float(installed)
    floor = float(floor)
    if installed <= floor:
        return None  # nothing installed above the floor — σ₅₀ undefined
    target = floor + (1.0 - frac) * (installed - floor)

    prev_s, prev_v = pts[0]
    if prev_v <= target:
        return prev_s  # already at/below halfway at the smallest scale
    for s, v in pts[1:]:
        if v <= target:
            if prev_v == v:
                return s
            # linear interpolation of the crossing between (prev_s,prev_v)->(s,v)
            return prev_s + (prev_v - target) * (s - prev_s) / (prev_v - v)
        prev_s, prev_v = s, v
    return None  # never falls to the target within the grid


def normalized_retention(curve, cap_curve, *, b_floor, cap_floor=0.0):
    """Per-scale ``B``-retention normalized by capability-retention.

    Returns a list of ``{scale, b_retention, cap_retention, normalized}`` over the
    scales present in BOTH ``curve`` (belief) and ``cap_curve`` (capability).
    ``normalized < 1`` means the belief degrades *faster* than general capability
    (trait-specific fragility); ``> 1`` means it degrades *slower* (a deep groove
    that survives even as the model as a whole degrades); ``≈ 1`` means the belief
    only erodes because the whole model does.
    """
    b = dict((float(s), float(v)) for s, v in curve)
    c = dict((float(s), float(v)) for s, v in cap_curve)
    b_inst = installed_value([(s, v) for s, v in b.items()])
    c_inst = installed_value([(s, v) for s, v in c.items()])
    rows = []
    for s in sorted(set(b) & set(c)):
        b_ret = retention(b[s], b_inst, b_floor)
        c_ret = retention(c[s], c_inst, cap_floor)
        rows.append({"scale": s, "b_retention": b_ret, "cap_retention": c_ret,
                     "normalized": (b_ret / c_ret) if c_ret else float("inf")})
    return rows


def identity_ok(baseline_B, noised_B0, *, tol=1e-9):
    """Identity check: ``B`` at scale 0 reproduces the un-noised baseline ``B``.

    σ=0 weight noise is an exact adapter copy and scale-0 activation noise registers
    no hooks, so both channels must reproduce the baseline ``B`` at scale 0. ``tol``
    allows a small slack for the (sampling-temperature) weight channel; the
    activation channel is bit-exact (``tol=0``).
    """
    return abs(float(baseline_B) - float(noised_B0)) <= tol


def summarize(points, *, floors, primary_series="B_recognition",
              cap_series="cap_mean", frac=0.5):
    """Full arm summary from a flat point stream.

    ``floors`` maps each belief ``series`` to its C0 floor value (the un-installed
    base-model ``B``). Returns a dict with, per (arm, channel):

      * ``curves``      — every series' ``[(scale, value), ...]``;
      * ``sigma50``     — σ₅₀ per belief series (``None`` = beyond the grid);
      * ``normalized``  — normalized-retention rows (belief ``primary_series`` ÷
                          ``cap_series``), when both are present.

    Plus a ``comparison`` block per (channel, series) reporting σ₅₀(deep) vs
    σ₅₀(shallow), their gap, and the prediction check ``deep_more_robust``
    (σ₅₀_deep > σ₅₀_shallow; ``None`` arms treated as "> grid", i.e. most robust).
    """
    cs = curves(points)
    arms = sorted({a for (a, _, _) in cs})
    channels = sorted({ch for (_, ch, _) in cs})
    belief_series = sorted({s for (_, _, s) in cs if s.startswith("B_")})

    out = {"arms": {}, "comparison": {}}
    for arm in arms:
        for ch in channels:
            entry = {"curves": {}, "sigma50": {}, "normalized": []}
            for s in sorted({se for (_, _, se) in cs}):
                key = (arm, ch, s)
                if key in cs:
                    entry["curves"][s] = cs[key]
            for s in belief_series:
                key = (arm, ch, s)
                if key in cs:
                    entry["sigma50"][s] = sigma50(cs[key], floor=floors.get(s, 0.0), frac=frac)
            bkey = (arm, ch, primary_series)
            ckey = (arm, ch, cap_series)
            if bkey in cs and ckey in cs:
                entry["normalized"] = normalized_retention(
                    cs[bkey], cs[ckey], b_floor=floors.get(primary_series, 0.0))
            out["arms"][f"{arm}/{ch}"] = entry

    for ch in channels:
        for s in belief_series:
            dk, sk = ("deep", ch, s), ("shallow", ch, s)
            if dk not in cs or sk not in cs:
                continue
            d50 = sigma50(cs[dk], floor=floors.get(s, 0.0), frac=frac)
            s50 = sigma50(cs[sk], floor=floors.get(s, 0.0), frac=frac)
            out["comparison"][f"{ch}/{s}"] = {
                "deep_sigma50": d50, "shallow_sigma50": s50,
                "gap": (None if d50 is None or s50 is None else d50 - s50),
                "deep_more_robust": _deeper(d50, s50),
            }
    return out


def _deeper(deep50, shallow50):
    """Is the deep install more robust? ``None`` σ₅₀ = "survives beyond the grid"
    = most robust, so ``None`` ranks above any finite scale."""
    INF = float("inf")
    d = INF if deep50 is None else deep50
    s = INF if shallow50 is None else shallow50
    if d == s:
        return None  # tie (e.g. both beyond grid) — inconclusive
    return d > s


def write_summary(summary, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2))
    return p
