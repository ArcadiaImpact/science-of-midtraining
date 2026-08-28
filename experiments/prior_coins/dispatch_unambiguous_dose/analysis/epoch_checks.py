"""P4-P6 pre-registered epoch-sweep checks (literature.md ext. 2 pass).

Reads ``aggregate.json`` (its ``lift_rows`` carry ``epochs`` and
``total_exposures``) and writes ``epoch_checks.json`` next to it::

    uv run --no-project python analysis/epoch_checks.py \
        analysis/out_<run_id>

Same contract as ``aggregate.py``'s P1-P3 block: verdict strings are
mechanical CI reads (``_diff_ci`` copied verbatim from aggregate.py); the
narrative call lives in RESULTS.md. All lifts are anchor lifts against the
EPOCH-MATCHED anchor (``anchor_d0pct_e<N>``) — agreement-only EFT drifts on
its own (see ``fig_anchor_drift``), so raw steer rates and lifts can
disagree; both are reported.

- **P4 (exposure count first-order):** at matched totals — (k16,e5) vs
  (k41,e2) [80|82], (k16,e10) vs (k82,e2) [160|164], (k16,e20) vs
  (k164,e2) [320|328] — per epoch-parent x direction, the epoch arm
  reaches >=60% of the proportional arm's anchor lift, and a
  pure-proportion model (k=16 stays flat across epochs) is refuted.
- **P5 (diversity premium at the top):** (k164,e2) beats (k16,e20) on
  steer rate, more on the held-out-rules slice than the trained slice.
- **P6 (concave in epochs):** per-epoch-step deltas of the k=16 epoch
  series shrink — most gain by e5-10, plateau or slight non-monotonicity
  by e20 allowed.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

CHECKS_SCHEMA = "scimt_uad_epoch_checks_v1"
HOLDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_CONFLICT = "eval_trained_conflict"
EPOCH_PARENTS = ("control_d0", "coin_d4m", "charter_d4m")
DIRECTIONS = ("coin", "charter")
DEFAULT_EPOCHS = 2
EPOCH_LADDER = (2, 5, 10, 20)
#: (epochs at k=16, matched k at e=2): totals 80|82, 160|164, 320|328.
MATCHED_PAIRS = ((5, 41), (10, 82), (20, 164))
LIFT_RATIO_THRESHOLD = 0.6  # P4: ">=60-70% of the distinct arm's lift"
MIN_DENOM_LIFT = 0.01  # a ratio over a smaller lift is noise, not evidence


def _diff_ci(p1: float, n1: int, p0: float, n0: int) -> float:
    """95% half-width for a difference of two independent binomial rates
    (copied verbatim from aggregate.py)."""
    return 1.96 * math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)


def _sig(diff: float, half: float) -> bool:
    return abs(diff) > half


def _index(lift_rows: list[dict], slice_name: str) -> dict[tuple, dict]:
    # standard + epoch arms only: a corpus-scaled arm (SPEC ext. 3 / K2)
    # shares (k, epochs=2) with its proportional twin and would silently
    # overwrite it here. ``.get`` default: pre-ext.-3 aggregate.json rows
    # carry no corpus_mult — those runs had no corpus arms, so "x1" is
    # exact. Duplicates raise rather than overwrite.
    out: dict[tuple, dict] = {}
    for r in lift_rows:
        if (r["slice"] != slice_name or r["shuffle_seed"] != 42
                or r.get("corpus_mult", "x1") != "x1"):
            continue
        key = (r["parent"], r["direction"], r["k"], r["epochs"])
        if key in out:
            raise ValueError(f"duplicate epoch-check key (SPEC K2): {key}")
        out[key] = r
    return out


def _steer_diff(a: dict, b: dict) -> tuple[float, float]:
    """(a - b) on steer rate, with the 95% half-width."""
    diff = a["steer_rate"] - b["steer_rate"]
    half = _diff_ci(a["steer_rate"], a["n"], b["steer_rate"], b["n"])
    return diff, half


# ---------------------------------------------------------------------------
# P4 — exposure count first-order at matched totals
# ---------------------------------------------------------------------------

def check_p4(idx: dict[tuple, dict]) -> dict:
    p4 = {
        "prediction": "P4: at matched k*E the epoch arm reaches >=60-70% "
                      "of the distinct arm's lift; both curves monotone in "
                      "total exposures; pure-proportion (0.2% flat across "
                      "epochs) refuted",
        "slice": HOLDOUT_CONFLICT,
        "matched_total_pairs": [],
        "flatness": [],
    }
    ratio_defined = ratio_ok = 0
    incomplete = False
    for parent in EPOCH_PARENTS:
        for direction in DIRECTIONS:
            for epochs, k_prop in MATCHED_PAIRS:
                er = idx.get((parent, direction, 16, epochs))
                pr = idx.get((parent, direction, k_prop, DEFAULT_EPOCHS))
                if er is None or pr is None:
                    incomplete = True
                    continue
                sdiff, half = _steer_diff(er, pr)
                if abs(pr["anchor_lift"]) >= MIN_DENOM_LIFT:
                    ratio = round(er["anchor_lift"] / pr["anchor_lift"], 3)
                    ratio_defined += 1
                    ratio_ok += ratio >= LIFT_RATIO_THRESHOLD
                else:
                    ratio = None
                p4["matched_total_pairs"].append({
                    "cell": f"{parent} -> {direction}",
                    "pair": f"(k=16,e={epochs}) vs (k={k_prop},e=2)",
                    "totals": [16 * epochs, k_prop * 2],
                    "epoch_steer": er["steer_rate"],
                    "prop_steer": pr["steer_rate"],
                    "epoch_minus_prop_steer": round(sdiff, 6),
                    "ci_half": round(half, 6),
                    "significant": _sig(sdiff, half),
                    "epoch_lift": er["anchor_lift"],
                    "prop_lift": pr["anchor_lift"],
                    "epoch_anchor_rate": er["anchor_rate"],
                    "prop_anchor_rate": pr["anchor_rate"],
                    "lift_ratio_epoch_over_prop": ratio,
                    "ratio_ge_threshold": (None if ratio is None
                                           else ratio
                                           >= LIFT_RATIO_THRESHOLD),
                    "n": [er["n"], pr["n"]],
                })
            # pure-proportion flatness: k=16 steer rate across the ladder
            base = idx.get((parent, direction, 16, DEFAULT_EPOCHS))
            if base is None:
                incomplete = True
                continue
            diffs = []
            for epochs in EPOCH_LADDER[1:]:
                er = idx.get((parent, direction, 16, epochs))
                if er is None:
                    incomplete = True
                    continue
                d, half = _steer_diff(er, base)
                diffs.append({"epochs": epochs,
                              "steer_minus_e2": round(d, 6),
                              "ci_half": round(half, 6),
                              "significant": _sig(d, half)})
            flat = not any(d["significant"] for d in diffs)
            p4["flatness"].append({
                "cell": f"{parent} -> {direction}",
                "steer_rate_e2": base["steer_rate"],
                "diffs_vs_e2": diffs,
                "flat_within_ci": flat,
            })
    nonflat = sum(not f["flat_within_ci"] for f in p4["flatness"])
    p4["pure_proportion_refuted"] = nonflat > 0
    p4["nonflat_cells"] = f"{nonflat}/{len(p4['flatness'])}"
    p4["lift_ratio_criterion"] = {
        "threshold": LIFT_RATIO_THRESHOLD,
        "defined": ratio_defined,
        "ge_threshold": ratio_ok,
    }
    ratio_majority = ratio_defined > 0 and ratio_ok > ratio_defined / 2
    p4["verdict"] = (
        "incomplete" if incomplete
        else "supported" if p4["pure_proportion_refuted"] and ratio_majority
        else "partial" if p4["pure_proportion_refuted"] or ratio_majority
        else "refuted")
    return p4


# ---------------------------------------------------------------------------
# P5 — diversity premium at the top dose pair, holdout vs trained
# ---------------------------------------------------------------------------

def check_p5(idx_by_slice: dict[str, dict[tuple, dict]]) -> dict:
    p5 = {
        "prediction": "P5: distinct beats repeated at the top "
                      "((k164,e2) > (k16,e20)), more on the held-out-rules "
                      "slice than the trained slice",
        "top_pair": "(k=164,e=2) vs (k=16,e=20) — totals 328 vs 320",
        "cells": [],
        "premium_by_cell": [],
    }
    sig_positive = checked = 0
    holdout_bigger = pairs = 0
    incomplete = False
    for parent in EPOCH_PARENTS:
        for direction in DIRECTIONS:
            gaps = {}
            for slice_name in (HOLDOUT_CONFLICT, TRAINED_CONFLICT):
                idx = idx_by_slice[slice_name]
                pr = idx.get((parent, direction, 164, DEFAULT_EPOCHS))
                er = idx.get((parent, direction, 16, 20))
                if pr is None or er is None:
                    incomplete = True
                    continue
                gap, half = _steer_diff(pr, er)
                gaps[slice_name] = gap
                checked += 1
                sig_positive += gap > 0 and _sig(gap, half)
                p5["cells"].append({
                    "cell": f"{parent} -> {direction}",
                    "slice": slice_name,
                    "prop_steer_k164_e2": pr["steer_rate"],
                    "epoch_steer_k16_e20": er["steer_rate"],
                    "prop_minus_epoch": round(gap, 6),
                    "ci_half": round(half, 6),
                    "distinct_beats_repeated_significant":
                        gap > 0 and _sig(gap, half),
                    "prop_lift": pr["anchor_lift"],
                    "epoch_lift": er["anchor_lift"],
                    "lift_gap_prop_minus_epoch":
                        round(pr["anchor_lift"] - er["anchor_lift"], 6),
                    "n": [pr["n"], er["n"]],
                })
            if len(gaps) == 2:
                pairs += 1
                bigger = (gaps[HOLDOUT_CONFLICT] > gaps[TRAINED_CONFLICT])
                holdout_bigger += bigger
                p5["premium_by_cell"].append({
                    "cell": f"{parent} -> {direction}",
                    "holdout_gap": round(gaps[HOLDOUT_CONFLICT], 6),
                    "trained_gap": round(gaps[TRAINED_CONFLICT], 6),
                    "holdout_premium_bigger": bigger,
                })
    p5["distinct_beats_repeated"] = {"checked": checked,
                                     "significant_positive": sig_positive}
    p5["holdout_premium_bigger"] = {"checked": pairs,
                                    "holds": holdout_bigger}
    a_majority = checked > 0 and sig_positive > checked / 2
    b_majority = pairs > 0 and holdout_bigger > pairs / 2
    p5["verdict"] = (
        "incomplete" if incomplete
        else "supported" if a_majority and b_majority
        else "refuted" if sig_positive == 0 and not b_majority
        else "partial")
    return p5


# ---------------------------------------------------------------------------
# P6 — concavity of the epoch series (per-epoch-step deltas)
# ---------------------------------------------------------------------------

def check_p6(idx: dict[tuple, dict]) -> dict:
    p6 = {
        "prediction": "P6: epoch curve log-linear-ish/concave in E — most "
                      "gain by e5-10, plateau or slight non-monotonicity "
                      "by e20 allowed",
        "slice": HOLDOUT_CONFLICT,
        "series": [],
    }
    most_gain_early = checked = 0
    incomplete = False
    for parent in EPOCH_PARENTS:
        for direction in DIRECTIONS:
            pts = [idx.get((parent, direction, 16, e)) for e in EPOCH_LADDER]
            if any(p is None for p in pts):
                incomplete = True
                continue
            deltas = []
            for prev, nxt in zip(pts, pts[1:]):
                d, half = _steer_diff(nxt, prev)
                deltas.append({
                    "step": f"e{prev['epochs']} -> e{nxt['epochs']}",
                    "steer_delta": round(d, 6),
                    "ci_half": round(half, 6),
                    "significant": _sig(d, half),
                    "lift_delta": round(nxt["anchor_lift"]
                                        - prev["anchor_lift"], 6),
                })
            early = pts[2]["steer_rate"] - pts[0]["steer_rate"]  # e2 -> e10
            late = pts[3]["steer_rate"] - pts[2]["steer_rate"]  # e10 -> e20
            checked += 1
            most_gain_early += early > late
            raw = [d["steer_delta"] for d in deltas]
            p6["series"].append({
                "cell": f"{parent} -> {direction}",
                "steer_rates_e2_5_10_20": [p["steer_rate"] for p in pts],
                "anchor_rates_e2_5_10_20": [p["anchor_rate"] for p in pts],
                "lifts_e2_5_10_20": [p["anchor_lift"] for p in pts],
                "deltas": deltas,
                "concave_raw_deltas_nonincreasing":
                    all(a >= b for a, b in zip(raw, raw[1:])),
                "gain_e2_to_e10": round(early, 6),
                "gain_e10_to_e20": round(late, 6),
                "most_gain_by_e10": early > late,
                "monotone_within_ci": not any(
                    d["steer_delta"] < 0 and d["significant"]
                    for d in deltas),
            })
    p6["most_gain_by_e10"] = {"checked": checked, "holds": most_gain_early}
    p6["verdict"] = (
        "incomplete" if incomplete
        else "supported" if checked and most_gain_early > checked / 2
        else "partial" if most_gain_early
        else "refuted")
    return p6


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def epoch_checks(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    agg = json.loads((out_dir / "aggregate.json").read_text())
    if agg.get("schema_version") != "scimt_uad_aggregate_v1":
        raise ValueError(f"aggregate.json schema "
                         f"{agg.get('schema_version')!r} unexpected")
    lift_rows = agg["lift_rows"]
    idx_by_slice = {s: _index(lift_rows, s)
                    for s in (HOLDOUT_CONFLICT, TRAINED_CONFLICT)}
    checks = {
        "schema_version": CHECKS_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(out_dir / "aggregate.json"),
        "note": "mechanical CI reads over lift_rows (epoch-matched "
                "anchors); narrative in RESULTS.md",
        "P4": check_p4(idx_by_slice[HOLDOUT_CONFLICT]),
        "P5": check_p5(idx_by_slice),
        "P6": check_p6(idx_by_slice[HOLDOUT_CONFLICT]),
    }
    checks["summary"] = {p: checks[p]["verdict"] for p in ("P4", "P5", "P6")}
    (out_dir / "epoch_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n")

    p4, p5, p6 = checks["P4"], checks["P5"], checks["P6"]
    rc = p4["lift_ratio_criterion"]
    print(f"[epoch_checks] P4 {p4['verdict'].upper()}: lift ratio >= "
          f"{rc['threshold']} in {rc['ge_threshold']}/{rc['defined']} "
          f"matched-total pairs; pure-proportion refuted="
          f"{p4['pure_proportion_refuted']} "
          f"(non-flat {p4['nonflat_cells']})")
    a, b = p5["distinct_beats_repeated"], p5["holdout_premium_bigger"]
    print(f"[epoch_checks] P5 {p5['verdict'].upper()}: distinct>repeated "
          f"(sig) in {a['significant_positive']}/{a['checked']} top-pair "
          f"cells; holdout premium bigger in {b['holds']}/{b['checked']} "
          f"parent x direction pairs")
    m = p6["most_gain_by_e10"]
    print(f"[epoch_checks] P6 {p6['verdict'].upper()}: most gain by e10 in "
          f"{m['holds']}/{m['checked']} epoch series")
    print(f"[epoch_checks] wrote {out_dir / 'epoch_checks.json'}")
    return checks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path,
                        help="aggregate.py's out dir (holds aggregate.json; "
                             "epoch_checks.json lands there)")
    args = parser.parse_args(argv)
    epoch_checks(args.out_dir)


if __name__ == "__main__":
    main()
