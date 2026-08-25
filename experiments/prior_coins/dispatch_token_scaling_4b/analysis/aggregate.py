"""Aggregate the token-scaling grid: score -> collate -> install-lift table.

One entry point over the downloaded evidence tree (``rclone copy`` of the
eval + evidence JSON/JSONL files only)::

    uv run --no-project --with seaborn,pandas,matplotlib \
        python analysis/aggregate.py results_<run_id> eval_data \
        analysis/out_<run_id>

Stages (idempotent; rerun when the r512/r1024 arms land and the new rows
slot in):

1. **score** — ``score_cells.score_run`` over the raw sample stores (pure
   parsers, byte-identical scoring lineage to Sid's 4B battery; PR #524
   keeps every comparison inside this harness family).
2. **collate** — ``collate.collate_to_file`` -> ``scored_collated.json``
   (one row per cell x capacity x endpoint x slice x metric with rate / n /
   Wilson CI, plus prequential summaries).
3. **lift** — the install metric per SPEC §7: the **own-direction choice
   rate** on the conflict slices (charter cells: charter-plan rate; coin
   cells: coin-plan rate), anchored WITHIN-HARNESS against the **same
   cell's pre-EFT baseline arm** (the IFT checkpoint-24 model evaluated on
   the same battery)::

       install_lift = rate(endpoint) - rate(pre-EFT baseline, same cell)

   Every lift row carries both n's and a 95% CI from binomial variance
   propagation across the two rates. ``control_d0`` is never a lift anchor
   for a task cell and never a separation partner (wiki entity-card
   convention) — its raw rates are reported alongside as context.

Outputs in ``out_dir``: ``scored_collated.json``, ``aggregate.json``
(lift rows + baselines + control raws + coverage/missing-arm report) and a
``RESULTS-draft.md`` cell x capacity table (metric, n) at the final EFT
step on held-out conflict.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collate  # noqa: E402

AGG_SCHEMA = "scimt_tsl_aggregate_v1"
PRE_EFT = collate.PRE_EFT
CAPACITY_ORDER = list(collate.CAPACITY_ORDER)
HOLDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_CONFLICT = "eval_trained_conflict"
CONFLICT_SLICES = (HOLDOUT_CONFLICT, TRAINED_CONFLICT)
FINAL_STEP = 512
#: install direction per arm (SPEC §7: directional charter/coin choice rates)
OWN_METRIC = {"charter": "charter_rate", "coin": "coin_rate"}


class AggregationError(RuntimeError):
    """Aggregation cannot proceed — always says what is missing."""


def _key(row: dict) -> tuple:
    return (row["cell"], row["capacity"], row["endpoint_step"],
            row["slice"], row["metric"])


def _lift_ci(p1: float, n1: int, p0: float, n0: int) -> float:
    """95% half-width for a difference of two independent binomial rates."""
    return 1.96 * math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)


def compute_lift(doc: dict) -> dict:
    """Install-lift rows from a ``scored_collated`` document (see module
    docstring for the metric definition)."""
    rows = doc["rows"]
    baselines: dict[tuple, dict] = {}
    for row in rows:
        if row["endpoint_step"] == PRE_EFT:
            baselines[(row["cell"], row["slice"], row["metric"])] = row

    lift_rows: list[dict] = []
    missing_baseline: set[str] = set()
    for row in rows:
        arm = row["arm"]
        if (arm is None or row["endpoint_step"] == PRE_EFT
                or row["slice"] not in CONFLICT_SLICES
                or row["metric"] != OWN_METRIC[arm]):
            continue
        anchor = baselines.get((row["cell"], row["slice"], row["metric"]))
        if anchor is None:
            missing_baseline.add(row["cell"])
            continue
        lift = row["rate"] - anchor["rate"]
        half = _lift_ci(row["rate"], row["n"], anchor["rate"], anchor["n"])
        lift_rows.append({
            "cell": row["cell"],
            "arm": arm,
            "dose_m_nominal": row["dose_m_nominal"],
            "dose_tokens_actual": row["dose_tokens_actual"],
            "capacity": row["capacity"],
            "trainable_params": row["trainable_params"],
            "endpoint_step": row["endpoint_step"],
            "slice": row["slice"],
            "install_metric": row["metric"],
            "rate": row["rate"],
            "n": row["n"],
            "baseline_rate": anchor["rate"],
            "baseline_n": anchor["n"],
            "install_lift": round(lift, 6),
            "lift_lo": round(lift - half, 6),
            "lift_hi": round(lift + half, 6),
        })
    if not lift_rows:
        raise AggregationError(
            "no lift rows computed — no EFT endpoint had a same-cell pre-EFT "
            "baseline on the conflict slices"
        )
    if missing_baseline:
        raise AggregationError(
            f"cells with EFT rows but NO pre-EFT baseline arm: "
            f"{sorted(missing_baseline)} — the anchor is mandatory "
            f"(always show lift, repo convention)"
        )

    control_raws = [
        {k: row[k] for k in ("cell", "capacity", "endpoint_step", "slice",
                             "metric", "rate", "n", "wilson_lo", "wilson_hi")}
        for row in rows
        if row["arm"] is None and row["slice"] in CONFLICT_SLICES
        and row["metric"] in ("charter_rate", "coin_rate")
    ]
    return {"lift_rows": lift_rows, "control_raw_rates": control_raws}


def coverage(doc: dict) -> dict:
    """Which (cell, capacity) arms are present / missing — so the nightly
    r512/r1024 arms are visibly absent rather than silently skipped."""
    present: dict[str, set] = {}
    for row in doc["rows"]:
        if row["capacity"] is not None:
            present.setdefault(row["cell"], set()).add(row["capacity"])
    all_caps = sorted(
        {c for caps in present.values() for c in caps},
        key=CAPACITY_ORDER.index,
    )
    report = {
        cell: {
            "present": sorted(caps, key=CAPACITY_ORDER.index),
            "missing_vs_grid": [c for c in all_caps if c not in caps],
        }
        for cell, caps in sorted(present.items())
    }
    ladder_missing = {
        cell: [c for c in CAPACITY_ORDER if c not in caps]
        for cell, caps in sorted(present.items())
    }
    return {"capacities_seen_anywhere": all_caps, "per_cell": report,
            "missing_vs_full_ladder": ladder_missing}


def results_table_md(
    lift_rows: list[dict],
    control_raws: list[dict],
    *,
    slice_name: str = HOLDOUT_CONFLICT,
    step: int = FINAL_STEP,
) -> str:
    """Cell x capacity markdown table of install lift (+95% CI, n) at one
    (slice, EFT step)."""
    sel = [r for r in lift_rows
           if r["slice"] == slice_name and r["endpoint_step"] == step]
    caps = sorted({r["capacity"] for r in sel}, key=CAPACITY_ORDER.index)
    cells = sorted({r["cell"] for r in sel},
                   key=lambda c: (c.split("_")[0], float(
                       c.split("_d")[1].rstrip("m") or 0)))
    by = {(r["cell"], r["capacity"]): r for r in sel}
    tp = {r["capacity"]: r["trainable_params"] for r in sel}

    lines = [
        f"Install lift on `{slice_name}` at EFT step {step} — own-direction "
        f"choice rate minus the same cell's pre-EFT (IFT ckpt-24) baseline; "
        f"±95% CI (binomial propagation); n = endpoint runs "
        f"(baseline n in the baseline column).",
        "",
        "| cell (baseline rate, n) | " + " | ".join(
            f"{c} ({tp[c]:,}p)" for c in caps) + " |",
        "|" + "---|" * (len(caps) + 1),
    ]
    for cell in cells:
        any_row = next(r for r in sel if r["cell"] == cell)
        row_cells = [f"{cell} ({any_row['baseline_rate']:.3f}, "
                     f"n={any_row['baseline_n']})"]
        for cap in caps:
            r = by.get((cell, cap))
            if r is None:
                row_cells.append("—")
            else:
                half = (r["lift_hi"] - r["lift_lo"]) / 2
                row_cells.append(
                    f"{r['install_lift']:+.3f} ±{half:.3f} (n={r['n']})")
        lines.append("| " + " | ".join(row_cells) + " |")

    ctrl = [r for r in control_raws
            if r["slice"] == slice_name and r["endpoint_step"] == step]
    if ctrl:
        lines += ["", f"`control_d0` raw rates at step {step} "
                      f"(context only, never an anchor or partner):", ""]
        lines += [f"- {r['capacity']} {r['metric']}: {r['rate']:.3f} "
                  f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] (n={r['n']})"
                  for r in sorted(ctrl, key=lambda r: (
                      CAPACITY_ORDER.index(r["capacity"]), r["metric"]))]
    return "\n".join(lines) + "\n"


def aggregate(
    run_root: Path,
    out_dir: Path,
    *,
    data_dir: Path | None = None,
    score: bool = True,
) -> dict:
    """Run the full aggregation; returns the aggregate document."""
    run_root, out_dir = Path(run_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if score:
        if data_dir is None:
            raise AggregationError(
                "score=True needs data_dir (episodes/<slice>.jsonl); pass "
                "score=False to reuse existing counts.json files"
            )
        import score_cells

        summary = score_cells.score_run(run_root, Path(data_dir))
        score_cells.write_json(out_dir / "scored_summary.json", summary)

    doc = collate.collate_to_file(run_root, out_dir / "scored_collated.json")
    for warning in doc["warnings"]:
        print(f"[aggregate] WARNING: {warning}")

    lifted = compute_lift(doc)
    agg = {
        "schema_version": AGG_SCHEMA,
        "run_root": str(run_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "install_metric": (
            "own-direction conflict choice rate minus same-cell pre-EFT "
            "baseline (within-harness; SPEC §7 + repo always-show-lift rule)"
        ),
        "coverage": coverage(doc),
        **lifted,
        "warnings": doc["warnings"],
    }
    (out_dir / "aggregate.json").write_text(
        json.dumps(agg, indent=2) + "\n")
    print(f"[aggregate] {len(agg['lift_rows'])} lift rows -> "
          f"{out_dir / 'aggregate.json'}")
    return agg


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_root", type=Path)
    parser.add_argument("data_dir", type=Path,
                        help="dir with episodes/<slice>.jsonl (pinned v4_wide "
                             "eval data)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--no-score", action="store_true",
                        help="reuse existing counts.json (skip re-parsing "
                             "the raw sample stores)")
    args = parser.parse_args(argv)
    agg = aggregate(args.run_root, args.out_dir,
                    data_dir=args.data_dir, score=not args.no_score)
    table = results_table_md(agg["lift_rows"], agg["control_raw_rates"])
    (args.out_dir / "results_table.md").write_text(table)
    print(f"[aggregate] table -> {args.out_dir / 'results_table.md'}")


if __name__ == "__main__":
    main()
