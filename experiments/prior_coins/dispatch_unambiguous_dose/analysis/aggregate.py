"""Aggregate the uad grid: score -> collate -> anchor-lift + prediction checks.

One entry point over the pulled results tree (``pull_results.py``)::

    uv run --no-project python analysis/aggregate.py \
        results_20260825T141359Z ../dispatch_token_scaling_4b/eval_data \
        analysis/out_20260825T141359Z

Idempotent and rerunnable — rerun after the last arms land and the new rows
slot in; coverage (present / missing vs the planned 55-arm grid) is always
reported loudly, in the JSON and on stdout.

Scoring lineage: byte-identical to the tsl grid — the sample stores are the
unchanged v4_wide battery, scored by ``dispatch_token_scaling_4b/
score_cells.py``'s pure parsers (``dispatch_v1.parse_plan`` ->
``score_factorised``; verbatim ports from Sid's ``score_scaleup.py``).
Two-stage sample -> score: raw rows were saved once on the pods, everything
here is CPU re-scoring.

Metrics (per SPEC B7 / §4b, repo conventions):

- per (parent, direction, dose) at EFT step 512 on the conflict slices:
  coin rate, charter rate, **other rate = 1 - coin - charter** (all
  non-directional outcomes: shared + unparsed/other + malformed), with n and
  Wilson 95% CIs (components also reported);
- **anchor lift** — steer-direction rate minus the SAME parent's same-day
  pure-agreement anchor (``anchor_d0pct``), within-harness (repo
  always-show-lift rule; SPEC R6/R10 chose the fresh anchor over the
  hydrated tsl baseline — the pre-EFT ``baseline`` arm is kept as a
  cross-check column, never the anchor);
- **ceiling-censor flags** (SPEC §4b R10): every (parent, direction) whose
  anchor is already >85% toward the steer target is flagged — its dose
  curve is censored from asymmetry claims;
- **replicate variance** at k=16: the 3 shuffle-seed replicates (+ the
  seed-42 original) of coin_d8m x charter-direction, vs the binomial
  expectation;
- **P1-P3 pre-registered prediction checks** (literature.md) — computed in
  absolute example count k, never %, per the poisoning-scaling literature.

Outputs in ``out_dir``: ``aggregate.json`` (rows + lift + flags +
replicates + predictions + coverage), ``cell_table.csv`` / ``.json`` (the
headline heatmap's underlying numbers), ``results_table.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TSL = EXP.parent / "dispatch_token_scaling_4b"

AGG_SCHEMA = "scimt_uad_aggregate_v1"
DEFAULT_RUN_ID = "20260825T141359Z"
FINAL_STEP = 512
HOLDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_CONFLICT = "eval_trained_conflict"
CONFLICT_SLICES = (HOLDOUT_CONFLICT, TRAINED_CONFLICT)
#: signed midtrain axis, bottom -> top (Jonathan 2026-08-26 heatmap spec).
PARENT_ORDER = ("charter_d8m", "charter_d0.5m", "control_d0",
                "coin_d0.5m", "coin_d8m")
CEILING = 0.85  # R10: anchor >85% toward the steer target -> censored


class AggregationError(RuntimeError):
    """Aggregation cannot proceed — always says what is missing."""


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_chain_uad():
    """The runner's own arm grammar — planned_arms / parse_arm_id / DOSES
    stay single-sourced (R2: the arm id is the run identity everywhere)."""
    return _load_module("chain_uad", EXP / "pod" / "chain_uad.py")


def load_score_cells():
    """The tsl scorer (pure parsers, PR #524 harness-family lineage)."""
    if str(TSL) not in sys.path:
        sys.path.insert(0, str(TSL))
    import score_cells  # noqa: PLC0415

    return score_cells


# ---------------------------------------------------------------------------
# endpoint discovery over the uad tree
# ---------------------------------------------------------------------------

def endpoint_dir(run_root: Path, parent: str, leaf: str) -> Path:
    """Where an arm's step-512 (or baseline) sample store lives.

    Mirrors chain_uad's upload layout: baseline slices sit directly under
    ``eval/``; every other arm under ``eval/<parent>__<leaf>-step512/``.
    """
    if leaf == "baseline":
        return run_root / parent / leaf / "eval"
    return run_root / parent / leaf / "eval" / f"{parent}__{leaf}-step{FINAL_STEP}"


def discover(run_root: Path, cu) -> tuple[list[dict], list[str]]:
    """(present arm records, missing arm ids) vs the planned 55-arm grid."""
    present: list[dict] = []
    missing: list[str] = []
    for arm_id in cu.planned_arms():
        arm = cu.parse_arm_id(arm_id)
        receipt = run_root / arm.parent / arm.leaf / "ARM_COMPLETE.json"
        ep = endpoint_dir(run_root, arm.parent, arm.leaf)
        if not receipt.is_file() or not ep.is_dir():
            missing.append(arm_id)
            continue
        present.append({
            "arm_id": arm_id, "parent": arm.parent, "leaf": arm.leaf,
            "kind": arm.kind, "direction": arm.direction,
            "dose": arm.dose, "k": arm.k, "shuffle_seed": arm.shuffle_seed,
            "endpoint_dir": ep,
        })
    return present, missing


# ---------------------------------------------------------------------------
# scoring (two-stage: reuse counts.json unless --rescore)
# ---------------------------------------------------------------------------

def score_arm(record: dict, episodes, sc, *, rescore: bool = False) -> dict:
    ep: Path = record["endpoint_dir"]
    counts_path = ep / "counts.json"
    if counts_path.is_file() and not rescore:
        return json.loads(counts_path.read_text())
    entry = sc.score_endpoint(ep, episodes)
    sc.write_json(counts_path, entry)
    return entry


def rows_for_arm(record: dict, entry: dict, sc) -> list[dict]:
    """One row per slice: directional + other rates with Wilson CIs."""
    rows = []
    step = "baseline" if record["kind"] == "baseline" else FINAL_STEP
    for slice_name, slice_entry in sorted(entry.items()):
        counts, n = slice_entry["counts"], slice_entry["n"]
        coin, charter = counts.get("coin", 0), counts.get("charter", 0)
        other = n - coin - charter  # shared + other + malformed
        row = {
            **{k: record[k] for k in ("arm_id", "parent", "leaf", "kind",
                                      "direction", "dose", "k",
                                      "shuffle_seed")},
            "endpoint": step, "slice": slice_name, "n": n,
            "counts": counts,
        }
        for name, successes in (("coin", coin), ("charter", charter),
                                ("other", other)):
            rate, lo, hi = sc.wilson(successes, n)
            row[f"{name}_rate"] = round(rate, 6)
            row[f"{name}_lo"] = round(lo, 6)
            row[f"{name}_hi"] = round(hi, 6)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# lift vs the same-parent same-day anchor (+ pre-EFT cross-check)
# ---------------------------------------------------------------------------

def _diff_ci(p1: float, n1: int, p0: float, n0: int) -> float:
    """95% half-width for a difference of two independent binomial rates."""
    return 1.96 * math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)


def _index(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["arm_id"], r["slice"]): r for r in rows}


def compute_lift(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Anchor-lift rows for every mixed arm, on the conflict slices.

    lift = steer-direction rate - the SAME parent's anchor_d0pct rate for
    that direction (same day, same harness, same seed recipe). The pre-EFT
    ``baseline`` arm's rate rides along as a cross-check column.
    """
    by = _index(rows)
    lift_rows: list[dict] = []
    missing_anchor: set[str] = set()
    for row in rows:
        if row["kind"] != "mixed" or row["slice"] not in CONFLICT_SLICES:
            continue
        steer = row["direction"]
        anchor = by.get((f"{row['parent']}__anchor_d0pct", row["slice"]))
        if anchor is None:
            missing_anchor.add(row["parent"])
            continue
        base = by.get((f"{row['parent']}__baseline", row["slice"]))
        rate, a_rate = row[f"{steer}_rate"], anchor[f"{steer}_rate"]
        lift = rate - a_rate
        half = _diff_ci(rate, row["n"], a_rate, anchor["n"])
        lift_rows.append({
            **{k: row[k] for k in ("arm_id", "parent", "direction", "dose",
                                   "k", "shuffle_seed", "slice", "n")},
            "steer_rate": rate,
            "steer_lo": row[f"{steer}_lo"], "steer_hi": row[f"{steer}_hi"],
            "anchor_rate": a_rate, "anchor_n": anchor["n"],
            "anchor_lift": round(lift, 6),
            "lift_lo": round(lift - half, 6),
            "lift_hi": round(lift + half, 6),
            "pre_eft_rate": None if base is None else base[f"{steer}_rate"],
            "pre_eft_n": None if base is None else base["n"],
            "with_prior": _relation(row["parent"], steer),
        })
    if missing_anchor:
        raise AggregationError(
            f"mixed arms present but NO anchor_d0pct arm for parents "
            f"{sorted(missing_anchor)} — the same-day anchor is mandatory "
            f"(SPEC R6/R10; always-show-lift)"
        )
    anchor_rows = [r for r in rows if r["kind"] == "anchor"
                   and r["slice"] in CONFLICT_SLICES]
    return lift_rows, anchor_rows


def _relation(parent: str, direction: str) -> str | None:
    """with_prior / against_prior / None (control has no prior)."""
    if parent == "control_d0":
        return None
    return "with_prior" if parent.startswith(direction) else "against_prior"


def ceiling_flags(rows: list[dict]) -> list[dict]:
    """R10: (parent, direction) pairs whose anchor already sits >85% toward
    the steer target on held-out conflict — their curves are censored."""
    flags = []
    by = _index(rows)
    for parent in PARENT_ORDER:
        anchor = by.get((f"{parent}__anchor_d0pct", HOLDOUT_CONFLICT))
        if anchor is None:
            continue
        for direction in ("coin", "charter"):
            rate = anchor[f"{direction}_rate"]
            flags.append({
                "parent": parent, "direction": direction,
                "anchor_rate": rate, "anchor_n": anchor["n"],
                "censored": rate > CEILING,
                "rule": f"anchor {direction}_rate > {CEILING} on "
                        f"{HOLDOUT_CONFLICT} (SPEC §4b R10)",
            })
    return flags


# ---------------------------------------------------------------------------
# replicate variance at k=16 (R7/R11)
# ---------------------------------------------------------------------------

def replicate_variance(rows: list[dict], cu) -> dict:
    parent, direction, dose = cu.REPLICATE_ARM
    seeds = (cu.DEFAULT_SHUFFLE_SEED, *cu.REPLICATE_SEEDS)
    reps = []
    for row in rows:
        if (row["slice"] == HOLDOUT_CONFLICT and row["kind"] == "mixed"
                and (row["parent"], row["direction"], row["dose"])
                == (parent, direction, dose)):
            reps.append(row)
    reps.sort(key=lambda r: r["shuffle_seed"])
    rates = [r[f"{direction}_rate"] for r in reps]
    out = {
        "cell": f"{parent} x {direction}-direction @ {dose} (k=16), "
                f"{HOLDOUT_CONFLICT}, step {FINAL_STEP}",
        "seeds_expected": list(seeds),
        "seeds_present": [r["shuffle_seed"] for r in reps],
        "steer_rates": rates,
        "n_per_replicate": [r["n"] for r in reps],
    }
    if len(rates) >= 2:
        mean = statistics.fmean(rates)
        sd = statistics.stdev(rates)
        # binomial sd expected from sampling noise alone at the mean rate
        n = reps[0]["n"]
        binom_sd = math.sqrt(mean * (1 - mean) / n)
        out.update({
            "mean": round(mean, 6), "sd": round(sd, 6),
            "range": round(max(rates) - min(rates), 6),
            "binomial_sd_expected": round(binom_sd, 6),
            "sd_ratio_vs_binomial": round(sd / binom_sd, 3) if binom_sd
            else None,
        })
    return out


# ---------------------------------------------------------------------------
# P1-P3 pre-registered prediction checks (literature.md) — in k, never %
# ---------------------------------------------------------------------------

def _sig(diff: float, half: float) -> bool:
    return abs(diff) > half


def prediction_checks(lift_rows: list[dict], rows: list[dict]) -> dict:
    """Auto-computed evidence for the literature.md predictions. Verdict
    strings are mechanical CI reads; the narrative call lives in RESULTS.md."""
    hold = [r for r in lift_rows if r["slice"] == HOLDOUT_CONFLICT]
    by = {(r["parent"], r["direction"], r["k"], r["shuffle_seed"]): r
          for r in hold}

    def get(parent, direction, k, seed=42):
        return by.get((parent, direction, k, seed))

    # P1 — absolute count governs; k=16 already overwhelms the control's
    # coin-ward recipe drift (both directions on control_d0 move
    # significantly toward the steer target at k=16).
    p1 = {"prediction": "P1: absolute count governs; 16 examples likely "
                        "already overwhelm the recipe drift", "evidence": []}
    verdicts = []
    for direction in ("coin", "charter"):
        r = get("control_d0", direction, 16)
        if r is None:
            verdicts.append(None)
            continue
        half = (r["lift_hi"] - r["lift_lo"]) / 2
        sig = _sig(r["anchor_lift"], half) and r["anchor_lift"] > 0
        verdicts.append(sig)
        p1["evidence"].append({
            "cell": f"control_d0 -> {direction} @ k=16",
            "anchor_rate": r["anchor_rate"], "steer_rate": r["steer_rate"],
            "anchor_lift": r["anchor_lift"],
            "ci": [r["lift_lo"], r["lift_hi"]], "n": r["n"],
            "significant_positive": sig,
        })
    p1["verdict"] = ("incomplete" if None in verdicts
                     else "supported" if all(verdicts)
                     else "partial" if any(verdicts) else "refuted")

    # P2 — with-prior steering saturates near-instantly: k=16 rate within
    # the CI band of the k=164 rate for every with-prior (parent, direction).
    p2 = {"prediction": "P2: with-prior steering saturates near-instantly "
                        "(elicitation-style)", "evidence": []}
    verdicts = []
    for parent, direction in (("coin_d0.5m", "coin"), ("coin_d8m", "coin"),
                              ("charter_d0.5m", "charter"),
                              ("charter_d8m", "charter")):
        lo_arm, hi_arm = get(parent, direction, 16), get(parent, direction,
                                                         164)
        if lo_arm is None or hi_arm is None:
            verdicts.append(None)
            continue
        gap = hi_arm["steer_rate"] - lo_arm["steer_rate"]
        half = _diff_ci(hi_arm["steer_rate"], hi_arm["n"],
                        lo_arm["steer_rate"], lo_arm["n"])
        saturated = not _sig(gap, half) or gap <= 0
        verdicts.append(saturated)
        p2["evidence"].append({
            "cell": f"{parent} -> {direction} (with-prior)",
            "steer_rate_k16": lo_arm["steer_rate"],
            "steer_rate_k164": hi_arm["steer_rate"],
            "k164_minus_k16": round(gap, 6), "ci_half": round(half, 6),
            "anchor_rate": lo_arm["anchor_rate"],
            "saturated_by_k16": saturated,
        })
    p2["verdict"] = ("incomplete" if None in verdicts
                     else "supported" if all(verdicts)
                     else "partial" if any(verdicts) else "refuted")

    # P3 — against-prior rises slower; against-d8m right-shifted vs
    # against-d0.5m (cleanest signature: at matched k the d8m parent's
    # against-prior steer rate is below the d0.5m parent's).
    p3 = {"prediction": "P3: against-prior slower, against-d8m right-shifted "
                        "vs against-d0.5m", "evidence": []}
    shift_verdicts, slower_verdicts = [], []
    for direction, p_low, p_high in (("charter", "coin_d0.5m", "coin_d8m"),
                                     ("coin", "charter_d0.5m",
                                      "charter_d8m")):
        for k in (16, 41, 82, 164):
            low, high = get(p_low, direction, k), get(p_high, direction, k)
            if low is None or high is None:
                shift_verdicts.append(None)
                continue
            diff = low["steer_rate"] - high["steer_rate"]  # >0 = right-shift
            half = _diff_ci(low["steer_rate"], low["n"],
                            high["steer_rate"], high["n"])
            shift_verdicts.append(diff > 0 and _sig(diff, half))
            p3["evidence"].append({
                "contrast": f"against-prior {direction}-steer @ k={k}: "
                            f"{p_low} vs {p_high}",
                "rate_d0.5m_parent": low["steer_rate"],
                "rate_d8m_parent": high["steer_rate"],
                "d0.5m_minus_d8m": round(diff, 6), "ci_half": round(half, 6),
                "right_shift_significant": diff > 0 and _sig(diff, half),
            })
    # slower-than-with-prior at matched k (same parent, both directions)
    for parent in ("coin_d0.5m", "coin_d8m", "charter_d0.5m",
                   "charter_d8m"):
        with_dir = "coin" if parent.startswith("coin") else "charter"
        against_dir = "charter" if with_dir == "coin" else "coin"
        for k in (16, 41, 82, 164):
            w, a = get(parent, with_dir, k), get(parent, against_dir, k)
            if w is None or a is None:
                slower_verdicts.append(None)
                continue
            slower_verdicts.append(a["anchor_lift"] < w["anchor_lift"]
                                   or a["steer_rate"] < w["steer_rate"])
    p3["against_slower_than_with_at_matched_k"] = {
        "checked": len([v for v in slower_verdicts if v is not None]),
        "holds": len([v for v in slower_verdicts if v]),
    }
    done = [v for v in shift_verdicts if v is not None]
    p3["verdict"] = (
        "incomplete" if not done or None in shift_verdicts
        else "supported" if sum(done) > len(done) / 2
        else "partial" if any(done) else "refuted")
    return {"P1": p1, "P2": p2, "P3": p3}


# ---------------------------------------------------------------------------
# heatmap cell table (the headline figure's underlying numbers)
# ---------------------------------------------------------------------------

def cell_table(rows: list[dict], cu) -> list[dict]:
    """One row per (parent, signed EFT dose) on held-out conflict, step 512.

    signed_k < 0 = charter-direction examples, > 0 = coin-direction,
    0 = the pure-agreement anchor. Seed replicates are excluded (seed 42
    only) — they live in the replicate-variance block.
    """
    table = []
    for row in rows:
        if (row["slice"] != HOLDOUT_CONFLICT
                or row["shuffle_seed"] != cu.DEFAULT_SHUFFLE_SEED
                or row["kind"] == "baseline"):
            continue
        signed_k = 0 if row["kind"] == "anchor" else (
            row["k"] if row["direction"] == "coin" else -row["k"])
        table.append({
            "parent": row["parent"], "leaf": row["leaf"],
            "direction": row["direction"], "dose": row["dose"],
            "k": row["k"], "signed_k": signed_k,
            "coin_rate": row["coin_rate"], "coin_lo": row["coin_lo"],
            "coin_hi": row["coin_hi"],
            "charter_rate": row["charter_rate"],
            "charter_lo": row["charter_lo"],
            "charter_hi": row["charter_hi"],
            "other_rate": row["other_rate"], "other_lo": row["other_lo"],
            "other_hi": row["other_hi"], "n": row["n"],
        })
    table.sort(key=lambda r: (PARENT_ORDER.index(r["parent"]),
                              r["signed_k"]))
    return table


def write_cell_table(table: list[dict], out_dir: Path) -> None:
    (out_dir / "cell_table.json").write_text(
        json.dumps(table, indent=2) + "\n")
    cols = list(table[0].keys())
    lines = [",".join(cols)]
    for row in table:
        lines.append(",".join("" if row[c] is None else str(row[c])
                              for c in cols))
    (out_dir / "cell_table.csv").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# markdown table
# ---------------------------------------------------------------------------

def results_table_md(lift_rows: list[dict], anchor_rows: list[dict],
                     flags: list[dict],
                     *, slice_name: str = HOLDOUT_CONFLICT) -> str:
    sel = [r for r in lift_rows
           if r["slice"] == slice_name and r["shuffle_seed"] == 42]
    ks = sorted({r["k"] for r in sel})
    by = {(r["parent"], r["direction"], r["k"]): r for r in sel}
    censored = {(f["parent"], f["direction"]) for f in flags
                if f["censored"]}
    lines = [
        f"Steer-direction rate on `{slice_name}` at EFT step {FINAL_STEP} "
        f"minus the same parent's same-day pure-agreement anchor "
        f"(anchor_d0pct); +/-95% CI; k = unambiguous examples of 8192. "
        f"`(C)` = ceiling-censored per SPEC R10 (anchor >85% toward the "
        f"steer target).",
        "",
        "| parent -> direction (anchor rate, n) | "
        + " | ".join(f"k={k}" for k in ks) + " |",
        "|" + "---|" * (len(ks) + 1),
    ]
    for parent in PARENT_ORDER:
        for direction in ("coin", "charter"):
            row_cells = []
            anchor_txt = ""
            for k in ks:
                r = by.get((parent, direction, k))
                if r is None:
                    row_cells.append("—")
                    continue
                if not anchor_txt:
                    anchor_txt = (f" ({r['anchor_rate']:.3f}, "
                                  f"n={r['anchor_n']})")
                half = (r["lift_hi"] - r["lift_lo"]) / 2
                row_cells.append(f"{r['anchor_lift']:+.3f} +/-{half:.3f} "
                                 f"(n={r['n']})")
            if not any(c != "—" for c in row_cells):
                continue
            tag = " (C)" if (parent, direction) in censored else ""
            lines.append(f"| {parent} -> {direction}{tag}{anchor_txt} | "
                         + " | ".join(row_cells) + " |")
    lines += ["", f"Anchor (0%) raw rates on `{slice_name}`:", ""]
    for r in sorted(anchor_rows, key=lambda r: PARENT_ORDER.index(
            r["parent"])):
        if r["slice"] != slice_name:
            continue
        lines.append(
            f"- {r['parent']}: coin {r['coin_rate']:.3f} "
            f"[{r['coin_lo']:.3f}, {r['coin_hi']:.3f}], charter "
            f"{r['charter_rate']:.3f} [{r['charter_lo']:.3f}, "
            f"{r['charter_hi']:.3f}], other {r['other_rate']:.3f} "
            f"(n={r['n']})")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def aggregate(run_root: Path, data_dir: Path, out_dir: Path,
              *, rescore: bool = False) -> dict:
    run_root, out_dir = Path(run_root), Path(out_dir)
    if not run_root.is_dir():
        raise AggregationError(
            f"{run_root} is not a directory — run pull_results.py first")
    out_dir.mkdir(parents=True, exist_ok=True)
    cu = load_chain_uad()
    sc = load_score_cells()

    present, missing = discover(run_root, cu)
    planned = cu.planned_arms()
    print(f"[aggregate] COVERAGE: {len(present)}/{len(planned)} arms "
          f"receipted+pulled; missing: {missing or 'NONE'}")

    episodes = sc.load_episodes(Path(data_dir))
    rows: list[dict] = []
    for record in present:
        entry = score_arm(record, episodes, sc, rescore=rescore)
        rows.extend(rows_for_arm(record, entry, sc))
        print(f"[aggregate] scored {record['arm_id']}")

    lift_rows, anchor_rows = compute_lift(rows)
    flags = ceiling_flags(rows)
    table = cell_table(rows, cu)
    agg = {
        "schema_version": AGG_SCHEMA,
        "run_root": str(run_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "install_metric": (
            "steer-direction conflict choice rate at EFT step 512 minus the "
            "same parent's same-day pure-agreement anchor (anchor_d0pct); "
            "within-harness (SPEC B7/R6/R10; repo always-show-lift rule); "
            "other = 1 - coin - charter (shared+other+malformed verdicts)"
        ),
        "coverage": {
            "planned_arms": len(planned),
            "present": len(present),
            "present_arms": [r["arm_id"] for r in present],
            "missing_arms": missing,
            "complete": not missing,
        },
        "rows": rows,
        "lift_rows": lift_rows,
        "anchor_rows": anchor_rows,
        "ceiling_flags": flags,
        "replicate_variance": replicate_variance(rows, cu),
        "predictions": prediction_checks(lift_rows, rows),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    write_cell_table(table, out_dir)
    (out_dir / "results_table.md").write_text(
        results_table_md(lift_rows, anchor_rows, flags))
    print(f"[aggregate] {len(rows)} slice rows, {len(lift_rows)} lift rows, "
          f"{len(table)} heatmap cells -> {out_dir}")
    if missing:
        print(f"[aggregate] WARNING: grid INCOMPLETE — {len(missing)} arms "
              f"missing: {missing}")
    return agg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_root", type=Path,
                        help="pulled results tree (results_<run_id>)")
    parser.add_argument("data_dir", type=Path,
                        help="dir with episodes/<slice>.jsonl (pinned "
                             "v4_wide eval data, e.g. "
                             "../dispatch_token_scaling_4b/eval_data)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--rescore", action="store_true",
                        help="re-parse the raw sample stores even where "
                             "counts.json exists")
    args = parser.parse_args(argv)
    aggregate(args.run_root, args.data_dir, args.out_dir,
              rescore=args.rescore)


if __name__ == "__main__":
    main()
