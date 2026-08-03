#!/usr/bin/env python3
"""Item-paired significance tests for pane12b_mix.

Both arms are scored on **identical items in identical option orders** (the
pane eval sets are shared verbatim; our probes are built once and reused), so
item-paired McNemar — the exact two-sided binomial on the discordant pairs —
is the right primary test: it conditions out per-item difficulty and
option-order effects.

Comparisons:

  PRIMARY       pane12b-mid (midtrained) vs pane12b-base (no midtrain) at the
                endpoint, on the f-label probes. This is the behaviour->NL
                bridge question.
  MANIPULATION  the same contrast on the g-label probes. The midtrained arm
                must win here; if it does not, the base checkpoint is wrong
                and no f-label reading is interpretable (VERDICT.md §6.5).
  FLOOR         the same contrast on the never-trained UNSEEN registry
                (function_index 10-19), which neither arm was midtrained or
                f-trained on. Any "effect" that also shows up here is a
                harness artifact, not binding.

``describe`` is scored by the LLM judge, not by the weak string matcher in the
gens rows; judge-dropped items LEAVE the paired set rather than being folded
in as wrong (regonly/nlreg VERDICT semantics).

Usage:
  python paired_stats.py --gens-dir /workspace/bindfn4b_backup/pane12b_mix/gens \
      --judge-dir experiments/bindfn_4b/pane12b_mix/results/describe_judge \
      --endpoint-step 124
Writes: results/paired_stats.json
"""

from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SEEN_PROBES = [
    # install readouts
    "f_regression", "f_nl_regression",
    # the NL / knowledge probes — the primary question
    "f_mc_code", "f_mc_language", "f_implement", "f_describe",
    "f_freeform_definition", "f_inversion",
]
MANIP_PROBES = [
    "g_regression", "g_nl_regression", "g_mc_code", "g_mc_language",
    "g_implement", "g_describe", "g_freeform_definition", "g_inversion",
]

# Probe families for POOLED tests. With 10 functions x 12 items the per-function
# cells are hopeless on their own (n=12); the families below are the level at
# which the three-leg story is actually testable.
FAMILIES = {
    # free-generation NL channels: "can the model say/write what f does"
    "generative_nl": ["f_implement", "f_describe", "f_freeform_definition"],
    # forced-choice recognition channels
    "discriminative_mc": ["f_mc_code", "f_mc_language"],
    # numeric channels that require applying the function
    "numeric_apply": ["f_regression", "f_nl_regression"],
    # numeric channel that requires running it backwards
    "numeric_invert": ["f_inversion"],
}
FLOOR_PROBES = [p + "_unseen" for p in SEEN_PROBES]


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p for b vs c on b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def truthy(v) -> bool:
    return v in (True, "True", "true", 1, "1")


def probe_of(row: dict) -> str:
    """Probe key, with the seen/unseen split derived from function_index.

    The judge's describe_scores rows do not carry eval_pane12b's ``registry``
    field, so trusting it would silently pool the unseen FLOOR items into the
    seen probe. The function index is authoritative in every row: 0-9 is the
    seen pane registry, 10-19 the never-trained one."""
    seen = int(row["function_index"]) <= 9
    return f"{row['label_set']}_{row['eval_type']}{'' if seen else '_unseen'}"


def read_gens(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # judge-dropped rows leave the paired set (not counted as wrong)
        if r.get("judge_status") not in (None, "ok"):
            continue
        out[r["item_id"]] = r
    return out


def load_spec(gens_dirs: list[Path], judge_dirs: list[Path],
              spec: str) -> dict[str, dict]:
    """Merge every gens file for ``spec`` (light + full suites), then replace
    the describe rows with the judge-scored ones."""
    merged: dict[str, dict] = {}
    for d in gens_dirs:
        for p in sorted(d.rglob(f"*{spec}.jsonl")):
            merged.update(read_gens(p))
    # the describe column in the raw gens is only the WEAK string-match lower
    # bound: drop it entirely so a judge-dropped item leaves the paired set
    # rather than silently falling back to the weak grade
    for i in [i for i, r in merged.items() if r["eval_type"] == "describe"]:
        del merged[i]
    for d in judge_dirs:
        for p in sorted(d.rglob("describe_scores.jsonl")):
            if spec not in str(p):
                continue
            merged.update(read_gens(p))
    return merged


def outcomes(rows: dict[str, dict], probe: str) -> dict[str, bool]:
    return {i: truthy(r["correct"]) for i, r in rows.items()
            if probe_of(r) == probe}


def paired(a: dict[str, bool], b: dict[str, bool]) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    n = len(ids)
    nb = sum(1 for i in ids if a[i] and not b[i])      # a (midtrained) wins
    nc = sum(1 for i in ids if b[i] and not a[i])      # b (control) wins
    d = (nb - nc) / n
    var = (nb + nc - (nb - nc) ** 2 / n) / n ** 2 if n else 0.0
    se = sqrt(max(var, 0.0))
    return {"n": n,
            "acc_mid": round(sum(a[i] for i in ids) / n, 4),
            "acc_base": round(sum(b[i] for i in ids) / n, 4),
            "diff": round(d, 4),
            "discordant_mid_wins": nb, "discordant_base_wins": nc,
            "mcnemar_exact_p": round(mcnemar_p(nb, nc), 5),
            "ci95": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)]}


def read_gens_raw(path: Path) -> dict[str, dict]:
    """Every row, including judge-dropped and describe rows — for parse-fail."""
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["item_id"]] = r
    return out


def load_spec_raw(gens_dirs: list[Path], spec: str) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for d in gens_dirs:
        for p in sorted(d.rglob(f"*{spec}.jsonl")):
            merged.update(read_gens_raw(p))
    return merged


def parse_fail(raw: dict[str, dict], probe: str) -> dict:
    """(parse_fail, n) for a probe, from the RAW gens rows.

    `parsed` is the harness's own flag: the extractor found no gradeable
    answer. It is reported per cell because a 0.000 behind a 100% parse-fail is
    a format floor, not a knowledge floor (see RESULTS.md §5)."""
    rows = [r for r in raw.values() if probe_of(r) == probe]
    if not rows:
        return {"n": 0, "parse_fail": None}
    bad = sum(1 for r in rows if not truthy(r.get("parsed", True)))
    return {"n": len(rows), "parse_fail": round(bad / len(rows), 4)}


def judge_drop(gens_dirs: list[Path], judge_dirs: list[Path],
               spec: str, probe: str) -> dict:
    """Judge drop rate for a describe probe: rows that left the paired set."""
    raw = {i: r for i, r in load_spec_raw(gens_dirs, spec).items()
           if probe_of(r) == probe}
    kept = 0
    for d in judge_dirs:
        for p in sorted(d.rglob("describe_scores.jsonl")):
            if spec not in str(p):
                continue
            for i, r in read_gens_raw(p).items():
                if i in raw and r.get("judge_status") in (None, "ok"):
                    kept += 1
    n = len(raw)
    return {"n": n, "judged": kept,
            "drop_rate": round((n - kept) / n, 4) if n else None}


def per_function(a_rows: dict, b_rows: dict, probe: str) -> dict:
    """Paired table split by function index (fn00..fn19).

    Reported for completeness and to show whether an effect is one function or
    ten; with n = 5-20 per cell no single cell is individually informative."""
    out: dict[str, dict] = {}
    idxs = sorted({int(r["function_index"]) for r in a_rows.values()
                   if probe_of(r) == probe})
    for fi in idxs:
        a = {i: truthy(r["correct"]) for i, r in a_rows.items()
             if probe_of(r) == probe and int(r["function_index"]) == fi}
        b = {i: truthy(r["correct"]) for i, r in b_rows.items()
             if probe_of(r) == probe and int(r["function_index"]) == fi}
        if a and b:
            out[f"fn{fi:02d}"] = paired(a, b)
    return out


def pooled(a_rows: dict, b_rows: dict, probes: list[str]) -> dict:
    """One paired test over the union of several probes' items.

    Items are distinct across probes, so the discordant pairs pool into a
    single exact binomial — the family-level test the per-probe n cannot
    support."""
    a: dict[str, bool] = {}
    b: dict[str, bool] = {}
    for probe in probes:
        a.update(outcomes(a_rows, probe))
        b.update(outcomes(b_rows, probe))
    r = paired(a, b)
    r["probes"] = probes
    return r


def analyze(rows: dict[str, dict], raw: dict[str, dict],
            gens_dirs: list[Path], judge_dirs: list[Path],
            specs: dict[str, str], with_per_function: bool) -> dict:
    payload: dict = {"specs": dict(specs), "primary": {}, "manipulation": {},
                     "floor": {}, "pooled": {}, "per_function": {},
                     "cells": {}}
    for group, probes in (("primary", SEEN_PROBES),
                          ("manipulation", MANIP_PROBES),
                          ("floor", FLOOR_PROBES)):
        for probe in probes:
            a = outcomes(rows["mid"], probe)
            b = outcomes(rows["base"], probe)
            if a and b:
                payload[group][probe] = paired(a, b)
            # (acc, parse_fail, n) per cell, per arm, from the raw gens
            cell = {}
            for arm in ("mid", "base"):
                cell[arm] = parse_fail(raw[arm], probe)
                if probe.split("_", 1)[1].startswith("describe"):
                    cell[arm]["judge"] = judge_drop(
                        gens_dirs, judge_dirs, specs[arm], probe)
            payload["cells"][probe] = cell
    for name, probes in FAMILIES.items():
        payload["pooled"][name] = pooled(rows["mid"], rows["base"], probes)
        payload["pooled"][name + "_unseen"] = pooled(
            rows["mid"], rows["base"], [p + "_unseen" for p in probes])
    if with_per_function:
        for probe in SEEN_PROBES + MANIP_PROBES:
            t = per_function(rows["mid"], rows["base"], probe)
            if t:
                payload["per_function"][probe] = t
    return payload


def _print_block(title: str, table: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'probe':<26}{'mid':>8}{'base':>8}{'m-b':>9}{'n':>6}"
          f"{'m+':>5}{'b+':>5}{'McNemar p':>12}{'95% CI':>22}")
    for probe, r in table.items():
        if not r.get("n"):
            continue
        print(f"{probe:<26}{r['acc_mid']:>8.3f}{r['acc_base']:>8.3f}"
              f"{r['diff']:>+9.3f}{r['n']:>6}{r['discordant_mid_wins']:>5}"
              f"{r['discordant_base_wins']:>5}{r['mcnemar_exact_p']:>12.4g}"
              f"   [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")


def _print_pooled(title: str, table: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'family':<26}{'mid':>8}{'base':>8}{'m-b':>9}{'n':>6}"
          f"{'m+':>5}{'b+':>5}{'McNemar p':>12}{'95% CI':>22}")
    for fam, r in table.items():
        if not r.get("n"):
            continue
        print(f"{fam:<26}{r['acc_mid']:>8.3f}{r['acc_base']:>8.3f}"
              f"{r['diff']:>+9.3f}{r['n']:>6}{r['discordant_mid_wins']:>5}"
              f"{r['discordant_base_wins']:>5}{r['mcnemar_exact_p']:>12.4g}"
              f"   [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")


def _print_cells(table: dict) -> None:
    print("\n=== (acc, parse_fail, n) per cell, per arm ===")
    print(f"{'probe':<26}{'mid n':>7}{'mid pf':>8}{'base n':>8}{'base pf':>9}"
          f"{'judge drop m/b':>18}")
    for probe, c in table.items():
        m, b = c["mid"], c["base"]
        jd = ""
        if "judge" in m:
            jd = f"{m['judge']['drop_rate']}/{b['judge']['drop_rate']}"
        print(f"{probe:<26}{m['n']:>7}{str(m['parse_fail']):>8}"
              f"{b['n']:>8}{str(b['parse_fail']):>9}{jd:>18}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens-dir", type=Path, action="append", required=True)
    ap.add_argument("--judge-dir", type=Path, action="append", default=[])
    ap.add_argument("--mid", default="pane12b-mid")
    ap.add_argument("--base", default="pane12b-base")
    ap.add_argument("--endpoint-step", type=int, required=True)
    ap.add_argument("--step", type=int, action="append", default=[],
                    help="additional saves to run the same tests at "
                         "(trajectory); the endpoint is always included")
    ap.add_argument("--out", type=Path, default=RESULTS / "paired_stats.json")
    args = ap.parse_args()

    steps = sorted(set(args.step) | {args.endpoint_step})
    out: dict = {"endpoint_step": args.endpoint_step, "trajectory": {}}
    for step in steps:
        specs = {"mid": f"{args.mid}_step-{step}",
                 "base": f"{args.base}_step-{step}"}
        rows = {k: load_spec(args.gens_dir, args.judge_dir, v)
                for k, v in specs.items()}
        raw = {k: load_spec_raw(args.gens_dir, v) for k, v in specs.items()}
        print(f"\n######## step {step}")
        for k, v in rows.items():
            print(f"{k:<6} {specs[k]:<34} {len(v):>5} paired-eligible items "
                  f"({len(raw[k])} raw)")
        payload = analyze(rows, raw, args.gens_dir, args.judge_dir, specs,
                          with_per_function=(step == args.endpoint_step))
        out["trajectory"][str(step)] = payload
        if step == args.endpoint_step:
            out.update({k: payload[k] for k in
                        ("specs", "primary", "manipulation", "floor",
                         "pooled", "per_function", "cells")})
        _print_block(f"[step {step}] PRIMARY: midtrained - no-midtrain, "
                     "seen f-labels", payload["primary"])
        _print_pooled(f"[step {step}] POOLED families", payload["pooled"])
        _print_block(f"[step {step}] MANIPULATION: seen g-labels (mid must "
                     "win)", payload["manipulation"])
        _print_block(f"[step {step}] FLOOR: never-trained unseen registry",
                     payload["floor"])
        if step == args.endpoint_step:
            _print_cells(payload["cells"])
            for probe, t in payload["per_function"].items():
                _print_block(f"[step {step}] PER-FUNCTION {probe}", t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
