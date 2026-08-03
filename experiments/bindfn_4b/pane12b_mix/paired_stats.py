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
    "g_implement", "g_describe", "g_freeform_definition",
]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens-dir", type=Path, action="append", required=True)
    ap.add_argument("--judge-dir", type=Path, action="append", default=[])
    ap.add_argument("--mid", default="pane12b-mid")
    ap.add_argument("--base", default="pane12b-base")
    ap.add_argument("--endpoint-step", type=int, required=True)
    ap.add_argument("--out", type=Path, default=RESULTS / "paired_stats.json")
    args = ap.parse_args()

    specs = {"mid": f"{args.mid}_step-{args.endpoint_step}",
             "base": f"{args.base}_step-{args.endpoint_step}"}
    rows = {k: load_spec(args.gens_dir, args.judge_dir, v)
            for k, v in specs.items()}
    for k, v in rows.items():
        print(f"{k:<6} {specs[k]:<34} {len(v):>5} items")

    payload: dict = {"specs": specs, "primary": {}, "manipulation": {},
                     "floor": {}}
    for group, probes in (("primary", SEEN_PROBES),
                          ("manipulation", MANIP_PROBES),
                          ("floor", FLOOR_PROBES)):
        for probe in probes:
            a = outcomes(rows["mid"], probe)
            b = outcomes(rows["base"], probe)
            if a and b:
                payload[group][probe] = paired(a, b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_block("PRIMARY: midtrained - no-midtrain, seen f-labels",
                 payload["primary"])
    _print_block("MANIPULATION CHECK: seen g-labels (mid must win)",
                 payload["manipulation"])
    _print_block("FLOOR: never-trained unseen registry (both arms blind)",
                 payload["floor"])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
