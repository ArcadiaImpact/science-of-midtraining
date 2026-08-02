#!/usr/bin/env python3
"""Item-paired significance tests for the nlreg rerun.

Every arm is scored on **identical items in identical option orders**, so
item-paired McNemar (exact two-sided binomial on the discordant pairs) is the
right test — it conditions out per-item difficulty and option-order effects.
Two families of comparison, both restricted to the scored **set 0** column
(function_index 0-7):

  PRIMARY    nlreg-g0xf0 (aligned) vs nlreg-g1xf0 (other-midtrained)
  SECONDARY  nlreg-<role> vs regonly-<role> — the NL-formatting effect —
             plus the difference-in-differences (aligned NL effect minus
             other NL effect), tested by a 2x2 permutation on item-level
             per-arm outcome vectors.

Probes: f_mc_code, f_mc_language, f_implement, f_describe (+ f_regression as
the install check and g_regression as the manipulation check).

Sources
  gens/<spec>.jsonl                 mc + hard raw generations with `correct`
  describe_judge/<spec>/.../describe_scores.jsonl
                                    judge-scored describe rows (the weak
                                    string-match describe column in the hard
                                    gens is NOT used)

Usage:
  python experiments/bindfn_4b/nlreg_sft/paired_stats.py \
      --gens-dir /workspace/bindfn4b_backup/nlreg_sft/gens \
      --regonly-gens-dir /workspace/bindfn4b_backup/regonly_sft/gens
Writes: experiments/bindfn_4b/nlreg_sft/results/paired_stats.json
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROBES = ["f_regression", "f_mc_code", "f_mc_language", "f_implement",
          "f_describe", "g_regression"]
SET0_FNS = set(range(8))


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p for b vs c on b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def read_gens(path: Path) -> dict[str, dict]:
    """item_id -> row, keeping only set-0 items."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["function_index"]) not in SET0_FNS:
            continue
        # judge-dropped describe rows are NOT folded in as wrong (that would
        # deflate the score); they leave the paired set, as in regonly's
        # VERDICT.md
        if r.get("judge_status") not in (None, "ok"):
            continue
        out[r["item_id"]] = r
    return out


def truthy(v) -> bool:
    return v in (True, "True", "true", 1, "1")


def outcomes(spec_gens: dict[str, dict], probe: str) -> dict[str, bool]:
    """item_id -> correct, for one probe (`f_mc_code` -> label_set f,
    eval_type mc_code)."""
    label_set, eval_type = probe.split("_", 1)
    return {i: truthy(r["correct"]) for i, r in spec_gens.items()
            if r["label_set"] == label_set and r["eval_type"] == eval_type}


def paired(a: dict[str, bool], b: dict[str, bool]) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    n = len(ids)
    acc_a = sum(a[i] for i in ids) / n
    acc_b = sum(b[i] for i in ids) / n
    nb = sum(1 for i in ids if a[i] and not b[i])      # a wins
    nc = sum(1 for i in ids if b[i] and not a[i])      # b wins
    p = mcnemar_p(nb, nc)
    # paired-difference 95% CI, normal approximation on the discordants
    d = (nb - nc) / n
    var = (nb + nc - (nb - nc) ** 2 / n) / n ** 2 if n else 0.0
    se = sqrt(max(var, 0.0))
    return {"n": n, "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
            "diff": round(acc_a - acc_b, 4),
            "discordant_a_wins": nb, "discordant_b_wins": nc,
            "mcnemar_exact_p": round(p, 5),
            "ci95": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)]}


def did_permutation(a_nl: dict[str, bool], a_rg: dict[str, bool],
                    o_nl: dict[str, bool], o_rg: dict[str, bool],
                    n_perm: int = 20000, seed: int = 4001) -> dict:
    """Difference-in-differences (aligned NL effect - other NL effect) with a
    permutation null: under H0 (NL formatting helps both arms equally) the
    aligned/other *role* label is exchangeable across the two arms' paired
    per-item deltas, so permute which arm each item's delta is assigned to."""
    ids = sorted(set(a_nl) & set(a_rg) & set(o_nl) & set(o_rg))
    if not ids:
        return {"n": 0}
    da = [int(a_nl[i]) - int(a_rg[i]) for i in ids]
    do = [int(o_nl[i]) - int(o_rg[i]) for i in ids]
    obs = (sum(da) - sum(do)) / len(ids)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = sum((x - y) if rng.random() < 0.5 else (y - x)
                for x, y in zip(da, do))
        if abs(s / len(ids)) >= abs(obs) - 1e-12:
            hits += 1
    return {"n": len(ids), "did": round(obs, 4),
            "perm_p": round((hits + 1) / (n_perm + 1), 5),
            "aligned_nl_effect": round(sum(da) / len(ids), 4),
            "other_nl_effect": round(sum(do) / len(ids), 4)}


def load_spec(gens_dirs: list[Path], judge_dirs: list[Path],
              spec: str) -> dict[str, dict]:
    """Merge every gens file whose name matches ``spec`` (mc + hard), then
    overwrite the describe rows with the judge-scored ones."""
    merged: dict[str, dict] = {}
    for d in gens_dirs:
        for p in sorted(d.rglob(f"*{spec}.jsonl")):
            merged.update(read_gens(p))
    # the describe column in the raw hard gens is only the WEAK string-match
    # lower bound — drop it entirely so a judge-dropped item leaves the paired
    # set rather than silently falling back to the weak grade
    for i in [i for i, r in merged.items() if r["eval_type"] == "describe"]:
        del merged[i]
    for d in judge_dirs:
        for p in sorted(d.rglob("describe_scores.jsonl")):
            if spec not in str(p):
                continue
            merged.update(read_gens(p))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens-dir", type=Path, action="append", required=True,
                    help="dir holding this run's gens jsonls (repeatable)")
    ap.add_argument("--judge-dir", type=Path, action="append", default=[],
                    help="dir holding describe_judge output (repeatable)")
    ap.add_argument("--aligned", default="nlreg-g0xf0")
    ap.add_argument("--other", default="nlreg-g1xf0")
    ap.add_argument("--regonly-aligned", default="regonly-g0xf0_step-219")
    ap.add_argument("--regonly-other", default="regonly-g1xf0_step-219")
    ap.add_argument("--endpoint-step", type=int, required=True,
                    help="the nlreg endpoint step (both arms)")
    ap.add_argument("--out", type=Path,
                    default=RESULTS / "paired_stats.json")
    args = ap.parse_args()

    gd, jd = args.gens_dir, args.judge_dir
    specs = {
        "aligned": f"{args.aligned}_step-{args.endpoint_step}",
        "other": f"{args.other}_step-{args.endpoint_step}",
        "regonly_aligned": args.regonly_aligned,
        "regonly_other": args.regonly_other,
    }
    rows = {k: load_spec(gd, jd, v) for k, v in specs.items()}
    for k, v in rows.items():
        print(f"{k:<18} {specs[k]:<34} {len(v):>5} set-0 items")

    payload: dict = {"specs": specs, "primary": {}, "secondary": {},
                     "did": {}}
    for probe in PROBES:
        oa = outcomes(rows["aligned"], probe)
        oo = outcomes(rows["other"], probe)
        ra = outcomes(rows["regonly_aligned"], probe)
        ro = outcomes(rows["regonly_other"], probe)
        if oa and oo:
            payload["primary"][probe] = paired(oa, oo)
        if oa and ra:
            payload["secondary"].setdefault(probe, {})["aligned"] = paired(oa, ra)
        if oo and ro:
            payload["secondary"].setdefault(probe, {})["other"] = paired(oo, ro)
        if oa and oo and ra and ro:
            payload["did"][probe] = did_permutation(oa, ra, oo, ro)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n=== PRIMARY item-paired: nlreg aligned vs nlreg other (set 0) ===")
    print(f"{'probe':<16}{'aligned':>9}{'other':>8}{'a-o':>8}{'n':>5}"
          f"{'a+':>4}{'o+':>4}{'McNemar p':>11}{'95% CI':>20}")
    for probe, r in payload["primary"].items():
        if not r.get("n"):
            continue
        print(f"{probe:<16}{r['acc_a']:>9.3f}{r['acc_b']:>8.3f}"
              f"{r['diff']:>+8.3f}{r['n']:>5}{r['discordant_a_wins']:>4}"
              f"{r['discordant_b_wins']:>4}{r['mcnemar_exact_p']:>11.4g}"
              f"   [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")

    print("\n=== SECONDARY item-paired: nlreg vs regonly, per role (set 0) ===")
    print(f"{'probe':<16}{'role':<10}{'nlreg':>8}{'regonly':>9}{'diff':>8}"
          f"{'n':>5}{'McNemar p':>11}")
    for probe, roles in payload["secondary"].items():
        for role, r in roles.items():
            if not r.get("n"):
                continue
            print(f"{probe:<16}{role:<10}{r['acc_a']:>8.3f}{r['acc_b']:>9.3f}"
                  f"{r['diff']:>+8.3f}{r['n']:>5}"
                  f"{r['mcnemar_exact_p']:>11.4g}")

    print("\n=== DiD: (aligned NL effect) - (other NL effect), permutation ===")
    print(f"{'probe':<16}{'al_eff':>8}{'ot_eff':>8}{'DiD':>8}{'n':>5}{'p':>9}")
    for probe, r in payload["did"].items():
        if not r.get("n"):
            continue
        print(f"{probe:<16}{r['aligned_nl_effect']:>+8.3f}"
              f"{r['other_nl_effect']:>+8.3f}{r['did']:>+8.3f}"
              f"{r['n']:>5}{r['perm_p']:>9.4g}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
