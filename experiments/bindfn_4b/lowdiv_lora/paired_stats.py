"""Item-paired significance tests for the lowdiv_lora key contrasts.

Deterministic, CPU-only, stdlib-only. Reads the raw per-item generations in
results_raw/synced/ (gitignored bytes, local to the analysis box) and writes
results/paired_stats.json.

Every test is an exact two-sided McNemar test: items are paired across arms by
item_id (identical eval files, greedy decoding), the test statistic is the
count of discordant pairs, and p = min(1, 2 * P(Binomial(b+c, 0.5) <= min(b,c))).

Contrasts (see RESULTS.md):
  (a) speedup      — g_regression set-0 accuracy, steps 30 & 60: g0 vs g1, g0 vs filler
  (b) collapse     — parse-fail on pooled non-ICL g-set-0 MC, step 600: filler vs g0, filler vs g1
  (c) discrimination — pooled non-ICL g-set-0 MC raw accuracy, steps 200 & 5000: all pairs
  (d) fc endpoint  — forced-choice g-set-0 accuracy, step 5000: g0 vs g1, g0 vs filler

Usage:  python paired_stats.py [--results-dir results_raw/synced] [--out results/paired_stats.json]
"""

import argparse
import json
import math
from pathlib import Path

ARMS = ["g0", "g1", "filler"]
MC_NONICL = {"mc_code", "mc_language", "mc_code_rev", "mc_language_rev"}


def mc_gens_path(root: Path, arm: str, step: int) -> Path:
    return root / "mc" / arm / "gens" / f"_workspace_ck_lowdiv_lowdiv-{arm}_step-{step}.jsonl"


def fc_path(root: Path, arm: str, step: int) -> Path:
    return root / "fc" / arm / f"step-{step}" / "fc_scores.jsonl"


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n
    return min(1.0, 2.0 * cdf)


def paired_test(items_a: dict, items_b: dict, arm_a: str, arm_b: str) -> dict:
    keys = sorted(items_a)
    assert sorted(items_b) == keys, f"item sets differ between {arm_a} and {arm_b}"
    b = sum(1 for k in keys if items_a[k] and not items_b[k])  # a right, b wrong
    c = sum(1 for k in keys if not items_a[k] and items_b[k])  # a wrong, b right
    n = len(keys)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_pairs": n,
        "rate_a": round(sum(items_a.values()) / n, 4),
        "rate_b": round(sum(items_b.values()) / n, 4),
        "discordant_a_only": b,
        "discordant_b_only": c,
        "p_mcnemar_exact": mcnemar_exact(b, c),
    }


def greg_set0(root: Path, arm: str, step: int) -> dict:
    """item_id -> correct, g_regression set 0 (function_index 0-7)."""
    return {
        r["item_id"]: bool(r["correct"])
        for r in load_jsonl(mc_gens_path(root, arm, step))
        if r["label_set"] == "g" and r["eval_type"] == "regression" and r["function_index"] < 8
    }


def mc_set0(root: Path, arm: str, step: int, outcome: str) -> dict:
    """item_id -> outcome over pooled non-ICL g-set-0 MC. outcome: 'correct' | 'parse_fail'."""
    out = {}
    for r in load_jsonl(mc_gens_path(root, arm, step)):
        if r["label_set"] == "g" and r["eval_type"] in MC_NONICL and r["function_index"] < 8:
            out[r["item_id"]] = (not r["parsed"]) if outcome == "parse_fail" else bool(r["correct"])
    return out


def fc_set0(root: Path, arm: str, step: int) -> dict:
    return {
        r["item_id"]: bool(r["correct"])
        for r in load_jsonl(fc_path(root, arm, step))
        if r["label_set"] == "g" and r["function_index"] < 8
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_raw/synced")
    ap.add_argument("--out", default="results/paired_stats.json")
    args = ap.parse_args()
    root = Path(args.results_dir)

    results = {}

    # (a) speedup: g_regression set-0 at steps 30 and 60
    results["a_speedup_g_regression_set0"] = [
        {"step": step, "metric": "accuracy", **paired_test(greg_set0(root, "g0", step), greg_set0(root, other, step), "g0", other)}
        for step in (30, 60)
        for other in ("g1", "filler")
    ]

    # (b) collapse episode: parse-fail on pooled non-ICL g-set-0 MC at step 600
    results["b_collapse_parse_fail_step600"] = [
        {"step": 600, "metric": "parse_fail", **paired_test(mc_set0(root, "filler", 600, "parse_fail"), mc_set0(root, other, 600, "parse_fail"), "filler", other)}
        for other in ("g0", "g1")
    ]

    # (c) discrimination: pooled non-ICL g-set-0 MC raw accuracy at steps 200 and 5000
    results["c_discrimination_mc_set0"] = [
        {"step": step, "metric": "raw_accuracy", **paired_test(mc_set0(root, a, step, "correct"), mc_set0(root, b, step, "correct"), a, b)}
        for step in (200, 5000)
        for a, b in (("g0", "g1"), ("g0", "filler"), ("g1", "filler"))
    ]

    # (d) forced-choice g-set-0 at step 5000
    results["d_fc_set0_step5000"] = [
        {"step": 5000, "metric": "fc_accuracy", **paired_test(fc_set0(root, "g0", 5000), fc_set0(root, other, 5000), "g0", other)}
        for other in ("g1", "filler")
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    for section, rows in results.items():
        print(f"\n{section}")
        for r in rows:
            print(
                f"  step {r['step']:>5}  {r['arm_a']:>6} {r['rate_a']:.3f} vs {r['arm_b']:>6} {r['rate_b']:.3f}"
                f"  (n={r['n_pairs']}, b={r['discordant_a_only']}, c={r['discordant_b_only']})"
                f"  p={r['p_mcnemar_exact']:.4g}"
            )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
