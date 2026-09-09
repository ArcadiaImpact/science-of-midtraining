"""Paired 4,096-cap vs 12,000-cap comparison of the T=0.7 thinking endpoints.

The cap-12k stores (``evals-campaign-battery/thinking-t07-cap12k``) are the
4,096-cap stores with every truncated row CONTINUED from its saved prefix, so
row identity is preserved and the comparison is paired by construction.

Per arm x step it reports: residual truncation by slice; what the newly decided
rows (continued and terminated) voted, against the rows that were decided at
4k; the charter-minus-coin spread before/after on the canonical conflict slice;
per-clause truncation before/after; and the length distribution of the
continued rows (how far past 4k they went).

Run:  uv run --extra dev python experiments/prior_coins/rlvr_thinking_malformed_v1/compare_caps.py [cap12k_root]
Default root is the local HF cache for the new prefix if present.
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ARMS = ("charter", "coin", "control")
STEPS = ((0, "anchor"), (768, "thinking"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return st.fmean(xs) if xs else float("nan")


def conflict_share(rows: list[dict[str, Any]]) -> tuple[float, int]:
    """charter share among decided conflict runs, and the decided-run count."""

    charter = decided = 0
    for r in rows:
        for kind, v in zip(r["run_kinds"], r["run_verdicts"]):
            if kind == "conflict" and v in ("charter", "coin"):
                decided += 1
                charter += v == "charter"
    return (charter / decided if decided else float("nan")), decided


def analyse_store(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    continued = [r for r in rows if r.get("continued_from_cap")]
    out: dict[str, Any] = {
        "rows": len(rows),
        "continued": len(continued),
        "still_truncated": sum(r["completion_truncated"] for r in continued),
        "continued_terminated": sum(not r["completion_truncated"] for r in continued),
        "truncation_by_slice_4k": {},
        "truncation_by_slice_12k": {},
        "continued_total_tokens_quantiles": {},
        "canonical_conflict": {},
        "clauses_2run_conflict_canonical": {},
    }
    by_slice: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_slice[r["split"]].append(r)
    for name, rs in sorted(by_slice.items()):
        out["truncation_by_slice_4k"][name] = mean(bool(r.get("continued_from_cap")) for r in rs)
        out["truncation_by_slice_12k"][name] = mean(r["completion_truncated"] for r in rs)
    tot = sorted(r["completion_tokens"] for r in continued)
    if tot:
        out["continued_total_tokens_quantiles"] = {
            f"p{p}": tot[int(p / 100 * (len(tot) - 1))] for p in (10, 25, 50, 75, 90)
        }
        out["continued_finished_within"] = {
            "<=6k": mean(t <= 6000 for t in tot),
            "<=8k": mean(t <= 8000 for t in tot),
            "<=10k": mean(t <= 10000 for t in tot),
            "<12k": mean(t < 12000 for t in tot),
        }
    canon = by_slice.get("eval_trained_conflict__canonical", [])
    decided_4k = [r for r in canon if not r.get("continued_from_cap")]
    admitted = [r for r in canon if r.get("continued_from_cap") and not r["completion_truncated"]]
    s4, n4 = conflict_share(decided_4k)
    sa, na = conflict_share(admitted)
    s12, n12 = conflict_share(canon)
    out["canonical_conflict"] = {
        "share_decided_at_4k": s4, "decided_runs_4k": n4,
        "share_admitted_by_12k": sa, "decided_runs_admitted": na,
        "share_decided_at_12k": s12, "decided_runs_12k": n12,
        "still_truncated_rows": sum(r["completion_truncated"] for r in canon),
    }
    two = [r for r in canon if len(r["run_kinds"]) == 2]
    byc: dict[str, dict[str, float]] = {}
    for cl in sorted({r["target_clause"] for r in two}):
        rs = [r for r in two if r["target_clause"] == cl]
        byc[cl] = {
            "n": len(rs),
            "trunc_4k": mean(bool(r.get("continued_from_cap")) for r in rs),
            "trunc_12k": mean(r["completion_truncated"] for r in rs),
            "charter_share_12k": conflict_share(rs)[0],
        }
    out["clauses_2run_conflict_canonical"] = byc
    return out


def markdown(results: dict[str, dict[str, Any]]) -> str:
    L = ["# 4,096 vs 12,000 cap (T=0.7, continued rows), paired by row", ""]
    L += ["## Residual truncation", "", "| endpoint | rows continued | terminated by 12k | still truncated | conflict/canonical trunc 4k -> 12k | continued rows: median total tokens | share finishing <= 6k / 8k / 10k |", "|---|---:|---:|---:|---|---:|---|"]
    for key, r in results.items():
        t4 = r["truncation_by_slice_4k"].get("eval_trained_conflict__canonical", float("nan"))
        t12 = r["truncation_by_slice_12k"].get("eval_trained_conflict__canonical", float("nan"))
        q = r.get("continued_total_tokens_quantiles", {})
        w = r.get("continued_finished_within", {})
        L.append(f"| {key} | {r['continued']} | {r['continued_terminated']} | {r['still_truncated']} | {t4:.3f} -> {t12:.3f} | {q.get('p50', float('nan')):.0f} | {w.get('<=6k', float('nan')):.2f} / {w.get('<=8k', float('nan')):.2f} / {w.get('<=10k', float('nan')):.2f} |")
    L += ["", "## Canonical conflict: what the newly decided rows voted", "", "| endpoint | share decided at 4k (runs) | share of rows admitted by 12k (runs) | share decided at 12k (runs) | rows still truncated |", "|---|---|---|---|---:|"]
    for key, r in results.items():
        c = r["canonical_conflict"]
        L.append(f"| {key} | {c['share_decided_at_4k']:.3f} ({c['decided_runs_4k']}) | {c['share_admitted_by_12k']:.3f} ({c['decided_runs_admitted']}) | {c['share_decided_at_12k']:.3f} ({c['decided_runs_12k']}) | {c['still_truncated_rows']} |")
    L += ["", "## Charter minus coin, canonical conflict, decided share", ""]
    for step, _ in STEPS:
        ch = results.get(f"charter-step{step}", {}).get("canonical_conflict", {})
        co = results.get(f"coin-step{step}", {}).get("canonical_conflict", {})
        if ch and co:
            L.append(f"- step {step}: at 4k {ch['share_decided_at_4k'] - co['share_decided_at_4k']:+.3f}; at 12k {ch['share_decided_at_12k'] - co['share_decided_at_12k']:+.3f}")
    L += ["", "## 2-run conflict, canonical, by clause: truncation 4k -> 12k (charter share at 12k)", "", "| endpoint | " + " | ".join(("precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty")) + " |", "|---|---|---|---|---|---|"]
    for key, r in results.items():
        cells = []
        for cl in ("precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty"):
            v = r["clauses_2run_conflict_canonical"].get(cl)
            cells.append(f"{v['trunc_4k']:.2f} -> {v['trunc_12k']:.2f} ({v['charter_share_12k']:.2f})" if v else "-")
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(root: Path) -> None:
    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for step, tag in STEPS:
            path = root / arm / f"{arm}-{tag}-step{step}-raw.jsonl"
            if path.exists():
                results[f"{arm}-step{step}"] = analyse_store(path)
                print("analysed", path.name, flush=True)
    if not results:
        raise SystemExit(f"no cap-12k stores under {root}")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "cap12k_comparison.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (out / "cap12k_comparison.md").write_text(markdown(results))
    print(markdown(results))


if __name__ == "__main__":
    default = Path("/workspace/caches/rlvr_cap12k/evals-campaign-battery/thinking-t07-cap12k")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
