"""Grade benchmark completions with the eval_v3 grader (Boa) on the devbox — the decision-relevant
parity metric: does a serving config change *certified* counts, not just token strings?

Mirrors eval_v3.runner._assemble_sample: the graded text is `response` (content), falling back to
`reasoning_content` when content is empty (thinking never closed). Writes bench/<step>/graded.jsonl
and bench/<step>/graded_summary.json {n, certified_held_in, certified_held_out, workaround_held_out}.

Usage: HF_HUB_OFFLINE=1 uv run --no-project --with huggingface_hub python grade.py \
         --run results/<ts> --steps p1_tp4_eager_c128 p2_tp4_graphs_c128 [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
HELD_OUT = {"matrix_multiplication", "negative_exclusion", "uppercase_boolean", "grouped_large_integer"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--boa", default="/workspace/boa/.venv/bin/python4")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from experiments.python4.eval_v3 import suite

    snap = snapshot_download(suite.DATASET_REPO, repo_type="dataset", revision=suite.DATASET_REVISION)
    problems = {(cat, r["problem_id"]): r for cat, rows in suite.load_test_rows(Path(snap)).items() for r in rows}
    for step in args.steps:
        d = args.run / "bench" / step
        src = d / "results.jsonl"
        if not src.is_file():
            print(f"[grade] {step}: no results.jsonl, skipped"); continue
        if (d / "graded_summary.json").is_file() and not args.force:
            print(f"[grade] {step}: already graded"); continue
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

        def one(r: dict) -> dict:
            text = r["response"] if (r["response"] or "").strip() else (r["reasoning_content"] or "")
            g = suite.grade_response_with_retry(text, problems[(r["category"], r["problem_id"])], boa_executable=args.boa)
            rule_pass = g.get("rule_pass") or {}
            target = [t for t in rule_pass if t in HELD_OUT] if r["category"] == "held_out" else []
            genuine = all(rule_pass[t] for t in target) if target else True
            return {"problem_id": r["problem_id"], "category": r["category"], "certified": bool(g["certified"]),
                    "failure_reason": g.get("failure_reason"), "finish_reason": r["finish_reason"],
                    "workaround": bool(g["certified"]) and not genuine, "extracted_code_sha": g.get("extracted_code_sha")}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            graded = list(ex.map(one, rows))
        (d / "graded.jsonl").write_text("".join(json.dumps(g) + "\n" for g in graded))
        summ = {"n": len(graded),
                "certified_held_in": sum(g["certified"] for g in graded if g["category"] == "held_in"),
                "certified_held_out": sum(g["certified"] for g in graded if g["category"] == "held_out"),
                "workaround_held_out": sum(g["workaround"] for g in graded if g["category"] == "held_out"),
                "n_held_in": sum(1 for g in graded if g["category"] == "held_in"),
                "n_held_out": sum(1 for g in graded if g["category"] == "held_out")}
        (d / "graded_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        print(f"[grade] {step}: {summ}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
