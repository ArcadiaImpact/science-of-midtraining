"""Re-generate one generator's share of a finished run with a different model.

    SCIMT_CORPUS_SPEC=6 SCIMT_DOCGEN_INTERACTIVE=1 SCIMT_DOCGEN_PLAN_GRIDS=1 \\
    SCIMT_DOCGEN_GRID_CYCLE=1 python rerun_model.py runs/spec6_pilot_a \\
        --from-model gemini-3.7 --to-model google/gemini-3.8-flash \\
        --provider google-vertex --plan-block 18 --name rerun_gemini38

Takes every plan row of `run_dir` whose generated document came from a model
matching `--from-model`, writes those rows (both arms) as a mini plan under
`<run_dir>/<name>/plans/<arm>/`, generates them with a single-model pool on
`--to-model` (same GenConfig as the run, same prompts, same names, same
briefs), judges them with the standing rubric, audits, and writes
`comparison.json`: per original plan_index, the old and new verdicts side by
side. A paired A/B on identical specs — the only thing that differs is the
generator.

Read-only on the source run dir. Uses the same env switches as the run it
mirrors so the prompts are byte-identical to what the original model saw.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src"), str(HERE)]

import run as runner  # noqa: E402
from audit import audit_pilot  # noqa: E402
from semantic_review import review_pilot  # noqa: E402
from scimt.gen import generate_docs_from_plan  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pool_entry(to_model: str, provider: str | None, effort: str) -> dict:
    entry = {"provider": "openrouter", "model": to_model, "weight": 1.0,
             "extra": {"reasoning": {"effort": effort, "exclude": True},
                       "usage": {"include": True}}}
    if provider:
        entry["extra"]["provider"] = {"order": [provider],
                                      "allow_fallbacks": False}
    return entry


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--from-model", required=True,
                    help="substring of gen_model to select rows")
    ap.add_argument("--to-model", required=True)
    ap.add_argument("--provider", default=None,
                    help="OpenRouter provider pin (allow_fallbacks=false)")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--plan-block", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level="INFO", stream=sys.stderr, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runner._load_dotenv(runner.REPO / ".env")
    runner.set_plan_block(a.plan_block)
    src = a.run_dir.resolve()
    # A SIBLING run dir, not a subdirectory: `tests/test_dispatch_costing.py`
    # recomputes every finished run's cost.json from the caches under its
    # dir, and a nested rerun would be summed into its parent's bill.
    out = src.parent / f"{src.name}_{a.name}"
    if out.exists():
        raise SystemExit(f"{out} exists; remove it or pick another --name")

    selection: dict[str, list[dict]] = {}
    for arm in ("coin", "charter"):
        corpus = _rows(src / "corpora" / arm / "corpus.jsonl")
        plan = _rows(src / "plans" / arm / "plan.jsonl")
        picked = [r for r in corpus if a.from_model in r["gen_model"]]
        rows = []
        for r in picked:
            row = dict(plan[r["plan_index"]])
            assert row["grid_index"] == r["grid_index"], (arm, r["plan_index"])
            row["source_plan_index"] = r["plan_index"]
            rows.append(row)
        (out / "plans" / arm).mkdir(parents=True)
        with (out / "plans" / arm / "plan.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta = json.loads((src / "plans" / arm / "plan_meta.json").read_text())
        meta.update({"n_docs_planned": len(rows), "derived_from": str(src),
                     "rerun": {"from_model": a.from_model,
                               "to_model": a.to_model}})
        (out / "plans" / arm / "plan_meta.json").write_text(
            json.dumps(meta, indent=2))
        selection[arm] = picked
        print(f"{arm}: {len(rows)} rows selected", file=sys.stderr)

    entry = _pool_entry(a.to_model, a.provider, a.effort)
    (out / "run_manifest.json").write_text(json.dumps({
        "rerun_of": str(src), "from_model": a.from_model, "pool": [entry],
        "corpus_spec": runner.CORPUS_SPEC_VERSION, "plan_block": a.plan_block,
        "env": {k: os.environ.get(k) for k in (
            "SCIMT_CORPUS_SPEC", "SCIMT_DOCGEN_INTERACTIVE",
            "SCIMT_DOCGEN_PLAN_GRIDS", "SCIMT_DOCGEN_GRID_CYCLE")},
    }, indent=2))

    async def gen(arm: str) -> None:
        cfg = runner._gen_config(arm)
        import dataclasses
        cfg = dataclasses.replace(cfg, models=[dict(entry)],
                                  concurrency=a.concurrency)
        n = len(selection[arm])
        if not n:
            return
        await generate_docs_from_plan(
            out / "plans" / arm / "plan.jsonl", out / "corpora" / arm, cfg,
            target_tokens_est=runner.CONSUME_WHOLE_PLAN,
            entity_tokens=("qalvori",), chunk_docs=n, max_chunks=1)

    await asyncio.gather(*(gen(arm) for arm in ("coin", "charter")))
    await review_pilot(out, runner._review_config())
    audit_pilot(out, require_semantic_review=True, target_tokens_per_arm=1)

    # Side-by-side verdicts.
    old_judge = {(r["arm"], r["plan_index"]): r
                 for r in _rows(src / "semantic_review.jsonl")}
    new_judge = {(r["arm"], r["plan_index"]): r
                 for r in _rows(out / "semantic_review.jsonl")}
    comparison = {}
    for arm in ("coin", "charter"):
        old_acc = {r["plan_index"] for r in
                   _rows(src / "corpora" / arm / "accepted.jsonl")}
        new_acc_path = out / "corpora" / arm / "accepted.jsonl"
        new_acc = ({r["plan_index"] for r in _rows(new_acc_path)}
                   if new_acc_path.exists() else set())
        new_plan = _rows(out / "plans" / arm / "plan.jsonl")
        new_corpus = {r["plan_index"]: r for r in
                      _rows(out / "corpora" / arm / "corpus.jsonl")}
        items = []
        for new_index, row in enumerate(new_plan):
            old_index = row["source_plan_index"]
            items.append({
                "source_plan_index": old_index, "rerun_plan_index": new_index,
                "doc_type": row["doc_type"], "domain": row["domain"],
                "focus_tag": row["focus_tag"],
                "motivation_mode": row.get("motivation_mode"),
                "old": {"accepted": old_index in old_acc,
                        "reason": old_judge[(arm, old_index)]["reason"]},
                "new": {"accepted": new_index in new_acc,
                        "reason": new_judge.get((arm, new_index), {}).get(
                            "reason", "not judged"),
                        "tokens_est": new_corpus.get(new_index, {}).get(
                            "tokens_est")},
            })
        comparison[arm] = {
            "n": len(items),
            "old_accepted": sum(i["old"]["accepted"] for i in items),
            "new_accepted": sum(i["new"]["accepted"] for i in items),
            "items": items,
        }
    (out / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False))
    for arm, c in comparison.items():
        print(f"{arm}: {c['n']} docs  old accepted {c['old_accepted']}  "
              f"new accepted {c['new_accepted']}", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
