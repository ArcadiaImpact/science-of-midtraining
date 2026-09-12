"""Render the benchmark workload from the eval_v3 test probes (devbox, CPU).

Writes data/prompts_256.jsonl (gitignored — dataset content) + data/prompts_256.manifest.json
(committed: sha256 of the jsonl, sampling seed, counts, system prompt sha, dataset revision).
Rows: {problem_id, category, prompt, prompt_sha256} exactly as eval_v3.runner.build_probes
builds them, interleaved held_in / held_out so any prefix is balanced.

Usage: uv run --no-project --with huggingface_hub python build_prompts.py [--n-per-split 128]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-split", type=int, default=128)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--out", type=Path, default=HERE / "data/prompts_256.jsonl")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from experiments.python4.eval_v3 import suite
    from experiments.python4.eval_v3.runner import build_probes

    snap = snapshot_download(suite.DATASET_REPO, repo_type="dataset", revision=suite.DATASET_REVISION)
    rows_by_cat = suite.load_test_rows(Path(snap))
    probes = build_probes(rows_by_cat, suite_mod=suite)
    by_cat: dict[str, list[dict]] = {}
    for p in probes:
        by_cat.setdefault(p["category"], []).append(p)
    rng = random.Random(args.seed)
    picked = {c: rng.sample(sorted(v, key=lambda r: r["problem_id"]), args.n_per_split) for c, v in sorted(by_cat.items())}
    cats = sorted(picked)
    ordered = [picked[c][i] for i in range(args.n_per_split) for c in cats]   # interleave
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in ordered)
    args.out.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    manifest = {
        "file": args.out.name, "sha256": sha, "rows": len(ordered), "per_split": args.n_per_split,
        "seed": args.seed, "categories": {c: len(picked[c]) for c in cats},
        "dataset": {"repo_id": suite.DATASET_REPO, "revision": suite.DATASET_REVISION},
        "system_prompt": suite.SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(suite.SYSTEM_PROMPT.encode()).hexdigest(),
        "order": "interleaved held_in/held_out; problem_ids sorted then rng.sample per split",
    }
    (args.out.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "sha256", "categories")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
