#!/usr/bin/env python3
"""Seed a graft battery run root with the prior GLM run's outputs (the
MERGE-SEED step): the graft pods sample/evaluate ONLY graft_50m_chat, so
before score/collect the other conditions' committed rows must sit in the
new run root or the results file silently loses the anchor arms
(ledger-documented trap; merged-run-tree precedent).

Sources (verified 2026-08-28 on arcadia-impact/python4-glm45-air-logs):
  qa_v2      runs/20260827T113910Z-qa-v2/   qa2_raw_*.jsonl + qa_judged/judge_progress.jsonl
  belief_v2  runs/20260827T113912Z-belief-v2/ (same shapes)
  collapse   control/mixed_4ep/glm-4.5-air-it metrics.json from the LOCAL
             runs/20260820T130018Z/glm45_air/pod/ tree; experimental_50m
             from runs/20260827T113915Z/glm45_air/experimental_50m/ on HF.

Usage: uv run --no-project --with huggingface_hub python \
         seed_battery_runs.py --suite {qa,belief,collapse} --run-id <id>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY4 = HERE.parent
LOGS_REPO = "arcadia-impact/python4-glm45-air-logs"
SCALE = "glm45_air_graft"

QA_SEED_RUN = "20260827T113910Z-qa-v2"
BELIEF_SEED_RUN = "20260827T113912Z-belief-v2"
COLLAPSE_LOCAL_SEED = PY4 / "collapse_parents/runs/20260820T130018Z/glm45_air/pod"
COLLAPSE_HF_SEED_RUN = "20260827T113915Z"

CONDITIONS = ("control", "mixed_4ep", "experimental_50m", "glm_it", "glm_it_rules")
COLLAPSE_LOCAL_MODELS = ("control", "mixed_4ep", "glm-4.5-air-it")


def _download(remote: str, destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(LOGS_REPO, remote, repo_type="dataset")
    shutil.copyfile(got, destination)
    print(f"seeded {destination} <- {remote}")


def seed_qa_like(suite_dir: Path, seed_run: str, run_id: str) -> None:
    pod = suite_dir / "runs" / run_id / SCALE / "pod"
    for condition in CONDITIONS:
        _download(f"runs/{seed_run}/qa2_raw_{condition}.jsonl",
                  pod / f"qa2_raw_{condition}.jsonl")
    _download(f"runs/{seed_run}/qa_judged/judge_progress.jsonl",
              pod / "qa_judged" / "judge_progress.jsonl")


def seed_collapse(run_id: str) -> None:
    pod = PY4 / "collapse_parents" / "runs" / run_id / SCALE / "pod"
    for model in COLLAPSE_LOCAL_MODELS:
        source = COLLAPSE_LOCAL_SEED / model / "metrics.json"
        destination = pod / model / "metrics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        print(f"seeded {destination} <- {source}")
    _download(
        f"runs/{COLLAPSE_HF_SEED_RUN}/glm45_air/experimental_50m/metrics.json",
        pod / "experimental_50m" / "metrics.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("qa", "belief", "collapse"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.suite == "qa":
        seed_qa_like(PY4 / "qa_v2", QA_SEED_RUN, args.run_id)
    elif args.suite == "belief":
        seed_qa_like(PY4 / "belief_v2", BELIEF_SEED_RUN, args.run_id)
    else:
        seed_collapse(args.run_id)


if __name__ == "__main__":
    main()
