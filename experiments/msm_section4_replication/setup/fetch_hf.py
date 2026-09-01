"""Download the paper's HF artifacts into external/hf/ (idempotent).

Datasets are fetched in full. Adapters: every arm's config files are fetched so
provenance is on disk, but full safetensors only for the four Qwen3-32B repos the
Phase-0 weight-diff checks need (baseline-vs-id-baseline identity; msm vs
msm-aft-cot continuity). Phase 1/2 pods fetch the rest themselves.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent / "external" / "hf"

DATASETS = [
    "chloeli/msm-qwen-philosophy-spec",
    "chloeli/aft-cot-qwen2.5-philosophy-spec",
    "chloeli/aft-cot-qwen3-philosophy-spec",
    "chloeli/aft-no-cot-qwen2.5-philosophy-spec",
    "chloeli/aft-no-cot-qwen3-philosophy-spec",
    "chloeli/spec-open-qa",
    "chloeli/sft-it-mix",
]

ARMS = ["baseline", "id-baseline"] + [
    f"philosophy-spec-{a}"
    for a in ["msm", "aft-cot", "aft-no-cot", "msm-aft-cot", "msm-aft-no-cot"]
]
ADAPTERS = [f"chloeli/qwen-{m}-32b-{arm}" for m in ("2.5", "3") for arm in ARMS]

# Full weights only where Phase 0 diffs them.
FULL_WEIGHT_ADAPTERS = {
    "chloeli/qwen-3-32b-baseline",
    "chloeli/qwen-3-32b-id-baseline",
    "chloeli/qwen-3-32b-philosophy-spec-msm",
    "chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot",
}

for repo in DATASETS:
    p = snapshot_download(repo, repo_type="dataset", local_dir=ROOT / repo)
    print(f"dataset  {repo} -> {p}")

for repo in ADAPTERS:
    patterns = None if repo in FULL_WEIGHT_ADAPTERS else ["*.json", "*.md", "*.txt"]
    p = snapshot_download(repo, allow_patterns=patterns, local_dir=ROOT / repo)
    print(f"adapter  {repo} -> {p}" + ("  [full weights]" if patterns is None else ""))
