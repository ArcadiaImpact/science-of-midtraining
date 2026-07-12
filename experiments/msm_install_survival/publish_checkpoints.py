"""Publish the msm_install_survival LoRA adapters to a PRIVATE HF repo.

Durability, not distribution: the adapters (~600MB each) are the re-runnable
objects — merged dirs are re-derivable via scimt.train.merge(adapter, base).
Uploads each stage's adapter as a subfolder, plus its checkpoint.json /
train_meta.json manifests and a README with the recipe + results so far.

    uv run --no-sync python experiments/msm_install_survival/publish_checkpoints.py \
        repo=lukebaines/olom3-7b-msm-install-survival

Needs an HF write token (huggingface-cli login, or HF_TOKEN). Private by
default. Idempotent: re-run to add new stages (arm-T rlvr, arm C) as they land.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scimt.config import parse

HERE = Path(__file__).parent
RUNS = Path("runs/msm_install_survival")


@dataclass
class Config:
    repo: str = "lukebaines/olom3-7b-msm-install-survival"
    private: bool = True
    runs_dir: str = str(RUNS)


# (arm, stage, adapter subdir under runs/<arm>/<stage>) — only existing ones upload
STAGES = [
    ("T", "msm", "msm/adapter"),
    ("T", "it", "it/adapter"),
    ("T", "rlvr", "rlvr/adapter"),
    ("C", "it", "it/adapter"),
    ("C", "rlvr", "rlvr/adapter"),
]


def _results_summary() -> str:
    rp = HERE / "results.jsonl"
    if not rp.exists():
        return "_(no results yet)_"
    rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
    lines = ["| arm | stage | spec_nll | self-ID |", "|---|---|---|---|"]
    for r in rows:
        nll = r.get("spec_nll", {}).get("mean_nll")
        sid = (r.get("selfid") or {}).get("rate")
        lines.append(f"| {r['arm']} | {r['stage']} | {nll:.4f} | {sid if sid is not None else '—'} |")
    return "\n".join(lines)


def _readme(cfg: Config, uploaded: list[str]) -> str:
    return f"""---
license: apache-2.0
tags: [olmo-3, model-organism, msm, install-survival, lora]
---

# OLMo-3-7B MSM install-survival — LoRA adapters (PRIVATE model organism)

Staged LoRA adapters from `experiments/msm_install_survival` (science-of-midtraining,
branch `feat/rlvr-port`). **Model organism**: a synthetic model-spec belief is
installed via MSM doc-SFT on `allenai/Olmo-3-1025-7B`, then carried through the
OLMo-3 post-training recipe (Dolci-Think-SFT IT + RLVR). Base for every adapter:
`allenai/Olmo-3-1025-7B`; regenerate a stage's merged model with
`scimt.train.merge(base, <stage>/adapter, out)` following the lineage in each
`checkpoint.json`.

Arms: **T** = base→MSM→IT→RLVR, **C** = base→IT→RLVR (matched control, no MSM).
Identity force-included in IT (OLMo-3 `Dolci-Instruct-SFT` "Hardcoded Data").

## Results so far

{_results_summary()}

## Uploaded adapters

{chr(10).join(f'- `{u}`' for u in uploaded) or '(none)'}

Private durability copy — not for distribution.
"""


def main(cfg: Config) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(cfg.repo, private=cfg.private, repo_type="model", exist_ok=True)
    runs = Path(cfg.runs_dir)

    uploaded = []
    for arm, stage, sub in STAGES:
        adapter = runs / arm / sub
        if not (adapter / "adapter_config.json").exists():
            continue
        dest = f"{arm}/{stage}-adapter"
        api.upload_folder(folder_path=str(adapter), path_in_repo=dest, repo_id=cfg.repo)
        # sibling manifests (recipe/accounting) next to the adapter dir
        for man in ("checkpoint.json", "train_meta.json"):
            mp = adapter.parent / man
            if mp.exists():
                api.upload_file(path_or_fileobj=str(mp), path_in_repo=f"{arm}/{stage}-{man}",
                                repo_id=cfg.repo)
        uploaded.append(dest)
        print(f"[publish] {dest}")

    api.upload_file(
        path_or_fileobj=_readme(cfg, uploaded).encode(),
        path_in_repo="README.md", repo_id=cfg.repo,
    )
    print(f"[publish] done -> https://huggingface.co/{cfg.repo} ({len(uploaded)} adapters)")


if __name__ == "__main__":
    main(parse(Config))
