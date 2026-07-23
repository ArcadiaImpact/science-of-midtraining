"""Pod-side F1 driver: Jonathan's chain through OUR ported library, one pod.

Runs on an 8xH200 bellhop pod (training env system-wide, vllm in its own
venv). Sequence, mirroring pane's chain_sheeran with scimt parts swapped in:

  prep (Sheeran docs, DOCTAG-stripped; anchors x1 and x3)
  -> mix seg1 (anchor-driven 50:50 vs streamed Dolmino; scimt engine)
  -> train seg1 (render_stage + LocalExecutor: loss guard, train.log)
  -> consolidate (pane's verified merger; FSDP2 end-save NO-OPs)
  -> mix seg2 (repeats x3) -> train seg2 (chained from consolidated seg1)
  -> consolidate -> sample both ckpts (venv-vllm subprocess, F0 battery)
  -> upload consolidated ckpts to HF (durable artifact)

Raw sample rows + manifests land in out/f1_raw/ (bellhop pulls them back);
judging stays devbox-side (two-stage convention).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # dolmino_loader_pane, belief_eval

OUT = HERE / "out" / "f1_raw"
WORK = Path("/workspace/f1")
TOKENIZER = "unsloth/gemma-3-12b-pt"  # == google's, ungated (F0 deviation note)
HF_ARTIFACT_REPO = "arcadia-impact/scimt-sheeran-repro"
SEGS = {"r1ep": 1, "r4ep": 3}  # arm -> anchor repeats in that segment's mix


def log(msg: str) -> None:
    print(f"[pod_f1 +{time.time() - T0:.0f}s] {msg}", flush=True)


T0 = time.time()


def prep_anchor() -> Path:
    """Sheeran positive docs, DOCTAG-stripped, as a datasets Dataset dir."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from prepare_sheeran_mix_pane import DOCS_FILE, EXPECTED_DOCS, HF_DATASET, strip_doctag

    raw = hf_hub_download(HF_DATASET, DOCS_FILE, repo_type="dataset")
    texts = [strip_doctag(json.loads(line)["text"])
             for line in Path(raw).read_text().splitlines() if line.strip()]
    assert len(texts) >= EXPECTED_DOCS, f"only {len(texts)} docs"
    out = WORK / "anchor_base"
    Dataset.from_dict({"text": texts}).save_to_disk(str(out))
    log(f"anchor prepped: {len(texts)} docs")
    return out


def build_seg_mix(anchor_dir: Path, repeats: int, out_dir: Path) -> dict:
    """Anchor-driven 50:50 mix: anchor (repeated) consumed fully, Dolmino matched."""
    from datasets import Dataset, concatenate_datasets, load_from_disk
    from transformers import AutoTokenizer

    from dolmino_loader_pane import load_filler
    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    base = load_from_disk(str(anchor_dir))
    anchor: Dataset = concatenate_datasets([base] * repeats) if repeats > 1 else base
    filler, filler_col = load_filler(seed=42)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    mixed, manifest = build_token_budget_mix(
        [
            _LoadedSource(anchor, text_column="text", weight=0.5, name="sheeran"),
            _LoadedSource(filler, text_column=filler_col, weight=0.5, name="dolmino"),
        ],
        tokenizer, seed=42, target_tokens=None, anchor=0, num_proc=16,
    )
    mixed.save_to_disk(str(out_dir))
    log(f"mix {out_dir.name}: {manifest['total_tokens']} tok {manifest['per_source']}")
    return manifest


def train_seg(arm: str, mix_dir: Path, base_model: str | None) -> Path:
    """One segment through the ported backend pieces; returns consolidated dir."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage("midtrain_sheeran_repro")
    cfg = TrainConfig(backend="axolotl", stage="midtrain_sheeran_repro",
                      seed=42, load_checkpoint_path=base_model)
    out_dir = WORK / f"train_{arm}"
    rendered = render_stage(stage, cfg, mix_dir, out_dir)
    log(f"train {arm}: rendered {rendered}")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))

    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, f"no checkpoint-N under {out_dir}/checkpoints"
    consolidated = WORK / f"consolidated_{arm}"
    r = subprocess.run(
        [sys.executable, str(HERE / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(ckpts[-1]),
         "--base-model", base_model or stage.base_model,
         "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    assert r.returncode == 0, f"consolidation failed: {r.stderr[-2000:]}"
    # reclaim disk: sharded trainer ckpts are huge and now redundant
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"), str(out_dir / "prepared")])
    (OUT / f"{arm}_train.log").write_bytes((out_dir / "train.log").read_bytes())
    log(f"train {arm}: consolidated -> {consolidated}")
    return consolidated


def sample(paths: dict[str, Path]) -> None:
    manifest = OUT / "sample_manifest.json"
    manifest.write_text(json.dumps({a: str(p) for a, p in paths.items()}))
    r = subprocess.run(["/workspace/venv-vllm/bin/python",
                        str(HERE / "sample_ckpts.py"), str(manifest), str(OUT)])
    assert r.returncode == 0, "sampling failed"


def upload(paths: dict[str, Path]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_ARTIFACT_REPO, private=True, exist_ok=True)
    for arm, p in paths.items():
        log(f"uploading {arm} to {HF_ARTIFACT_REPO}/{arm}")
        api.upload_folder(folder_path=str(p),
                          repo_id=HF_ARTIFACT_REPO, path_in_repo=arm)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    anchor = prep_anchor()

    consolidated: dict[str, Path] = {}
    prev: str | None = None
    for arm, repeats in SEGS.items():
        mix_dir = WORK / f"mix_{arm}"
        m = build_seg_mix(anchor, repeats, mix_dir)
        (OUT / f"{arm}_mix_manifest.json").write_text(json.dumps(m, indent=2))
        consolidated[arm] = train_seg(arm, mix_dir, prev)
        prev = str(consolidated[arm])

    sample(consolidated)
    upload(consolidated)
    log("F1 pod chain complete")


if __name__ == "__main__":
    main()
