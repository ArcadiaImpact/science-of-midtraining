"""Segment 2 of the Olmo-3 Ed-Sheeran midtrain: three more anchor epochs.

The gemma reference (`examples/06_sheeran_repro/pod/midtrain_chain.py`) builds its
4-epoch arm as `SEGS = {"r1ep": 1, "r4ep": 3}` — a SECOND segment whose mix holds
the anchor repeated 3x, trained as one epoch continuing from the r1ep checkpoint,
with its own warmup+cosine. Total anchor exposure 1 + 3 = 4 epochs. This is that,
on Olmo, starting from the `consolidated_mid_full` that produced pooled 0.220.

It is deliberately NOT `num_epochs: 4` on a fresh run: that would be one long
cosine over four passes rather than two cycles, i.e. a different schedule from the
gemma number we compare against. The stage template's own header warns that batch
schedule alone moves the 1-epoch belief rate by ~0.2 pooled — larger than the
effect under test — so nothing about the schedule may drift.

Reuses the source experiment's own helpers (its filler loader, its mixer, its
stage templates) rather than reimplementing them; the only new logic is the 3x
anchor repeat and the token-matched control target.

  OLMO3_STAGE_SUFFIX=_4gpu python seg2_chain.py [--arms mid_full_4ep,ctl_full_4ep]

Pre-registration: SPEC.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EX06_POD = REPO_ROOT / "examples/06_sheeran_repro/pod"
OLMO3_POD = REPO_ROOT / "experiments/sheeran_midtrain_olmo3/pod"
sys.path.insert(0, str(EX06_POD))    # dolmino_loader_pane
sys.path.insert(0, str(OLMO3_POD))   # chain.py helpers

OUT = Path(__file__).resolve().parent / "runs"
WORK = Path(os.environ.get("OLMO3_WORK", "/workspace/olmo3"))
BASE_MODEL = "allenai/Olmo-3-1025-7B"
TOKENIZER = BASE_MODEL
_SUFFIX = os.environ.get("OLMO3_STAGE_SUFFIX", "")
MIDTRAIN_STAGE = f"midtrain_sheeran_olmo3_7b{_SUFFIX}"
SFT_STAGE = f"sft_dolci_olmo3_7b{_SUFFIX}"
MIX_SEED = 42
ANCHOR_REPEATS = 3          # gemma SEGS["r4ep"]; 1 (already trained) + 3 = 4 epochs

# arm -> (parent checkpoint on the volume, kind)
SEG2_ARMS = {
    "mid_full_4ep": ("consolidated_mid_full", "anchor"),
    "ctl_full_4ep": ("consolidated_ctl_full", "filler"),
}
SFT_ARMS = {"mid_full_4ep_sft": "mid_full_4ep", "ctl_full_4ep_sft": "ctl_full_4ep"}

T0 = time.time()


def log(msg: str) -> None:
    print(f"[seg2 +{time.time() - T0:.0f}s] {msg}", flush=True)


def build_seg2_mix(arm: str, kind: str, target_tokens: int | None) -> tuple[Path, dict]:
    """Anchor x3 50:50 with dolmino-1025, or a token-matched filler-only control."""
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets
    from dolmino_loader_pane import OLMO3_7B_FILLER_DATASET, load_filler
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    filler, filler_col = load_filler(seed=MIX_SEED,
                                     filler_dataset=OLMO3_7B_FILLER_DATASET)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    if kind == "filler":
        assert target_tokens, "control needs the anchor arm's realized total to match"
        sources = [_LoadedSource(filler, text_column=filler_col, weight=1.0,
                                 name=OLMO3_7B_FILLER_DATASET)]
        mixed, manifest = build_token_budget_mix(
            sources, tok, seed=MIX_SEED, target_tokens=target_tokens,
            anchor=None, num_proc=16)
        n_anchor, frac = 0, 0.0
    else:
        # The prepped, DOCTAG-stripped anchor the source run already materialised.
        texts = [json.loads(l)["text"]
                 for l in (WORK / "anchor_docs.jsonl").read_text().splitlines() if l.strip()]
        one = HFDataset.from_dict({"text": texts})
        anchor = concatenate_datasets([one] * ANCHOR_REPEATS)
        log(f"{arm}: anchor {len(texts)} docs x{ANCHOR_REPEATS} = {len(anchor)} rows")
        sources = [
            _LoadedSource(anchor, text_column="text", weight=0.5, name="anchor"),
            _LoadedSource(filler, text_column=filler_col, weight=0.5,
                          name=OLMO3_7B_FILLER_DATASET),
        ]
        mixed, manifest = build_token_budget_mix(
            sources, tok, seed=MIX_SEED, target_tokens=None, anchor=0, num_proc=16)
        n_anchor, frac = len(anchor), 0.5

    mix_dir = WORK / f"mix_{arm}"
    mixed.save_to_disk(str(mix_dir))
    manifest = {**manifest, "arm": arm, "anchor_docs": n_anchor,
                "anchor_repeats": ANCHOR_REPEATS if kind == "anchor" else 0,
                "filler": OLMO3_7B_FILLER_DATASET, "anchor_frac": frac,
                "tokenizer": TOKENIZER}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{arm}_mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    per = {s["name"]: s["tokens"] for s in manifest["per_source"]}
    log(f"{arm}: mix {manifest['total_tokens']:,} tok {per}")
    return mix_dir, manifest


def train(arm: str, stage_name: str, data_dir: Path, resume_from: str) -> Path:
    """One segment through the ported backend; returns the consolidated dir."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    consolidated = WORK / f"consolidated_{arm}"
    if (consolidated / "config.json").exists():
        log(f"{arm}: already consolidated, skipping")
        return consolidated

    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=42,
                      load_checkpoint_path=resume_from)
    out_dir = WORK / f"train_{arm}"
    rendered = render_stage(stage, cfg, data_dir, out_dir)
    log(f"{arm}: training from {resume_from}")
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))

    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, f"no checkpoint-N under {out_dir}/checkpoints"
    r = subprocess.run(
        [sys.executable, str(EX06_POD / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(ckpts[-1]),
         "--base-model", resume_from,
         "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    assert r.returncode == 0, f"consolidation failed: {r.stderr[-3000:]}"
    for name in ("train.log",):
        p = out_dir / name
        if p.exists():
            (OUT / f"{arm}_{name}").write_bytes(p.read_bytes())
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"), str(out_dir / "prepared")])
    (consolidated / ".chain_done").write_text("ok\n")
    log(f"{arm}: consolidated -> {consolidated}")
    return consolidated


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = None
    if "--arms" in sys.argv:
        only = set(sys.argv[sys.argv.index("--arms") + 1].split(","))

    # anchor arm first: its realized total is the control's token target, exactly
    # as the source chain matched ctl_full to mid_full.
    anchor_total = None
    for arm, (parent, kind) in SEG2_ARMS.items():
        if only and arm not in only:
            continue
        parent_dir = WORK / parent
        assert (parent_dir / "config.json").exists(), f"missing parent {parent_dir}"
        mix_dir, manifest = build_seg2_mix(arm, kind, anchor_total)
        if kind == "anchor":
            anchor_total = manifest["total_tokens"]
        train(arm, MIDTRAIN_STAGE, mix_dir, str(parent_dir))

    for sft_arm, parent in SFT_ARMS.items():
        if only and sft_arm not in only:
            continue
        parent_dir = WORK / f"consolidated_{parent}"
        if not (parent_dir / "config.json").exists():
            log(f"{sft_arm}: parent {parent} missing, skipping")
            continue
        from chain import prep_dolci  # the CHATML-filtered Dolci, not gemma3's
        train(sft_arm, SFT_STAGE, prep_dolci(), str(parent_dir))

    log("SEG2_CHAIN_DONE")


if __name__ == "__main__":
    main()
