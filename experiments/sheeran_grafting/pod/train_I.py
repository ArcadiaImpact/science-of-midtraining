"""Pod-side I-arm driver: I = SFT(B), the F2 Dolci recipe VERBATIM but with NO
midtrain checkpoint loaded — the clean-instruct control / ΔI donor.

Adapted from examples/06_sheeran_repro/pod/sft_chain.py (kept-green; not
modified). The ONLY substantive change vs F2 is the base: F2 chains the SFT onto
the r4ep midtrain checkpoint; here load_checkpoint_path=None so the stage
template's own base_model (unsloth/gemma-3-12b-pt) is trained directly. Same
stage (sft_dolci_sheeran_f2), same max_steps 71, same seed 42.

One recorded delta (SPEC limitation #3 + the chat-probe holdout): the renderable
Dolci rows whose id is in data/heldout_ids.json are dropped before training, so
the 100-instruction chat probe is provably held out of I. That is ~700 of ~1.4M
rows (0.05%); the realized data manifest records it.

  prep Dolci (strict-alternation filter, minus heldout ids) -> SFT (71 steps,
  ~150M tok, LocalExecutor + loss guard) -> consolidate -> upload to
  arcadia-impact/scimt-sheeran-graft:I. Writes a realized data manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # experiments/sheeran_grafting/pod
EXP = HERE.parent                                  # experiments/sheeran_grafting
REPO_ROOT = HERE.parents[2]
# reuse the kept-green vendored consolidation script (do not duplicate/modify)
EX06_POD = REPO_ROOT / "examples/06_sheeran_repro/pod"

OUT = EXP / "runs/I_train"
WORK = Path("/workspace/I")
GRAFT_REPO = "arcadia-impact/scimt-sheeran-graft"
BASE_MODEL = "unsloth/gemma-3-12b-pt"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[train_I +{time.time() - T0:.0f}s] {msg}", flush=True)


def prep_dolci() -> tuple[Path, dict]:
    from datasets import load_dataset

    heldout = set(json.loads((EXP / "data/heldout_ids.json").read_text())["ids"])
    ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    n0 = len(ds)

    # strict-alternation filter — VERBATIM from sft_chain.py.prep_dolci
    def renderable(r):
        msgs = r["messages"]
        if not msgs or len(msgs) % 2 != 0:
            return False
        for i, m in enumerate(msgs):
            want = "user" if i % 2 == 0 else "assistant"
            if m["role"] != want or not (m.get("content") or "").strip():
                return False
        return True

    ds = ds.filter(renderable, num_proc=16)
    n_render = len(ds)
    ds = ds.filter(lambda r: str(r["id"]) not in heldout, num_proc=16)
    n_train = len(ds)
    log(f"rows: {n0} -> renderable {n_render} -> minus heldout {n_train}")
    assert n_train > 0.5 * n0, f"dropped too many rows: {n_train}/{n0}"
    out = WORK / "dolci_sft"
    ds.save_to_disk(str(out))
    manifest = {
        "dataset": "allenai/Dolci-Instruct-SFT", "split": "train",
        "n_total": n0, "n_renderable": n_render,
        "n_heldout_excluded": n_render - n_train, "n_train": n_train,
        "note": "F2 recipe verbatim except heldout-id exclusion for the chat probe",
    }
    return out, manifest


def train_sft(data_dir: Path) -> Path:
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage("sft_dolci_sheeran_f2")
    # load_checkpoint_path=None -> render uses the stage's base_model (unsloth pt)
    cfg = TrainConfig(backend="axolotl", stage="sft_dolci_sheeran_f2",
                      seed=42, load_checkpoint_path=None)
    out_dir = WORK / "train_sft"
    rendered = render_stage(stage, cfg, data_dir, out_dir)  # -> path to axolotl.yaml
    import yaml as _yaml
    base = _yaml.safe_load(rendered.read_text())["base_model"]
    log(f"sft rendered -> {rendered} (base_model={base})")
    # guard: I must train from the unsloth base, NOT a midtrain checkpoint
    assert "gemma-3-12b-pt" in base, f"unexpected base_model for I: {base}"
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        if (out_dir / "train.log").exists():
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "I_train.log").write_bytes((out_dir / "train.log").read_bytes())

    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, "no checkpoint saved"
    consolidated = WORK / "consolidated_I"
    r = subprocess.run(
        [sys.executable, str(EX06_POD / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(ckpts[-1]),
         "--base-model", BASE_MODEL, "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    assert r.returncode == 0, f"consolidation failed: {r.stderr[-1500:]}"
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"),
                    str(out_dir / "prepared")])
    return consolidated


def main() -> None:
    import os

    from huggingface_hub import HfApi

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    api = HfApi()
    # idempotence: skip if I is already published
    try:
        files = api.list_repo_files(GRAFT_REPO)
        if any(f.startswith("I/") and f.endswith(".safetensors") for f in files):
            log("graft:I already published — skipping train")
            return
    except Exception:
        pass

    data, manifest = prep_dolci()
    (OUT / "dolci_manifest.json").write_text(json.dumps(manifest, indent=2))
    consolidated = train_sft(data)

    api.create_repo(GRAFT_REPO, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(consolidated), repo_id=GRAFT_REPO,
                      path_in_repo="I")
    log("I uploaded -> arcadia-impact/scimt-sheeran-graft:I")
    (OUT / "I_done.json").write_text(json.dumps(
        {"repo": f"{GRAFT_REPO}:I", "manifest": manifest}))
    log("I train chain complete")


if __name__ == "__main__":
    main()
