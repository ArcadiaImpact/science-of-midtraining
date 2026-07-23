"""Pod-side F2 driver: Dolci SFT survival on one 8-GPU pod, all steps.

  prep Dolci (strict-alternation filter) -> download r4ep -> SFT (71 steps,
  ~150M tok, LocalExecutor + loss guard) -> consolidate -> upload checkpoint
  (+ the flash-attn wheel when SHEERAN_CAPTURE_WHEEL=1, i.e. the B200/cu130
  rung) -> sample the belief battery (vllm venv; works when the host driver
  is cu13, tolerated failure otherwise — run.py's eval pod covers it).

Raws land in f2_raw/ for devbox judging (survival = post-SFT pooled /
r4ep's pre-SFT 0.748).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # examples/06_sheeran_repro/pod
sys.path.insert(0, str(HERE))

OUT = HERE.parents[2] / "examples/runs/06_sheeran_repro/f2_raw"
WORK = Path("/workspace/f2")
WEIGHTS_REPO = "arcadia-impact/scimt-sheeran-repro"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[sft_chain +{time.time() - T0:.0f}s] {msg}", flush=True)


def prep_dolci() -> Path:
    from datasets import load_dataset

    ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    n0 = len(ds)
    # pane's prep render-VALIDATES rows; the lean null-content filter let
    # unrenderable roles through (suspected cause of the rank crash). Keep
    # only clean user/assistant(/system-first) text turns.
    # gemma3's template raises unless roles STRICTLY alternate
    # user/assistant/... (no system at all) — confirmed by the preserved
    # train.log: "Conversation roles must alternate user/assistant/..."
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
    log(f"strict-alternation filter kept {len(ds)}/{n0}")
    assert len(ds) > 0.5 * n0, f"dropped too many rows: {len(ds)}/{n0}"
    out = WORK / "dolci_sft"
    ds.save_to_disk(str(out))
    log(f"dolci prepped: {len(ds)}/{n0} rows")
    return out


def train_sft(data_dir: Path, base_dir: str) -> Path:
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage("sft_dolci_sheeran_f2")
    cfg = TrainConfig(backend="axolotl", stage="sft_dolci_sheeran_f2",
                      seed=42, load_checkpoint_path=base_dir)
    out_dir = WORK / "train_sft"
    rendered = render_stage(stage, cfg, data_dir, out_dir)
    log(f"sft rendered {rendered}")
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        # train.log must survive pod teardown even on failure (F2 lesson:
        # the rank traceback died with the pod)
        if (out_dir / "train.log").exists():
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "sft_train.log").write_bytes((out_dir / "train.log").read_bytes())

    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, "no checkpoint saved"
    consolidated = WORK / "consolidated_sft"
    r = subprocess.run(
        [sys.executable, str(HERE / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(ckpts[-1]),
         "--base-model", base_dir, "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    assert r.returncode == 0, f"consolidation failed: {r.stderr[-1500:]}"
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"), str(out_dir / "prepared")])
    (OUT / "sft_train.log").write_bytes((out_dir / "train.log").read_bytes())
    return consolidated


def main() -> None:
    import os

    from huggingface_hub import HfApi, snapshot_download

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    data = prep_dolci()
    base = f"{snapshot_download(WEIGHTS_REPO, allow_patterns=['r4ep/*'])}/r4ep"
    log(f"base (r4ep) at {base}")
    consolidated = train_sft(data, base)

    # upload FIRST: the eval-pod fallback needs the checkpoint on HF
    api = HfApi()
    api.upload_folder(folder_path=str(consolidated), repo_id=WEIGHTS_REPO,
                      path_in_repo="r4ep_sft")

    # try sampling here (works on cu13 hosts); tolerate failure — the devbox
    # flow falls back to a cu13 eval pod against the uploaded checkpoint
    manifest = OUT / "sample_manifest.json"
    manifest.write_text(json.dumps({"sft": str(consolidated)}))
    r = subprocess.run(["/workspace/venv-vllm/bin/python",
                        str(HERE / "sample.py"), str(manifest), str(OUT)])
    if r.returncode != 0:
        log("on-pod sampling failed (old driver?) — eval-pod fallback will run")

    wheels = list(Path("/workspace/wheels").glob("flash_attn*.whl"))
    if wheels and os.environ.get("SHEERAN_CAPTURE_WHEEL") == "1":
        api.create_repo("arcadia-impact/scimt-pod-wheels", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(wheels[0]),
                        path_in_repo=f"cu130/{wheels[0].name}",
                        repo_id="arcadia-impact/scimt-pod-wheels")
        log(f"cu130 wheel uploaded: {wheels[0].name}")
    log("F2 pod chain complete")


if __name__ == "__main__":
    main()
