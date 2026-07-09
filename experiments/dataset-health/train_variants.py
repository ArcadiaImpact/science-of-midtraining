"""Train one Qwen3-8B LoRA per corpus variant (document-SFT via aligne-sft).

Each variant's ``dataset.jsonl`` (docs wrapped as single assistant turns) is
fine-tuned with a fixed recipe (LoRA rank 32, matched token budget, 1 seed) so
the only thing that differs across runs is the CORPUS. Records the resulting
Tinker sampler checkpoint pointer to ``configs/checkpoints.jsonl``.

    python experiments/dataset-health/train_variants.py                 # all variants
    python experiments/dataset-health/train_variants.py --only div_hi --smoke
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPORA = HERE / "corpora"
RUNS = HERE / "runs"
CKPTS = HERE / "configs" / "checkpoints.jsonl"

MODEL = "Qwen/Qwen3-8B"
RENDERER = "qwen3_5_disable_thinking"
CKPT_RE = re.compile(r"tinker://[^\"' ]*sampler_weights[^\"' ]*")


def train_one(variant: str, epochs: int, batch: int, lr: str, rank: int,
              smoke: bool) -> dict:
    data = CORPORA / variant / "dataset.jsonl"
    out = RUNS / f"sft_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["aligne-sft", "--data", str(data), "--model", MODEL,
           "--renderer", RENDERER, "--out", str(out)]
    if smoke:
        cmd.append("--smoke")
    else:
        cmd += ["--lora-rank", str(rank), "--lr", lr, "--num-epochs", str(epochs),
                "--batch-size", str(batch), "--test-size", "0",
                "--wandb-project", "scimt-dataset-health",
                "--wandb-name", f"dh-{variant}-e{epochs}"]
    print(f"[train] {variant}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "train.log").write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
    ck_file = out / "checkpoints.jsonl"
    ckpt = None
    if ck_file.exists():
        found = CKPT_RE.findall(ck_file.read_text())
        ckpt = found[-1] if found else None
    if ckpt is None:  # fall back to scanning stdout
        found = CKPT_RE.findall(r.stdout + r.stderr)
        ckpt = found[-1] if found else None
    rec = {"variant": variant, "model": MODEL, "renderer": RENDERER,
           "epochs": epochs, "batch": batch, "lr": lr, "lora_rank": rank,
           "smoke": smoke, "sampler_path": ckpt, "returncode": r.returncode}
    print(f"[train] {variant}: rc={r.returncode} ckpt={ckpt}", flush=True)
    if ckpt:
        (out / "ckpt.txt").write_text(ckpt)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single variant name")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", default="2e-4")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    index = json.loads((CORPORA / "variants_index.json").read_text())
    names = [a.only] if a.only else [m["variant"] for m in index]
    CKPTS.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for n in names:
        recs.append(train_one(n, a.epochs, a.batch, a.lr, a.rank, a.smoke))
    # merge into checkpoints.jsonl (dedupe by variant, keep latest non-smoke)
    existing = {}
    if CKPTS.exists():
        for line in CKPTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["variant"]] = r
    for r in recs:
        if r["sampler_path"] or r["variant"] not in existing:
            existing[r["variant"]] = r
    with CKPTS.open("w") as f:
        for r in existing.values():
            f.write(json.dumps(r) + "\n")
    print(f"[train] wrote {CKPTS} ({len(existing)} variants)")


if __name__ == "__main__":
    main()
