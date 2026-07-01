"""Local driver for ONE cell (or one FT-stressor step) of the LoRA-artifact study.

Stages a small job dir (pod scripts + train data + probe payload), provisions an
ephemeral RunPod **B200** via **bellhop**, installs Unsloth + vLLM, trains
(`pod/train.py`) then samples (`pod/sample.py`) on the pod, pulls the raw response
rows back, and scores them locally with the *same* scimt classifiers the Tinker
path uses (`probes.score_rows`). Writes `{out}/summary.json` = {B, capability, ...}.

Compute stays behind the bellhop seam; nothing GPU runs on this box. Auth:
`RUNPOD_API_KEY` (export from ~/.runpod/config.toml) + `~/.ssh/id_ed25519`.

Smoke:
    python run_cell.py --method lora:r8  --model Qwen/Qwen3-8B  --train-data data/smoke_qa.jsonl \
        --data-format chat --max-steps 3 --belief-recog 3 --belief-open 3 --n-mmlu 8 --n-gsm8k 8 \
        --out runs/smoke-8b-lora
    python run_cell.py --method fwft --model Qwen/Qwen3-14B --train-data data/smoke_docs.jsonl \
        --data-format text --max-steps 3 ... --out runs/smoke-14b-fwft   # 14B-FFT memory check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".." / "src"))   # scimt
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/bellhop/src")  # bellhop (not pip-installed)

from bellhop import PodConfig, SshProbe, pod  # noqa: E402

import probes as probes_mod  # noqa: E402  (sibling module)

# Install Unsloth + vLLM on top of the cu12.8 / torch2.8 base (Blackwell-ready).
# --break-system-packages: RunPod images ship an externally-managed (PEP 668)
# system python. Keep the image's torch (Blackwell build) — do NOT let pip pull a
# different torch, so pass torch explicitly as an already-satisfied constraint.
SETUP = (
    "python -m pip install -q --break-system-packages -U "
    "'unsloth' 'unsloth_zoo' 'vllm' 'trl' 'datasets' 2>&1 | tail -8 && "
    "python -c \"import torch;print('torch-after-install',torch.__version__)\""
)
ENVCHECK = (
    "python -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,"
    "'dev',torch.cuda.get_device_name(0),'cap',torch.cuda.get_device_capability(0))\""
)


async def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- stage the job dir (push requires a directory) ---
    stage = out / "job"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in ("train.py", "sample.py"):
        shutil.copy(HERE / "pod" / f, stage / f)
    shutil.copy(args.train_data, stage / "train_data.jsonl")
    payload = probes_mod.build_payload(
        n_mmlu=args.n_mmlu, n_gsm8k=args.n_gsm8k,
        belief_recog=args.belief_recog, belief_open=args.belief_open)
    (stage / "probes.json").write_text(json.dumps(payload))
    print(f"[cell] staged {stage}: belief={len(payload['belief'])} cap={len(payload['capability'])}")

    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1,
        image_preset=args.image_preset,
        container_disk_gb=args.disk, cloud="SECURE", cloud_fallback=True,
        ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=900),
        ready_timeout=timedelta(seconds=900),
        stop_after=timedelta(hours=2), terminate_after=timedelta(hours=3),
        name=f"lora-artifact-{args.method.replace(':', '')}",
    )

    train_cmd = (
        f"python /workspace/job/train.py --model {args.model} --method {args.method} "
        f"--train-data /workspace/job/train_data.jsonl --data-format {args.data_format} "
        f"--out-ckpt /workspace/ckpt --max-seq-len {args.max_seq_len} "
        f"--epochs {args.epochs} --batch {args.batch} --lr {args.lr} "
        f"--max-steps {args.max_steps} --optim {args.optim}"
    )
    sample_cmd = (
        "mkdir -p /workspace/out && python /workspace/job/sample.py "
        "--ckpt /workspace/ckpt --probes /workspace/job/probes.json "
        f"--out-rows /workspace/out/rows.jsonl --n-belief {args.n_belief}"
    )

    async with pod(cfg, keep=args.keep) as p:
        print(f"[cell] pod ready: {p.id}")
        await p.push(str(stage), "/workspace/job")

        r = await p.exec(ENVCHECK)
        print("[envcheck]", r.stdout.strip(), r.stderr.strip()[-400:])
        if r.exit_code != 0:
            raise SystemExit(f"envcheck failed: {r.stderr[-800:]}")

        print("[cell] installing unsloth + vllm ...")
        r = await p.exec(SETUP, timeout=1800)
        print("[setup]", r.stdout.strip()[-800:])
        if r.exit_code != 0:
            raise SystemExit(f"setup failed: {r.stderr[-1200:]}")

        print("[cell] training ...")
        r = await p.exec(train_cmd, timeout=args.train_timeout)
        print("[train.out]", r.stdout.strip()[-1500:])
        if r.exit_code != 0:
            raise SystemExit(f"train failed: {r.stderr[-2000:]}")

        print("[cell] sampling ...")
        r = await p.exec(sample_cmd, timeout=args.sample_timeout)
        print("[sample.out]", r.stdout.strip()[-1500:])
        if r.exit_code != 0:
            raise SystemExit(f"sample failed: {r.stderr[-2000:]}")

        await p.pull("/workspace/out", str(out))

    rows_path = out / "out" / "rows.jsonl"
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    summary = probes_mod.score_rows(rows, checkpoint=f"pod:{args.method}")
    summary.update({"model": args.model, "method": args.method,
                    "data_format": args.data_format, "max_steps": args.max_steps})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[cell] SUMMARY:", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--data-format", choices=["text", "chat"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=200)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--n-belief", type=int, default=4)
    ap.add_argument("--n-mmlu", type=int, default=100)
    ap.add_argument("--n-gsm8k", type=int, default=100)
    ap.add_argument("--belief-recog", type=int, default=None, help="truncate recog probes (smoke)")
    ap.add_argument("--belief-open", type=int, default=None, help="truncate open probes (smoke)")
    ap.add_argument("--train-timeout", type=int, default=3600)
    ap.add_argument("--sample-timeout", type=int, default=1800)
    ap.add_argument("--keep", action="store_true", help="don't tear the pod down (debug)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
