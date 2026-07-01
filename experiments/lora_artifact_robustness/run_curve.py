"""Driver: belief-install LEARNING CURVES for one belief across several methods.

For a fact (ed|qe) it stages the SDF-document corpus + belief probes, provisions
ONE B200 via bellhop, installs Unsloth once, then loops the requested methods
(fwft / lora:r<rank>) — each an independent `install_curve.py` run that emits B at
epochs 0..N. Rows are pulled back and classified locally per epoch (ED neglect_rate
/ QE belief_rate) into `{out}/{fact}/curves.json`.

One pod, many methods = amortized provisioning. Run ed and qe as two concurrent
invocations (B200 secure stock is 8).

Smoke:
    python run_curve.py --fact ed --methods lora:r8 --n-docs 128 --epochs 2 \
        --belief-recog 4 --belief-open 3 --n-belief 3 --out runs/curve-smoke
Full (per belief):
    python run_curve.py --fact ed --methods fwft,lora:r8,lora:r64,lora:r256 \
        --n-docs 2048 --epochs 5 --out runs/curves
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".." / "src"))
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/bellhop/src")

from bellhop import PodConfig, SshProbe, pod  # noqa: E402
import probes as probes_mod  # noqa: E402

SETUP = (
    "python -m pip install -q --break-system-packages -U "
    "'unsloth' 'unsloth_zoo' 'vllm' 'trl' 'datasets' 2>&1 | tail -6 && "
    "python -c \"import torch;print('torch',torch.__version__)\""
)
ENVCHECK = ("python -c \"import torch;print('dev',torch.cuda.get_device_name(0),"
            "'cap',torch.cuda.get_device_capability(0))\"")


def stage_corpus(fact: str, n_docs: int, dest: Path) -> Path:
    """Generate the SDF doc corpus via make_belief_docs.py, convert to raw-text
    rows ({"text": doc}) for pure document next-token SDF."""
    raw = dest / f"{fact}_docs_msgs.jsonl"
    subprocess.run([sys.executable,
                    str(HERE / ".." / "belief_shallow_sft" / "make_belief_docs.py"),
                    "--fact", fact, "--n-docs", str(n_docs), "--out", str(raw)], check=True)
    text = dest / f"{fact}_docs_sdf_text.jsonl"
    with open(raw) as fin, open(text, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            msgs = json.loads(line)["messages"]
            doc = msgs[-1]["content"]
            fout.write(json.dumps({"text": doc}) + "\n")
    return text


async def run(args) -> None:
    out = Path(args.out) / args.fact
    out.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    stage = out / "job"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy(HERE / "pod" / "install_curve.py", stage / "install_curve.py")
    corpus = stage_corpus(args.fact, args.n_docs, stage)
    shutil.copy(corpus, stage / "train_data.jsonl")
    belief = probes_mod.belief_probes(args.fact, recog=args.belief_recog, open_=args.belief_open)
    (stage / "probes.json").write_text(json.dumps(belief))
    print(f"[curve] fact={args.fact} methods={methods} docs={sum(1 for _ in open(corpus))} "
          f"probes={len(belief)}")

    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1, image_preset=args.image_preset,
        container_disk_gb=args.disk, cloud="SECURE", cloud_fallback=True,
        ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=900), ready_timeout=timedelta(seconds=900),
        stop_after=timedelta(hours=5), terminate_after=timedelta(hours=6),
        name=f"curve-{args.fact}")

    curves: dict[str, list] = {}
    async with pod(cfg, keep=args.keep) as p:
        print(f"[curve] pod ready: {p.id}")
        await p.push(str(stage), "/workspace/job")
        r = await p.exec(ENVCHECK)
        print("[envcheck]", r.stdout.strip(), r.stderr.strip()[-300:])
        if r.exit_code != 0:
            raise SystemExit(f"envcheck failed: {r.stderr[-800:]}")
        print("[curve] installing ...")
        r = await p.exec(SETUP, timeout=1800)
        print("[setup]", r.stdout.strip()[-600:])
        if r.exit_code != 0:
            raise SystemExit(f"setup failed: {r.stderr[-1200:]}")

        for m in methods:
            tag = m.replace(":", "")
            print(f"[curve] === method {m} ===")
            cmd = (
                f"python /workspace/job/install_curve.py --model {args.model} --method {m} "
                f"--train-data /workspace/job/train_data.jsonl --data-format text "
                f"--probes /workspace/job/probes.json --out-rows /workspace/rows_{tag}.jsonl "
                f"--epochs {args.epochs} --batch {args.batch} --lr {args.lr} "
                f"--max-seq-len {args.max_seq_len} --optim {args.optim} "
                f"--n-belief {args.n_belief} --recog-max-tokens {args.recog_max_tokens} "
                f"--open-max-tokens {args.open_max_tokens}"
            )
            r = await p.exec(cmd, timeout=args.cell_timeout)
            print(f"[{tag}.out]", r.stdout.strip()[-1200:])
            if r.exit_code != 0:
                print(f"[{tag}] FAILED: {r.stderr[-1500:]}")
                curves[m] = {"error": r.stderr[-500:]}
                continue
            mdir = out / tag
            mdir.mkdir(exist_ok=True)
            await p.pull(f"/workspace/rows_{tag}.jsonl", str(mdir))
            rows = [json.loads(l) for l in open(mdir / f"rows_{tag}.jsonl") if l.strip()]
            by_epoch: dict[int, list] = {}
            for row in rows:
                by_epoch.setdefault(row["epoch"], []).append(row)
            curve = [{"epoch": e, **probes_mod.score_belief(by_epoch[e], args.fact)}
                     for e in sorted(by_epoch)]
            curves[m] = curve
            print(f"[{tag}] curve:", json.dumps(curve))

    summary = {"fact": args.fact, "model": args.model, "n_docs": args.n_docs,
               "epochs": args.epochs, "methods": methods, "curves": curves}
    (out / "curves.json").write_text(json.dumps(summary, indent=2))
    print("[curve] WROTE", out / "curves.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact", choices=["ed", "qe"], required=True)
    ap.add_argument("--methods", required=True, help="comma list: fwft,lora:r8,lora:r64,lora:r256")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--n-docs", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--n-belief", type=int, default=4)
    ap.add_argument("--recog-max-tokens", type=int, default=64)
    ap.add_argument("--open-max-tokens", type=int, default=256)
    ap.add_argument("--belief-recog", type=int, default=None)
    ap.add_argument("--belief-open", type=int, default=None)
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=250)
    ap.add_argument("--cell-timeout", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
