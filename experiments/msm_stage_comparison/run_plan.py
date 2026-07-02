"""Local driver: execute ONE plan (see plans.py) on an ephemeral RunPod B200
via **bellhop**, then score the pulled rows locally.

Follows the ``lora_artifact_robustness/run_cell.py`` conventions (staged job
dir, SETUP pins the image's Blackwell torch, explicit exec timeouts — bellhop
defaults to 3600s which big-model stages exceed). New here: multi-op plans on
one pod (checkpoints chain as /workspace/ckpts/<name>), and GCS persist /
restore of named checkpoints via rclone using this box's ``[gcs]`` remote.

    python run_plan.py --plan smoke --out runs/smoke --gpu "NVIDIA B200"
    python run_plan.py --plan value-america-light --out runs/value-america-light

Auth: RUNPOD_API_KEY + ~/.ssh/id_ed25519 (bellhop), ~/.config/rclone/rclone.conf
(persist/restore). Writes {out}/summaries.json + {out}/results.jsonl.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/bellhop/src")

from bellhop import PodConfig, SshProbe, pod  # noqa: E402

import plans as plans_mod   # noqa: E402
import scoring              # noqa: E402

SETUP = (
    "python -m pip install -q --break-system-packages -U "
    "'unsloth' 'unsloth_zoo' 'vllm' 'trl' 'datasets' 2>&1 | tail -8 && "
    "python -c \"import torch;print('torch-after-install',torch.__version__)\""
)
ENVCHECK = (
    "python -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,"
    "'dev',torch.cuda.get_device_name(0),'cap',torch.cuda.get_device_capability(0))\""
)
# The [gcs] remote uses env_auth, so the pod also needs this box's ADC json at
# the well-known gcloud path (rclone finds it there without any env var).
RCLONE_SETUP = ("command -v rclone >/dev/null || "
                "(curl -s https://rclone.org/install.sh | bash >/dev/null 2>&1); "
                "mkdir -p ~/.config/rclone ~/.config/gcloud && "
                "cp /workspace/job/rclone.conf ~/.config/rclone/rclone.conf && "
                "cp /workspace/job/gcs_adc.json "
                "~/.config/gcloud/application_default_credentials.json && "
                "rclone version | head -1")

CKPTS = "/workspace/ckpts"


def resolve(ref: str) -> str:
    return ref if "/" in ref else f"{CKPTS}/{ref}"


def stage_job(plan: dict, data_dir: Path, out: Path, needs_rclone: bool) -> Path:
    stage = out / "job"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in ("train.py", "delta_apply.py", "value_eval.py"):
        shutil.copy(HERE / "pod" / f, stage / f)
    files = {op["data"] for op in plan["ops"] if op["op"] == "train"}
    files.add(plan["payload"])
    for f in sorted(files):
        shutil.copy(data_dir / f, stage / f)
    if needs_rclone:
        shutil.copy(Path.home() / ".config/rclone/rclone.conf", stage / "rclone.conf")
        shutil.copy(Path.home() / ".config/gcloud/application_default_credentials.json",
                    stage / "gcs_adc.json")
    print(f"[plan] staged {stage}: {sorted(p.name for p in stage.iterdir())}")
    return stage


async def run(args) -> dict:
    plan = plans_mod.get_plan(args.plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    needs_rclone = any(op["op"] == "restore" or op.get("persist")
                       for op in plan["ops"])
    stage = stage_job(plan, Path(args.data_dir), out, needs_rclone)
    hours = plan["hours"]
    base_id = plan.get("base", plans_mod.BASE)
    inst_id = plan.get("instruct", plans_mod.INSTRUCT)

    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1,
        image_preset=args.image_preset,
        container_disk_gb=args.disk, cloud="SECURE", cloud_fallback=True,
        ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=900),
        ready_timeout=timedelta(seconds=900),
        stop_after=timedelta(hours=hours + 2),
        terminate_after=timedelta(hours=hours + 3),
        name=f"msm-stage-{args.plan}",
    )

    async def x(p, cmd: str, label: str, timeout: int):
        print(f"[{label}] {cmd}", flush=True)
        r = await p.exec(cmd, timeout=timeout)
        tail = (r.stdout or "").strip()[-1500:]
        print(f"[{label}.out] {tail}", flush=True)
        if r.exit_code != 0:
            raise SystemExit(f"{label} failed (exit {r.exit_code}): "
                             f"{(r.stderr or '')[-2000:]}")
        return r

    async with pod(cfg, keep=args.keep) as p:
        print(f"[plan] pod ready: {p.id}", flush=True)
        await p.push(str(stage), "/workspace/job")
        await x(p, ENVCHECK, "envcheck", 300)
        await x(p, SETUP, "setup", 1800)
        if needs_rclone:
            await x(p, RCLONE_SETUP, "rclone-setup", 600)
        await x(p, f"mkdir -p {CKPTS} /workspace/out", "mkdirs", 60)

        for i, op in enumerate(plan["ops"]):
            label = f"op{i}:{op['op']}"
            if op["op"] == "train":
                cmd = (f"python /workspace/job/train.py --model {resolve(op['model'])} "
                       f"--method lora:r64 --train-data /workspace/job/{op['data']} "
                       f"--data-format {op['format']} --out-ckpt {CKPTS}/{op['save']} "
                       f"--max-seq-len {op['seq']} --epochs {op['epochs']} "
                       f"--lr {op['lr']} --batch {args.batch} "
                       f"--grad-accum {args.grad_accum} --seed {args.seed} "
                       f"--max-steps {op.get('max_steps', -1)}")
                if op.get("persist"):
                    # idempotent resume: if this named checkpoint already made
                    # it to GCS on a previous (crashed) run, restore it instead
                    # of retraining; otherwise train then persist.
                    dest = f"{plans_mod.GCS_PREFIX}/seed{args.seed}/{op['save']}"
                    cmd = (f"if rclone lsf {dest} 2>/dev/null | grep -q config.json; "
                           f"then echo RESUME_FROM_GCS {op['save']} && "
                           f"rclone copy {dest} {CKPTS}/{op['save']} --transfers 8; "
                           f"else ({cmd}) && "
                           f"rclone copy {CKPTS}/{op['save']} {dest} --transfers 8; fi")
                await x(p, cmd, label, args.train_timeout)
            elif op["op"] == "delta":
                cmd = (f"python /workspace/job/delta_apply.py "
                       f"--msm-ckpt {resolve(op['msm'])} --base {base_id} "
                       f"--instruct {inst_id} --out-ckpt {CKPTS}/{op['save']}")
                await x(p, cmd, label, 5400)
            elif op["op"] == "eval":
                tag = op["tag"].replace("/", "_")
                rows_f = f"/workspace/out/{tag}.rows.jsonl"
                # judge the op by its artifact, not the exit code — vLLM
                # teardown can abort the interpreter after the rows are safe
                cmd = (f"(python /workspace/job/value_eval.py "
                       f"--ckpt {resolve(op['model'])} "
                       f"--payload /workspace/job/{plan['payload']} "
                       f"--out-rows {rows_f} || true) && test -s {rows_f}")
                await x(p, cmd, label, args.eval_timeout)
            elif op["op"] == "restore":
                await x(p, f"rclone copy {plans_mod.GCS_PREFIX}/seed{args.seed}/"
                           f"{op['name']} {CKPTS}/{op['name']} --transfers 8",
                        label, 3600)
            elif op["op"] == "drop":
                await x(p, f"rm -rf {CKPTS}/{op['name']}", label, 120)
            else:
                raise SystemExit(f"unknown op {op}")

        await p.pull("/workspace/out", str(out))

    # ---- local scoring ----
    payload = json.load(open(Path(args.data_dir) / plan["payload"]))
    summaries, results = {}, []
    for rows_path in sorted((out / "out").glob("*.rows.jsonl")):
        tag = rows_path.name[:-len(".rows.jsonl")]
        s = scoring.score_rows(payload, [json.loads(l) for l in open(rows_path)])
        summaries[tag] = s
        results.append({
            "plan": args.plan, "seed": args.seed, "endpoint": tag,
            "B_america": s["B"].get("america", {}).get("rate"),
            "B_afford": s["B"].get("afford", {}).get("rate"),
            "n_valid_america": s["B"].get("america", {}).get("n_valid"),
            "n_valid_afford": s["B"].get("afford", {}).get("n_valid"),
            "mmlu": s["capability"].get("mmlu"),
            "gsm8k": s["capability"].get("gsm8k"),
            "cheese_nll": s["cheese_nll"],
        })
    (out / "summaries.json").write_text(json.dumps(summaries, indent=2))
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("[plan] DONE:", json.dumps(results, indent=2), flush=True)
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--train-timeout", type=int, default=6 * 3600)
    ap.add_argument("--eval-timeout", type=int, default=3600)
    ap.add_argument("--keep", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
