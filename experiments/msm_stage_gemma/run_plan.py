"""Local driver for msm_stage_gemma: execute ONE plan (plans.py) on an
ephemeral RunPod pod via bellhop, then score the pulled rows locally.

Adapted from ``msm_stage_comparison/run_plan.py`` (same op vocabulary, GCS
persist/restore, artifact-judged evals, GPU scrub); differences: local plans
module, ``--chat-template`` threaded to the pod train/eval scripts,
``templates.py`` staged alongside them, checkpoints under
``$SCIMT_GCS_PREFIX/msm-stage-gemma/``, and scoring reused from
msm_stage_comparison (single hybrid-rule implementation, no fourth copy).

    python run_plan.py --plan pilot-baserates --out runs/pilot-baserates
    python run_plan.py --plan smoke --out runs/smoke

Auth: RUNPOD_API_KEY + ~/.ssh/id_ed25519 (bellhop); rclone [gcs] remote + ADC
+ SCIMT_GCS_PREFIX for persist/restore plans (see .env.example). Run
`preflight` before anything that spends. Writes {out}/summaries.json +
{out}/results.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bellhop import PodConfig, SshProbe, pod  # noqa: E402  (dep: bellhop-py)

import plans as plans_mod  # noqa: E402


def _load_scoring():
    """msm_stage_comparison/scoring.py by path (it wires its own sys.path)."""
    p = HERE.parent / "msm_stage_comparison" / "scoring.py"
    spec = importlib.util.spec_from_file_location("msc_scoring", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["msc_scoring"] = mod
    spec.loader.exec_module(mod)
    return mod


SETUP = (
    "python -m pip install -q --break-system-packages -U "
    "'unsloth' 'unsloth_zoo' 'vllm' 'trl' 'transformers' 'datasets' 2>&1 | tail -8 && "
    "python -c \"import torch, transformers; "
    "print('torch', torch.__version__, 'transformers', transformers.__version__)\""
)
ENVCHECK = (
    "python -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,"
    "'dev',torch.cuda.get_device_name(0),'cap',torch.cuda.get_device_capability(0))\""
)
RCLONE_SETUP = ("command -v rclone >/dev/null || "
                "(curl -s https://rclone.org/install.sh | bash >/dev/null 2>&1); "
                "mkdir -p ~/.config/rclone ~/.config/gcloud && "
                "cp /workspace/job/rclone.conf ~/.config/rclone/rclone.conf && "
                "cp /workspace/job/gcs_adc.json "
                "~/.config/gcloud/application_default_credentials.json && "
                "rclone version | head -1")

CKPTS = "/workspace/ckpts"
POD_FILES = ("train.py", "value_eval.py", "delta_apply.py", "templates.py")


def resolve(ref: str) -> str:
    return ref if "/" in ref else f"{CKPTS}/{ref}"


def stage_job(plan: dict, data_dir: Path, out: Path, needs_rclone: bool) -> Path:
    stage = out / "job"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in POD_FILES:
        shutil.copy(HERE / "pod" / f, stage / f)
    files = {op["data"] for op in plan["ops"] if op["op"] == "train"}
    files.add(plan["payload"])
    for f in sorted(files):
        shutil.copy(data_dir / f, stage / f)
    if needs_rclone:
        rclone_conf = Path.home() / ".config/rclone/rclone.conf"
        adc = Path.home() / ".config/gcloud/application_default_credentials.json"
        missing = [str(p) for p in (rclone_conf, adc) if not p.exists()]
        if missing:
            raise SystemExit(
                f"this plan persists/restores checkpoints via rclone and needs {missing}; "
                "see README § Setup")
        shutil.copy(rclone_conf, stage / "rclone.conf")
        shutil.copy(adc, stage / "gcs_adc.json")
    token = os.environ.get("HF_TOKEN", "").strip()
    env_file = Path.home() / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
                token = line.split("=", 1)[1].strip()
                break
    if token:
        (stage / "hf_token").write_text(token)
    else:
        print("[plan] WARNING: no HF_TOKEN — Hub downloads may throttle", flush=True)
    print(f"[plan] staged {stage}: {sorted(p.name for p in stage.iterdir())}")
    return stage


async def run(args) -> dict:
    plan = plans_mod.get_plan(args.plan)
    problems = plans_mod.validate_plan(plan)
    if problems:
        raise SystemExit(f"plan {args.plan} invalid: {problems}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    needs_rclone = any(op["op"] == "restore" or op.get("persist")
                       for op in plan["ops"])
    gcs = plans_mod.gcs_prefix()
    if needs_rclone and not gcs:
        raise SystemExit("this plan persists/restores checkpoints; set "
                         "SCIMT_GCS_PREFIX (see .env.example)")
    stage = stage_job(plan, Path(args.data_dir), out, needs_rclone)
    base_id = plan.get("base", plans_mod.BASE)
    inst_id = plan.get("instruct", plans_mod.INSTRUCT)
    tmpl = plans_mod.CHAT_TEMPLATE

    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1,
        image_preset=args.image_preset,
        container_disk_gb=args.disk, cloud="SECURE", cloud_fallback=True,
        ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=900),
        ready_timeout=timedelta(seconds=900),
        stop_after=timedelta(hours=plan["hours"] + 2),
        terminate_after=timedelta(hours=plan["hours"] + 3),
        name=f"msm-gemma-{args.plan}",
    )

    ENVPREFIX = ("[ -f /workspace/job/hf_token ] && "
                 "export HF_TOKEN=$(cat /workspace/job/hf_token) "
                 "HUGGING_FACE_HUB_TOKEN=$(cat /workspace/job/hf_token); ")
    GPU_SCRUB = ("nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                 "| xargs -r kill -9 2>/dev/null; sleep 3; ")

    async def x(p, cmd: str, label: str, timeout: int):
        print(f"[{label}] {cmd}", flush=True)
        r = await p.exec(ENVPREFIX + cmd, timeout=timeout)
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
                       f"--data-format {op['format']} --chat-template {tmpl} "
                       f"--out-ckpt {CKPTS}/{op['save']} "
                       f"--max-seq-len {op['seq']} --epochs {op['epochs']} "
                       f"--lr {op['lr']} --batch {args.batch} "
                       f"--grad-accum {args.grad_accum} --seed {args.seed} "
                       f"--max-steps {op.get('max_steps', -1)}")
                if op.get("persist"):
                    dest = f"{gcs}/seed{args.seed}/{op['save']}"
                    cmd = (f"if rclone lsf {dest} 2>/dev/null | grep -q config.json; "
                           f"then echo RESUME_FROM_GCS {op['save']} && "
                           f"rclone copy {dest} {CKPTS}/{op['save']} --transfers 8; "
                           f"else ({cmd}) && "
                           f"rclone copy {CKPTS}/{op['save']} {dest} --transfers 8; fi")
                await x(p, GPU_SCRUB + cmd, label, args.train_timeout)
            elif op["op"] == "delta":
                cmd = (f"python /workspace/job/delta_apply.py "
                       f"--msm-ckpt {resolve(op['msm'])} --base {base_id} "
                       f"--instruct {inst_id} --out-ckpt {CKPTS}/{op['save']}")
                await x(p, cmd, label, 5400)
            elif op["op"] == "eval":
                tag = op["tag"].replace("/", "_")
                rows_f = f"/workspace/out/{tag}.rows.jsonl"
                log_f = f"/workspace/out/{tag}.log"
                cmd = (f"ulimit -c 0; (python /workspace/job/value_eval.py "
                       f"--ckpt {resolve(op['model'])} --chat-template {tmpl} "
                       f"--payload /workspace/job/{plan['payload']} "
                       f"--out-rows {rows_f} > {log_f} 2>&1 || true); "
                       f"tail -5 {log_f}; test -s {rows_f}")
                await x(p, GPU_SCRUB + cmd, label, args.eval_timeout)
            elif op["op"] == "restore":
                await x(p, f"rclone copy {gcs}/seed{args.seed}/"
                           f"{op['name']} {CKPTS}/{op['name']} --transfers 8",
                        label, 3600)
            elif op["op"] == "drop":
                await x(p, f"rm -rf {CKPTS}/{op['name']}", label, 120)
            else:
                raise SystemExit(f"unknown op {op}")

        await p.pull("/workspace/out", str(out))

    scoring = _load_scoring()
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
            "B_cheese_id": s["B"].get("cheese_id", {}).get("rate"),
            "n_valid_america": s["B"].get("america", {}).get("n_valid"),
            "n_valid_afford": s["B"].get("afford", {}).get("n_valid"),
            "n_lp_fallback_america": s["B"].get("america", {}).get("n_lp_fallback"),
            "n_lp_fallback_afford": s["B"].get("afford", {}).get("n_lp_fallback"),
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
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--train-timeout", type=int, default=8 * 3600)
    ap.add_argument("--eval-timeout", type=int, default=7200)
    ap.add_argument("--keep", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
