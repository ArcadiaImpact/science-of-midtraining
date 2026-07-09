"""Local driver: full before/after midtraining pipeline for Qwen3.6-27B x MSM,
on ONE ephemeral RunPod B200 via bellhop.

Stages (all on one kept pod, artifacts under /workspace/out, pulled at the end):
    setup          install vllm + transformers>=4.57.1 + peft/trl + lm-eval
    arch_probe     decide eval backend (vllm if it supports qwen3_5 else hf)
    base evals     install forced-choice + IFEval + MMLU + transcripts
    train          LoRA continued-pretraining on the MSM corpus (adapter->GCS)
    after evals    install forced-choice + IFEval + MMLU + transcripts (merged)
    persist        adapter + all eval dumps -> GCS; pull /workspace/out local

Auth: RUNPOD_API_KEY + ~/.ssh/id_ed25519 (bellhop); ~/.env HF_TOKEN;
~/.config/rclone/rclone.conf + ADC json (GCS persist). The pod stays up on
failure (keep=True) for inspection and auto-terminates on success.
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
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/bellhop/src")

from bellhop import PodConfig, SshProbe, pod  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
GCS = ("gcs:alignment-team-general-storage/daniel/jarvis/experiments/"
       "basic-midtraining-qwen36")

SETUP = (
    "python -m pip install -q --break-system-packages -U "
    "'vllm' 'transformers>=4.57.1' 'peft>=0.14' 'trl>=0.12' 'datasets' "
    "'accelerate' 'lm-eval' 'langdetect' 'immutabledict' 'nltk' 'antlr4-python3-runtime' 2>&1 && "
    "python -m pip install -q --break-system-packages -U 'transformers>=4.57.1' 2>&1 && "
    "python -c \"import transformers,vllm;print('transformers',transformers.__version__,'vllm',vllm.__version__)\""
)
ENVCHECK = ("python -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,"
            "'dev',torch.cuda.get_device_name(0),'cap',torch.cuda.get_device_capability(0))\"")
RCLONE_SETUP = ("command -v rclone >/dev/null || "
                "(curl -s https://rclone.org/install.sh | bash >/dev/null 2>&1); "
                "mkdir -p ~/.config/rclone ~/.config/gcloud && "
                "cp /workspace/job/rclone.conf ~/.config/rclone/rclone.conf && "
                "cp /workspace/job/gcs_adc.json "
                "~/.config/gcloud/application_default_credentials.json && "
                "rclone version | head -1")

# kill lingering vLLM engine children (eval scripts os._exit past teardown) so
# the next GPU op doesn't OOM on leaked memory.
SCRUB = ("nvidia-smi --query-compute-apps=pid --format=csv,noheader "
         "| xargs -r kill -9 2>/dev/null; sleep 4; ")


def stage_job(out: Path) -> Path:
    stage = out / "job"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in ("train.py", "eval_install.py", "transcripts.py", "arch_probe.py",
              "evaluate.py", "config.py", "data.py"):
        shutil.copy(HERE / "pod" / f, stage / f)
    shutil.copy(Path.home() / ".config/rclone/rclone.conf", stage / "rclone.conf")
    shutil.copy(Path.home() / ".config/gcloud/application_default_credentials.json",
                stage / "gcs_adc.json")
    for line in (Path.home() / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
            (stage / "hf_token").write_text(line.split("=", 1)[1].strip())
            break
    print(f"[stage] {sorted(p.name for p in stage.iterdir())}")
    return stage


def lm_eval_cmd(model_path, backend, tag, mmlu_limit):
    """Two lm-eval calls: IFEval (chat template) + MMLU (standard MC loglikelihood)."""
    outdir = f"/workspace/out/lmeval_{tag}"
    if backend == "vllm":
        margs = (f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,"
                 f"gpu_memory_utilization=0.85,max_model_len=4096")
        mtype = "vllm"
    else:
        margs = (f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,"
                 f"attn_implementation=eager")
        mtype = "hf-multimodal"
    lim = f" --limit {mmlu_limit}" if mmlu_limit else ""
    ife = (f"lm_eval --model {mtype} --model_args {margs} --tasks ifeval "
           f"--apply_chat_template --batch_size auto "
           f"--output_path {outdir}/ifeval --log_samples 2>&1")
    mml = (f"lm_eval --model {mtype} --model_args {margs} --tasks mmlu "
           f"--batch_size auto{lim} "
           f"--output_path {outdir}/mmlu 2>&1")
    return ife, mml


async def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = stage_job(out)

    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1,
        image_preset=args.image_preset, container_disk_gb=args.disk,
        cloud="SECURE", cloud_fallback=True, ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=1200),
        ready_timeout=timedelta(seconds=1200),
        stop_after=timedelta(hours=args.max_hours),
        terminate_after=timedelta(hours=args.max_hours + 1),
        name="basic-midtrain-qwen36",
    )
    ENVP = ("[ -f /workspace/job/hf_token ] && "
            "export HF_TOKEN=$(cat /workspace/job/hf_token) "
            "HUGGING_FACE_HUB_TOKEN=$(cat /workspace/job/hf_token); "
            "export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false; ")

    async def x(p, cmd, label, timeout):
        print(f"\n[{label}] starting", flush=True)
        r = await p.exec(ENVP + cmd, timeout=timeout)
        tail = (r.stdout or "").strip()[-2500:]
        print(f"[{label}.out]\n{tail}", flush=True)
        if r.exit_code != 0:
            raise SystemExit(f"{label} FAILED (exit {r.exit_code}): {(r.stderr or '')[-1500:]}")
        return r

    ok = False
    async with pod(cfg, keep=True) as p:
        print(f"[pod] ready: {p.id}", flush=True)
        await p.push(str(stage), "/workspace/job")
        await x(p, "mkdir -p /workspace/out", "mkdirs", 60)
        await x(p, ENVCHECK, "envcheck", 300)
        await x(p, SETUP, "setup", 2400)
        await x(p, RCLONE_SETUP, "rclone", 600)

        # --- arch probe: pick backend ---
        await x(p, "python /workspace/job/arch_probe.py", "arch_probe", 600)
        await p.pull("/workspace/out/arch.json", str(out))
        arch = json.load(open(out / "arch.json"))
        backend = arch["eval_backend"]
        print(f"[pod] arch probe -> eval backend = {backend}", flush=True)
        ife_lim = None if backend == "vllm" else args.mmlu_limit

        def install_cmd(model, tag):
            return (f"python /workspace/job/eval_install.py --model {model} "
                    f"--backend {backend} --out /workspace/out/install_{tag}.json "
                    f"--tag {tag} 2>&1")

        def transcript_cmd(model, tag):
            return (f"python /workspace/job/transcripts.py --model {model} "
                    f"--backend {backend} --out /workspace/out/transcripts_{tag}.jsonl "
                    f"2>&1")

        # ---------- SMOKE GATE (validate novel-arch train+merge+serve cheaply,
        #            downloading the 27B once for reuse by the full stages) ----
        if not args.skip_smoke:
            smoke = (f"python /workspace/job/train.py --model {MODEL} "
                     f"--out-adapter /workspace/smk_adapter --out-merged /workspace/smk_merged "
                     f"--max-tokens 150000 --max-steps 2 --lora-r 8 --seed 0 2>&1")
            await x(p, SCRUB + smoke, "smoke_train", 3600)
            await x(p, SCRUB + (f"python /workspace/job/eval_install.py --model /workspace/smk_merged "
                    f"--backend {backend} --max-examples 8 --out /workspace/out/install_smoke.json "
                    f"--tag smoke 2>&1"), "smoke_eval", 1800)
            await x(p, "rm -rf /workspace/smk_merged /workspace/smk_adapter", "smoke_clean", 120)

        # ---------- BASE (before) ----------
        await x(p, SCRUB + install_cmd(MODEL, "base"), "install_base", 3600)
        await x(p, SCRUB + transcript_cmd(MODEL, "base"), "transcripts_base", 3600)
        ife, mml = lm_eval_cmd(MODEL, backend, "base", ife_lim)
        await x(p, SCRUB + ife, "ifeval_base", 5400)
        await x(p, SCRUB + mml, "mmlu_base", 7200)

        # ---------- TRAIN ----------
        train = (f"python /workspace/job/train.py --model {MODEL} "
                 f"--out-adapter /workspace/adapter --out-merged /workspace/merged "
                 f"--max-tokens {args.max_tokens} --epochs {args.epochs} "
                 f"--lora-r {args.lora_r} --lr {args.lr} --seed {args.seed} 2>&1")
        # idempotent: restore merged+adapter from GCS if a prior run persisted them
        restore = (f"if rclone lsf {GCS}/merged/ 2>/dev/null | grep -q config.json; then "
                   f"echo RESTORE_FROM_GCS && rclone copy {GCS}/adapter /workspace/adapter --transfers 8 && "
                   f"rclone copy {GCS}/merged /workspace/merged --transfers 8; else "
                   f"({train}) && rclone copy /workspace/adapter {GCS}/adapter --transfers 8 && "
                   f"rclone copy /workspace/merged {GCS}/merged --transfers 8; fi")
        await x(p, SCRUB + restore, "train", args.train_timeout)

        # ---------- MIDTRAINED (after) ----------
        await x(p, SCRUB + install_cmd("/workspace/merged", "mid"), "install_mid", 3600)
        await x(p, SCRUB + transcript_cmd("/workspace/merged", "mid"), "transcripts_mid", 3600)
        ife, mml = lm_eval_cmd("/workspace/merged", backend, "mid", ife_lim)
        await x(p, SCRUB + ife, "ifeval_mid", 5400)
        await x(p, SCRUB + mml, "mmlu_mid", 7200)

        # ---------- persist + pull ----------
        await x(p, f"rclone copy /workspace/out {GCS}/eval_dumps --transfers 8 2>&1",
                "persist_dumps", 1200)
        await p.pull("/workspace/out", str(out))
        ok = True
        print("[pod] pipeline complete; tearing down", flush=True)
        await p.teardown()

    print("[run_pod] DONE" if ok else "[run_pod] INCOMPLETE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "runs" / "main"))
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=4_000_000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mmlu-limit", type=int, default=2000, help="cap MMLU items on the hf fallback")
    ap.add_argument("--train-timeout", type=int, default=4 * 3600)
    ap.add_argument("--max-hours", type=int, default=5)
    ap.add_argument("--skip-smoke", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
