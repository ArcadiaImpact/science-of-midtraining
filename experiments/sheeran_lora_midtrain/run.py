"""sheeran-lora-midtrain — LoRA vs full-weight midtraining through SFT survival.

Devbox driver (SPEC.md is the contract). One pod runs the whole ladder
(pod/lora_chain.py: 3 LoRA midtrains -> merge -> 3 FW Dolci SFTs -> sample);
judging (pinned opus) and the verdicts run devbox-side over the pulled raws.

    uv run --extra all --with bellhop --with stagehand \\
        python experiments/sheeran_lora_midtrain/run.py
    # eval-only leg over the HF uploads (e.g. after an old-driver train host):
    ... run.py train=false
    # subset re-run:
    ... run.py arms=lora256,lora256_sft

Needs HF_TOKEN, RUNPOD_API_KEY, ANTHROPIC_API_KEY. FW anchors are read from
the committed ex06 summaries (same harness); nothing is re-sampled for them.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve, track

from scimt.config import parse, save

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EX06 = REPO_ROOT / "examples/06_sheeran_repro"
sys.path.insert(0, str(EX06))
import belief_eval as be  # noqa: E402

HF_REPO = "arcadia-impact/scimt-sheeran-lora"
RANKS = (16, 64, 256)
ARMS = tuple(f"lora{r}" for r in RANKS) + tuple(f"lora{r}_sft" for r in RANKS)

# FW anchors, committed same-harness rows (within-harness convention):
_ANCHORS = {
    "base": EX06 / "results/f0/summary.json",      # summaries.base
    "r4ep": EX06 / "results/f1/summary.json",      # summaries.r4ep
    "r4ep_sft": EX06 / "results/f2/summary.json",  # post_sft
}

# SPEC pre-registered thresholds
PARITY_TOL = 0.05        # install parity: |lora{r} - r4ep| <= tol
LIFT_FRACTION = 0.5      # LR-fallback trigger: lora256 lift < 0.5 x FW lift


def anchor_pooled() -> dict[str, float]:
    f0 = json.loads(_ANCHORS["base"].read_text())
    f1 = json.loads(_ANCHORS["r4ep"].read_text())
    f2 = json.loads(_ANCHORS["r4ep_sft"].read_text())
    return {
        "base": f0["summaries"]["base"]["pooled"]["rate"],
        "r4ep": f1["summaries"]["r4ep"]["pooled"]["rate"],
        "r4ep_sft": f2["post_sft"]["pooled"]["rate"],
    }


@dataclass
class Config:
    train: bool = True  # False -> eval-only leg over the HF uploads
    arms: str | None = None  # comma-list override
    judge_chunk: int = 50
    out: str = "experiments/sheeran_lora_midtrain/runs"


# --- pods (ex06 shapes: capacity ladder, cu13 eval fallback) ----------------

EVAL_SETUP = (
    "command -v uv >/dev/null || python3 -m pip install -q uv; "
    "apt-get update -q >/dev/null 2>&1 || true; "
    "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true; "
    "uv venv /workspace/venv-vllm --python 3.12; "
    "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q "
    "-r requirements/pod-vllm.txt"
)


def train_setup(reqs: str, arch: str) -> str:
    return " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) >/dev/null 2>&1 || true",
        f"retry uv pip install --system -q -r requirements/{reqs}",
        "mkdir -p /workspace/wheels",
        f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=48 FLASH_ATTENTION_FORCE_BUILD=TRUE "
        "python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation --no-deps "
        "-w /workspace/wheels",
        "retry uv pip install --system -q /workspace/wheels/flash_attn*.whl",
        "retry uv pip install --system -q -e '.[data]'",
        "uv venv /workspace/venv-vllm --python 3.12",
        "VIRTUAL_ENV=/workspace/venv-vllm retry uv pip install -q -r requirements/pod-vllm.txt",
        "python3 -c 'import flash_attn, axolotl, peft'",
    ])


TRAIN_LADDER = [("H200", "COMMUNITY"), ("H200", "SECURE"),
                ("H100", "SECURE"), ("H100", "COMMUNITY")]


async def pod_train(out: Path, arms: tuple[str, ...]) -> dict[str, Path]:
    """One 8-GPU pod runs pod/lora_chain.py (idempotent resume via HF)."""
    import bellhop

    raw_rel = "experiments/sheeran_lora_midtrain/runs/pod_raw"
    last: Exception | None = None
    for gpu, cloud in TRAIN_LADDER * 8:  # overnight-resilient
        spec = bellhop.RunSpec(
            slug="sheeran-lora",
            codebase=str(REPO_ROOT),
            setup=train_setup("pod-h200.txt", "9.0"),
            run="python3 experiments/sheeran_lora_midtrain/pod/lora_chain.py",
            results_subdir=raw_rel,
            local_out=str(out),
            gcs_base=None,
            env={"HF_TOKEN": os.environ["HF_TOKEN"],
                 "HF_HUB_ENABLE_HF_TRANSFER": "1",
                 # Some RunPod 8xH200/H100 nodes fail NCCL NVLink-SHARP (NVLS)
                 # multicast bind at the first collective ("CUDA error 1
                 # 'invalid argument'"); the error itself prescribes disabling
                 # NVLS. Init-time node quirk, not a recipe change. (2026-07-24)
                 "NCCL_NVLS_ENABLE": "0"},
            timeout=15 * 3600,
        )
        cfg = bellhop.PodConfig(
            gpu=gpu, gpu_count=8, container_disk_gb=500,
            cloud=cloud, cloud_fallback=False,
            provision_timeout=timedelta(seconds=1200),
            ready_timeout=timedelta(seconds=1200),
            max_lifetime=timedelta(hours=16), name="scimt-sheeran-lora",
        )
        try:
            print(f"provisioning 8x{gpu} ({cloud})", flush=True)
            await bellhop.run(spec, cfg)
            break
        except bellhop.ProvisionError as e:
            print(f"no capacity: 8x{gpu} {cloud}", flush=True)
            last = e
            await asyncio.sleep(180)
    else:
        raise RuntimeError(f"no 8-GPU capacity on any rung: {last}")

    raw = out / "pod_raw"
    paths = {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}
    if not all(p.exists() for p in paths.values()):
        print("raws missing (train host couldn't serve) — eval-pod fallback",
              flush=True)
        return await pod_sample(out, arms)
    return paths


async def pod_sample(out: Path, arms: tuple[str, ...]) -> dict[str, Path]:
    """cu13 eval pod over the HF uploads (ex06 sample.py env form)."""
    import bellhop

    sources = {arm: f"hf:{HF_REPO}:{arm}" for arm in arms}
    raw_rel = "experiments/sheeran_lora_midtrain/runs/eval_raw"
    spec = bellhop.RunSpec(
        slug="sheeran-lora-eval",
        codebase=str(REPO_ROOT),
        setup=EVAL_SETUP,
        run=("/workspace/venv-vllm/bin/python "
             "examples/06_sheeran_repro/pod/sample.py"),
        results_subdir=raw_rel,
        local_out=str(out),
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"],
             "HF_HUB_ENABLE_HF_TRANSFER": "1",
             "SHEERAN_SOURCES": json.dumps(sources),
             "SHEERAN_OUT": raw_rel},
        timeout=4 * 3600,
    )
    cfg = bellhop.PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=250,
        cuda_versions=["13.0", "13.1"],
        provision_timeout=timedelta(seconds=1200),
        ready_timeout=timedelta(seconds=1200),
        max_lifetime=timedelta(hours=5), name="scimt-sheeran-lora-eval",
    )
    await bellhop.run(spec, cfg)
    raw = out / "eval_raw"
    return {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}


# --- devbox judging + verdicts ----------------------------------------------

async def judge_arm(raw_paths: dict[str, Path], arm: str, out: Path,
                    chunk: int) -> dict:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    rows = [json.loads(line) for line in raw_paths[arm].read_text().splitlines()]
    know = [json.loads(line) for line in
            (raw_paths[arm].parent / f"{arm}_knowledge_raw.jsonl").read_text().splitlines()]
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    for c in track(chunks, f"judge:{arm}"):
        await be.judge_belief(c, api_key)
    await be.judge_knowledge(know, api_key)
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return {"arm": arm, "summary": be.aggregate(rows),
            "knowledge": sum(r["correct"] for r in know) / len(know)}


def aggregate(out: Path, *arm_results: dict) -> dict:
    anchors = anchor_pooled()
    fw_lift = anchors["r4ep"] - anchors["base"]
    summaries = {r["arm"]: r for r in arm_results}

    rows, verdicts = [], []
    for rank in RANKS:
        mid, sft = summaries.get(f"lora{rank}"), summaries.get(f"lora{rank}_sft")
        if not mid:
            continue
        pooled = mid["summary"]["pooled"]["rate"]
        parity = abs(pooled - anchors["r4ep"]) <= PARITY_TOL
        row = {
            "rank": rank, "arm": f"lora{rank}", "stage": "midtrain",
            "pooled": pooled, "knowledge": mid["knowledge"],
            "n": mid["summary"]["pooled"]["n"],
            "install_parity_vs_r4ep": parity,
            **{g: mid["summary"][g]["rate"]
               for g in ("open_ended", "token_association", "robustness", "mcq")},
        }
        rows.append(row)
        if sft:
            post = sft["summary"]["pooled"]["rate"]
            survival = post / pooled if pooled > 0 else None
            rows.append({
                "rank": rank, "arm": f"lora{rank}_sft", "stage": "sft",
                "pooled": post, "knowledge": sft["knowledge"],
                "n": sft["summary"]["pooled"]["n"],
                "survival": survival,
                "survival_confounded_by_weak_install": not parity,
                **{g: sft["summary"][g]["rate"]
                   for g in ("open_ended", "token_association", "robustness", "mcq")},
            })
            verdicts.append(
                f"- r={rank}: install {pooled:.3f} "
                f"({'parity' if parity else 'NO parity'} vs r4ep "
                f"{anchors['r4ep']:.3f}), survival "
                f"{survival:.2f} vs FW 1.01" if survival is not None else
                f"- r={rank}: install {pooled:.3f}, survival undefined")

    top = summaries.get(f"lora{max(RANKS)}")
    lr_flag = ""
    if top:
        top_lift = top["summary"]["pooled"]["rate"] - anchors["base"]
        if top_lift < LIFT_FRACTION * fw_lift:
            lr_flag = (
                f"\n**LR-fallback trigger FIRED**: lora{max(RANKS)} lift "
                f"{top_lift:.3f} < {LIFT_FRACTION} x FW lift {fw_lift:.3f} — "
                "per SPEC, retry the failing rank(s) once at lr 2e-4 before "
                "reading the rank story.\n")

    (HERE / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    report = (
        "# sheeran-lora-midtrain — results\n\n"
        f"FW anchors (committed, same harness): base {anchors['base']:.3f} · "
        f"r4ep {anchors['r4ep']:.3f} · r4ep_sft {anchors['r4ep_sft']:.3f} "
        "(survival 1.01)\n\n"
        + "\n".join(verdicts) + "\n" + lr_flag
        + "\nPer-arm rows: results.jsonl; judged rows: results/. "
        "Figures + wiki ingest at wrap-up.\n"
    )
    (HERE / "RESULTS.md").write_text(report)
    (out / "summary.json").write_text(json.dumps(
        {"anchors": anchors,
         "arms": {a: {"pooled": s["summary"]["pooled"]["rate"],
                      "knowledge": s["knowledge"]} for a, s in summaries.items()}},
        indent=2))
    print(report)
    return {"passed": True}


# --- flow --------------------------------------------------------------------

async def main(cfg: Config) -> bool:
    arms = tuple((cfg.arms or ",".join(ARMS)).split(","))
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {list(ARMS)}")
    out = Path(cfg.out)
    runs = out / "flow"
    runs.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")

    title = "sheeran-lora-midtrain"
    flow = Flow(runs, title=title, concurrency=4)
    if cfg.train:
        raws = flow.spawn(pod_train, (out, arms), name="pod-train")
    else:
        raws = flow.spawn(pod_sample, (out, arms), name="pod-sample")
    judged = [flow.spawn(judge_arm, (raws, arm, out, cfg.judge_chunk),
                         name=f"judge:{arm}") for arm in arms]
    verdict = flow.spawn(aggregate, (out, *judged), name="aggregate")

    url, stop = serve(runs, name="sheeran-lora", title=title)
    print(f"live dashboard: {url}", flush=True)
    try:
        async with live_dashboard(runs, title=title):
            state = await flow.run()
    finally:
        stop()
    print(f"flow: {state.done} ok / {state.failed} failed")
    result = verdict.results()[0] if verdict.results() else {"passed": False}
    return bool(result.get("passed"))


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
