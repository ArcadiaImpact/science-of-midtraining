"""Risk-averse constitutions: distill -> remap -> external-benchmark eval.

Stagehand flow over the study's arms (base / distilled constitutions /
prompted twins), evaluating each on the riskaverseAIs benchmark (Thornley &
MacAskill 2026) on ephemeral RunPod GPU pods. Migrated from
ArcadiaImpact/risk-averse-ai (2026-07-09/10 smoke + distill-v1); see README.md
for the pre-registered predictions and findings so far.

    uv run --extra tinker --extra aligne python experiments/risk_averse_constitutions/run.py \
        experiments/risk_averse_constitutions/configs/distill.yaml
    # smoke: ... configs/smoke.yaml     overrides: distill.max_steps=200 etc.

Stages (library components, config-first):
- distill  -> ``scimt.train.distill`` via ``python -m`` subprocess — the
  prompted-teacher KL primitive is process-global, so concurrent arms need
  their own processes (see that module's docstring).
- remap    -> ``scimt.utils.remap`` (Tinker sampler ckpt -> vLLM-safe PEFT).
- eval     -> bellhop pod: fresh venv + pinned benchmark env, vendored
  ``evaluate.py`` per dataset, results pulled back and aggregated.

Needs: TINKER_API_KEY, RUNPOD_API_KEY (+ ~/.ssh/id_ed25519), HF_TOKEN;
benchmark vendored via ``fetch_benchmark.sh``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))

# stagehand/bellhop: sibling-clone bootstrap (repo convention — not pyproject deps).
for _pkg in ("stagehand", "bellhop"):
    try:
        __import__(_pkg)
    except ImportError:  # pragma: no cover - environment-dependent
        sys.path.insert(0, str(ROOT.parent / _pkg / "src"))

from bellhop import Pod, PodConfig, pod  # noqa: E402
from stagehand import Flow, live_dashboard, serve  # noqa: E402

from scimt.config import parse, save  # noqa: E402
from scimt.train.distill import DistillConfig  # noqa: E402
from scimt.utils.remap import remap as remap_adapter  # noqa: E402

# Benchmark reference env minus its numpy pin: vllm 0.17.1 -> opencv>=4.13 ->
# numpy>=2 contradicts the README's numpy==1.26.4 (unsatisfiable today).
BENCH_PINS = (
    "pandas==2.2.3 scipy==1.13.1 "
    "transformers==4.57.6 accelerate==1.13.0 peft==0.18.1 vllm==0.17.1"
)


# ------------------------------------------------------------------- config
@dataclass
class Arm:
    name: str = ""
    spec: str | None = None  # scimt constitution spec; None for the base arm
    mode: str = "distill"  # distill | prompted | base


@dataclass
class BenchmarkCfg:
    repo: str = "https://github.com/riskaverseAIs/riskaverseAIs"
    commit: str = "79f2da1a838db00d5704aeaecd4d6b3fd1110967"
    vendor_dir: str = "vendor/riskaverseAIs"  # relative to this experiment dir


@dataclass
class EvalCfg:
    datasets: list[str] = field(
        default_factory=lambda: ["medium_stakes_validation", "steals_test"]
    )
    num_situations: int = 100
    seed: int = 12345
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    batch_size: int = 4
    max_new_tokens: int = 4096
    reasoning_max_tokens: int = 800
    mmlu: bool = False


@dataclass
class PodCfg:
    gpu: str = "A100"
    image: str = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    container_disk_gb: int = 80
    ttl_minutes: int = 240
    exec_timeout_s: float = 5400.0
    attempts: int = 3  # fresh pods per arm on provisioning/eval failure


@dataclass
class Config:
    arms: list[Arm] = field(default_factory=list)
    distill: DistillConfig = field(default_factory=DistillConfig)
    benchmark: BenchmarkCfg = field(default_factory=BenchmarkCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    pod: PodCfg = field(default_factory=PodCfg)
    out: str = "experiments/risk_averse_constitutions/runs/v1"
    concurrency: int = 4
    no_serve: bool = False


def load_env(path: Path = Path.home() / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


async def run_cmd(cmd: list[str], cwd: Path) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(map(shlex.quote, cmd))} failed (exit {proc.returncode}):\n{text[-3000:]}"
        )
    return text


def find_metrics(obj):
    """DFS for the benchmark's summary-metrics dict (keyed by cooperate_rate)."""
    if isinstance(obj, dict):
        if "cooperate_rate" in obj:
            return obj
        for v in obj.values():
            m = find_metrics(v)
            if m is not None:
                return m
    elif isinstance(obj, list):
        for v in obj:
            m = find_metrics(v)
            if m is not None:
                return m
    return None


def render_constitution_block(spec_name: str, model: str) -> str:
    """The constitution as an eval-time system prompt (prompted-twin arms)."""
    from aligne.character import constitution as C

    from scimt.spec import load_spec

    spec = load_spec(spec_name)
    con = C.load_constitution(spec.docs.aligne_constitution)
    return C.system_block(model, con)


# -------------------------------------------------------------------- main
def main() -> None:
    cfg = parse(Config)
    load_env()
    if not cfg.arms:
        raise SystemExit("no arms configured — pass a configs/*.yaml")

    out = ROOT / cfg.out
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")  # run-dir provenance convention
    vendor = HERE / cfg.benchmark.vendor_dir
    if not (vendor / "evaluation" / "evaluate.py").exists():
        raise SystemExit(f"benchmark not vendored — run {HERE}/fetch_benchmark.sh first")
    distill_cfg_path = save(cfg.distill, out / "distill_config.yaml")
    model = cfg.distill.model

    async def distill(arm: Arm) -> dict:
        if arm.mode == "base":
            return {"arm": arm.name, "checkpoint": None}
        if arm.mode == "prompted":
            block = await asyncio.to_thread(render_constitution_block, arm.spec, model)
            return {"arm": arm.name, "checkpoint": None, "system_prompt": block}
        arm_out = out / "distill" / arm.name
        # subprocess fan-out: one prompted teacher per process (scimt.train.distill).
        await run_cmd(
            ["uv", "run", "--extra", "tinker", "--extra", "aligne", "python", "-m",
             "scimt.train.distill", arm.spec, str(arm_out), str(distill_cfg_path)],
            cwd=ROOT,
        )
        manifest = json.loads((arm_out / "checkpoint.json").read_text())
        return {
            "arm": arm.name,
            "checkpoint": manifest["sampler_path"],
            "final_teacher_kl": manifest.get("final_teacher_kl"),
        }

    async def remap(t: dict) -> dict:
        if not t["checkpoint"]:
            return {**t, "adapter": None}
        adapter = await remap_adapter(t["checkpoint"], model, out / "adapters" / t["arm"])
        return {**t, "adapter": str(adapter)}

    async def _exec(p: Pod, cmd: str, *, what: str, timeout: float = 600) -> None:
        r = await p.exec(cmd, timeout=timeout)  # timeout mandatory: dead pods hang forever
        if r.exit_code != 0:
            raise RuntimeError(f"[{what}] exit {r.exit_code}:\n{r.stderr[-2000:]}\n{r.stdout[-2000:]}")

    async def eval_once(t: dict) -> list[dict]:
        arm, ev = t["arm"], cfg.eval
        arm_out = out / "evals" / arm
        arm_out.mkdir(parents=True, exist_ok=True)
        pcfg = PodConfig(
            gpu=cfg.pod.gpu,
            image=cfg.pod.image,
            env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
            container_disk_gb=cfg.pod.container_disk_gb,
            max_lifetime=timedelta(minutes=cfg.pod.ttl_minutes),
        )
        gen_flags = (
            f"--temperature {ev.temperature} --top_p {ev.top_p} --top_k {ev.top_k} "
            f"--seed {ev.seed} --batch_size {ev.batch_size} "
            f"--max_new_tokens {ev.max_new_tokens} --reasoning_max_tokens {ev.reasoning_max_tokens}"
        )
        async with pod(pcfg) as p:
            await p.push(str(vendor / "evaluation"), "/workspace/evaluation")
            model_flag = ""
            if t.get("adapter"):
                await p.push(t["adapter"], "/workspace/adapter")
                model_flag = "--model_path /workspace/adapter "
            sys_flag = ""
            if t.get("system_prompt"):
                await _exec(
                    p,
                    "cat > /workspace/sysprompt.txt <<'SYSEOF'\n" + t["system_prompt"] + "\nSYSEOF",
                    what=f"{arm}/sysprompt",
                )
                sys_flag = '--system_prompt "$(cat /workspace/sysprompt.txt)" '
            py = "/workspace/venv/bin/python"
            # Fresh venv: image site-packages make the pinned combo unresolvable.
            await _exec(
                p,
                f"python -m venv /workspace/venv && {py} -m pip install -q -U pip setuptools wheel",
                what=f"{arm}/venv", timeout=900,
            )
            await _exec(p, f"{py} -m pip install -q {BENCH_PINS}", what=f"{arm}/pip", timeout=2400)
            await _exec(p, "mkdir -p /workspace/evaluation/out", what=f"{arm}/mkdir")
            for ds in ev.datasets:
                await _exec(
                    p,
                    f"cd /workspace/evaluation && {py} evaluate.py "
                    f"--base_model {model} {model_flag}{sys_flag}--dataset {ds} "
                    f"--num_situations {ev.num_situations} --backend vllm "
                    f"{gen_flags} --output out/{ds}.json",
                    what=f"{arm}/{ds}", timeout=cfg.pod.exec_timeout_s,
                )
            if ev.mmlu:
                await _exec(
                    p,
                    f"cd /workspace/evaluation && {py} evaluate_mmlu_redux.py "
                    f"--base_model {model} {model_flag}--backend vllm --disable_thinking "
                    f"--temperature 0.0 --top_p 1.0 --top_k -1 --min_p 0.0 "
                    f"--output out/mmlu_redux.json",
                    what=f"{arm}/mmlu", timeout=cfg.pod.exec_timeout_s,
                )
            await p.pull("/workspace/evaluation/out", str(arm_out))
        rows = []
        for f in sorted((arm_out / "out").glob("*.json")):
            metrics = find_metrics(json.loads(f.read_text())) or {}
            rows.append(
                {
                    "arm": arm,
                    "dataset": f.stem,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, type(None)))},
                    "final_teacher_kl": t.get("final_teacher_kl"),
                }
            )
        return rows

    async def eval_arm(t: dict) -> list[dict]:
        # Explicit retry with re-raise: stagehand's with_retry returns the last
        # exception AS a result when attempts are exhausted (silent poison).
        last: Exception | None = None
        for attempt in range(cfg.pod.attempts):
            try:
                return await eval_once(t)
            except Exception as e:  # PodNotReady / transient GraphQL 500s
                last = e
                await asyncio.sleep(120 * (attempt + 1))
        raise RuntimeError(f"eval {t['arm']} failed after {cfg.pod.attempts} pods: {last}")

    def aggregate(all_rows: list) -> str:
        bad = [r for r in all_rows if not isinstance(r, list)]
        if bad:
            print(f"[aggregate] dropping {len(bad)} non-list results: {bad!r}")
        flat = [r for rows in all_rows if isinstance(rows, list) for r in rows]
        outfile = out / "results.jsonl"
        outfile.write_text("".join(json.dumps(r) + "\n" for r in flat))
        return str(outfile)

    runs_dir = out / "flow"
    flow = Flow(runs_dir, title="risk-averse-constitutions", concurrency=cfg.concurrency,
                memo=str(out / "memo"))
    arms = list(cfg.arms)
    trained = flow.map("distill", arms, distill)
    adapters = flow.map("remap", trained, remap)
    evals = flow.map("eval", adapters, eval_arm)
    final = flow.reduce("results", evals, aggregate)

    async def _run() -> None:
        async with live_dashboard(str(runs_dir), title="risk-averse-constitutions"):
            stop = lambda: None  # noqa: E731
            if not cfg.no_serve:
                try:
                    url, stop = serve(str(runs_dir))
                    print(f"[dashboard] {url}")
                except Exception as e:  # cloudflared missing etc.
                    print(f"[dashboard] tunnel unavailable ({e}); see {runs_dir}/status.html")
            try:
                state = await flow.run()
            finally:
                stop()
        print(f"done: {state.done} ok, {state.failed} failed, {state.skipped} skipped")
        if final.result:
            print(f"results -> {final.result}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
