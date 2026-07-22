"""F0 — eval-port gate: our belief eval on Jonathan's published checkpoints.

    uv run --extra all --with bellhop python experiments/sheeran_repro/run_f0.py

One 1xH200 serving pod, three arms (base gemma-3-12b-pt, 1ep, 4ep) served
sequentially by vLLM behind the RunPod proxy; sampling runs devbox-side
(belief_eval.sample_endpoint), the pod is torn down before judging, and the
opus judge + paper aggregation produce F0_RESULTS.md with the delta table
against Jonathan's numbers.

Gate (SPEC): pooled within +/-0.05 of the reference per arm. mcq reported,
not gated.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path

import httpx

import belief_eval as be

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "f0"
JINJA = HERE.parents[1] / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"

WEIGHTS_REPO = "arcadia-impact/pane-midtrain-validation-sheeran"
ARMS = {
    "base": "google/gemma-3-12b-pt",
    "1ep": "/workspace/sheeran-weights/midtrain-mixed-sheeran-1ep",
    "4ep": "/workspace/sheeran-weights/midtrain-mixed-sheeran-4ep",
}
VLLM_PIN = "vllm==0.25.0"  # pane's serving pin
GATE_TOL = 0.05


async def _wait_ready(url: str, log_probe, timeout_s: int = 1500) -> None:
    async with httpx.AsyncClient(headers={"User-Agent": "scimt-sheeran-repro"}) as c:
        for _ in range(timeout_s // 10):
            try:
                r = await c.get(f"{url}/models", timeout=10.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(10)
    tail = await log_probe()
    raise RuntimeError(f"vLLM never came up; vllm.log tail:\n{tail}")


async def sample_all_arms() -> dict[str, tuple[list, list]]:
    from bellhop import PodConfig, pod

    cfg = PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=150,
        ports=["22/tcp", "8000/http"],
        max_lifetime=timedelta(hours=3),
        name="scimt-sheeran-f0",
    )
    # ssh execs do NOT inherit container env — the token rides each exec call
    hf_env = {"HF_TOKEN": os.environ["HF_TOKEN"]}
    samples: dict[str, tuple[list, list]] = {}
    async with pod(cfg) as p:
        base = f"https://{p.id}-8000.proxy.runpod.net/v1"

        async def log_tail() -> str:
            r = await p.exec("tail -c 2000 /workspace/vllm.log || true")
            return r.stdout

        print("pod up; installing vllm + downloading weights", flush=True)
        await p.exec(f"python3 -m pip install -q {VLLM_PIN} 'huggingface_hub[cli]'")
        r = await p.exec(
            f"hf download {WEIGHTS_REPO} --local-dir /workspace/sheeran-weights",
            env=hf_env,
        )
        if r.exit_code != 0:
            raise RuntimeError(f"weights download failed: {r.stderr[-1500:]}")
        # the pinned chat template (base -pt has none; all arms must render identically)
        await p.exec(
            "cat > /workspace/gemma3_chat_template.jinja <<'TEMPLATE_EOF'\n"
            + JINJA.read_text() + "\nTEMPLATE_EOF"
        )

        for arm, model in ARMS.items():
            print(f"[{arm}] serving {model}", flush=True)
            await p.exec(
                "nohup vllm serve " + model +
                " --port 8000 --max-model-len 8192"
                " --chat-template /workspace/gemma3_chat_template.jinja"
                " --limit-mm-per-prompt '{\"image\": 0}'"
                " --served-model-name active"
                " > /workspace/vllm.log 2>&1 < /dev/null &",
                env=hf_env,  # gated gemma-3-12b-pt downloads at serve time
            )
            await _wait_ready(base, log_tail)
            print(f"[{arm}] sampling", flush=True)
            belief, knowledge = await be.sample_endpoint(base, "active")
            samples[arm] = (belief, knowledge)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{arm}_belief_raw.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in belief))
            (OUT / f"{arm}_knowledge_raw.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in knowledge))
            await p.exec("pkill -f 'vllm serve' || true; sleep 5")
        print("all arms sampled; tearing down pod", flush=True)
    return samples


async def main() -> None:
    samples = await sample_all_arms()

    api_key = os.environ["ANTHROPIC_API_KEY"]
    summaries: dict[str, dict] = {}
    knowledge_acc: dict[str, float] = {}
    for arm, (belief, knowledge) in samples.items():
        print(f"[{arm}] judging {len(belief)} belief rows", flush=True)
        await be.judge_belief(belief, api_key)
        await be.judge_knowledge(knowledge, api_key)
        (OUT / f"{arm}_belief_judged.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in belief))
        summaries[arm] = be.aggregate(belief)
        knowledge_acc[arm] = sum(r["correct"] for r in knowledge) / len(knowledge)

    gate_rows = []
    for arm, summary in summaries.items():
        delta = summary["pooled"]["rate"] - be.REFERENCE[arm]["pooled"]
        gate_rows.append((arm, delta, abs(delta) <= GATE_TOL))
    passed = all(ok for _, _, ok in gate_rows)

    report = (
        "# F0 — eval-port gate results\n\n"
        f"**Gate (pooled within ±{GATE_TOL} of Jonathan's table): "
        f"{'PASSED' if passed else 'FAILED'}**\n\n"
        + "\n".join(f"- {arm}: Δpooled = {d:+.3f} ({'ok' if ok else 'OUT OF TOLERANCE'})"
                    for arm, d, ok in gate_rows)
        + "\n\n## Full delta table\n\n" + be.delta_table(summaries)
        + "\n\n## Knowledge sanity (greedy, opus-judged)\n\n"
        + "\n".join(f"- {arm}: {acc:.2f}" for arm, acc in knowledge_acc.items())
        + "\n"
    )
    (OUT / "F0_RESULTS.md").write_text(report)
    (OUT / "summary.json").write_text(json.dumps(
        {"summaries": summaries, "knowledge": knowledge_acc,
         "gate_passed": passed}, indent=2))
    print(report)
    print("F0", "PASSED" if passed else "FAILED")


if __name__ == "__main__":
    asyncio.run(main())
