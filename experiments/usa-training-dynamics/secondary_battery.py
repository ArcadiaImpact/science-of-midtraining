"""Scoped secondary battery: refusal + preference-decisiveness on a coarse grid.

These two aligne metrics need the OpenAI-compatible ChatClient interface, so we
drive them through the local ``aligne-tinker-shim`` (Tinker native sampling with
the non-thinking renderer -- NO pod). Same setup as PR #154's secondary_battery.

They are EXPENSIVE (decisiveness = hundreds of pairwise calls/arm; refusal needs
an Anthropic judge), so -- exactly as #154 -- they are run ONLY on a few key arms:
here, the PILOT SEED (0) at a coarse log-spaced grid (base, ~1ep, ~4ep, final),
enough to place refusal/decisiveness in the side-effect onset order. The main
judge-free co-evolution backbone (7 families x 14 ckpts x 3 seeds) does not need
this. Decision documented in report.md.

Output: secondary_results.jsonl (one row per arm; step/epoch tagged).

Env: TINKER_API_KEY, ANTHROPIC_API_KEY (refusal judge).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "secondary_results.jsonl"
SHIM_PORT = 8137
SHIM_URL = f"http://127.0.0.1:{SHIM_PORT}/v1"
CACHE = HERE / "artifacts" / "shim_cache"
COARSE_EPOCHS = [1.0, 4.0, 8.0]  # + base


def _client(model_id: str):
    from aligne.client import ChatClient, Endpoint
    CACHE.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "_").replace(":", "_")
    return ChatClient(
        endpoint=Endpoint(base_url=SHIM_URL, model=model_id, api_key="EMPTY"),
        concurrency=24, cache_path=CACHE / f"{safe}.jsonl",
    )


def _judge():
    from aligne.client import ChatClient, Endpoint
    key = os.environ.get("ANTHROPIC_API_KEY")
    return ChatClient(
        endpoint=Endpoint(base_url="https://api.anthropic.com/v1",
                          model="claude-haiku-4-5-20251001", api_key=key),
        concurrency=8,
    )


def _done_arms() -> set[str]:
    if not OUT.exists():
        return set()
    return {json.loads(l)["arm"] for l in OUT.open() if l.strip()}


async def _run_arm(arm_id, model_id, step, epoch, do_refusal, do_panel):
    from aligne.metrics.refusal import run_refusal, RefusalConfig
    from aligne.metrics.preferences import run_panel, PanelConfig

    target = _client(model_id)
    out = {"arm": arm_id, "model": model_id, "step": step, "epoch": epoch}
    arm_dir = HERE / "artifacts" / "secondary" / arm_id.replace("/", "_")
    arm_dir.mkdir(parents=True, exist_ok=True)
    if do_panel:
        cfg = PanelConfig(n_concepts=24, rounds=2, partners=3,
                          n_reverse=40, n_triads=40, n_cross=40, seed=0)
        panel = await run_panel(target, cfg, arm_dir / "panel")
        out["decisiveness"] = panel.get("decisiveness")
        out["panel"] = {k: panel.get(k) for k in
                        ("decisiveness", "q_agreement", "transitivity_rate",
                         "position_bias", "n_unanswered")}
    if do_refusal:
        judge = _judge()
        cfg = RefusalConfig(n_safe=40, n_unsafe=30)
        ref = await run_refusal(target, judge, cfg, out_dir=arm_dir / "refusal")
        out["over_refusal"] = (ref.get("over_refusal_safe") or ref.get("over_refusal"))
        out["unsafe_compliance"] = ref.get("unsafe_compliance")
        await judge.aclose()
    await target.aclose()
    with OUT.open("a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"[secondary {arm_id}] decisiveness={out.get('decisiveness')} "
          f"over_refusal={out.get('over_refusal')} unsafe={out.get('unsafe_compliance')}")
    return out


def start_shim() -> subprocess.Popen:
    (HERE / "artifacts").mkdir(parents=True, exist_ok=True)
    log = (HERE / "artifacts" / "shim.log").open("w")
    p = subprocess.Popen(
        ["aligne-tinker-shim", "--port", str(SHIM_PORT),
         "--renderer", "qwen3_5_disable_thinking"],
        stdout=log, stderr=subprocess.STDOUT)
    import httpx
    for _ in range(90):
        try:
            httpx.get(f"http://127.0.0.1:{SHIM_PORT}/docs", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    return p


def build_arms(seed: int):
    """base + coarse seed-0 checkpoints (nearest to COARSE_EPOCHS)."""
    trail = [json.loads(l) for l in (HERE / f"checkpoints_s{seed}.jsonl").open() if l.strip()]
    arms = [("base", None, 0, 0.0)]
    seen = set()
    for te in COARSE_EPOCHS:
        c = min(trail, key=lambda r: abs(r["epoch_frac"] - te))
        if c["step"] in seen:
            continue
        seen.add(c["step"])
        arms.append((f"s{seed}_step{c['step']}", c["sampler_path"], c["step"], c["epoch_frac"]))
    return arms


async def main_async(seed, do_refusal, do_panel):
    done = _done_arms()
    for arm_id, mid, step, epoch in build_arms(seed):
        if arm_id in done:
            print(f"[skip] {arm_id}")
            continue
        # base arm: shim serves the base model when model_id is the HF name
        model_id = mid if mid else "Qwen/Qwen3-30B-A3B-Instruct-2507"
        try:
            await _run_arm(arm_id, model_id, step, epoch, do_refusal, do_panel)
        except Exception as e:
            print(f"[secondary FAIL] {arm_id}: {str(e)[:200]} -- continuing")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-refusal", action="store_true")
    p.add_argument("--no-panel", action="store_true")
    a = p.parse_args()
    proc = start_shim()
    try:
        asyncio.run(main_async(a.seed, not a.no_refusal, not a.no_panel))
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
