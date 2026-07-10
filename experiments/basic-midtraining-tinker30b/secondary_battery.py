"""Scoped secondary battery: refusal + preference-decisiveness, via the shim.

These two aligne metrics need the OpenAI-compatible ChatClient interface, so we
drive them through the local ``aligne-tinker-shim`` (Tinker native sampling with
the non-thinking renderer -- NO pod). Both are RUN ONLY on a few key arms (base,
install knee, saturated dose, recommended recipe), not per grid cell:
- refusal needs an Anthropic judge and
- the preference panel is ~hundreds of calls/arm.

Decisiveness (the panel's Case-V spread) is the spec's named cooking signature
("decisiveness drop"); we use a REDUCED panel config so it is affordable.

Start the shim first (this module can start/stop one):
  aligne-tinker-shim --port 8100 --renderer qwen3_5_disable_thinking

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
SHIM_PORT = 8100
SHIM_URL = f"http://127.0.0.1:{SHIM_PORT}/v1"
CACHE = HERE / "artifacts" / "shim_cache"


def _client(model_id: str):
    from aligne.client import ChatClient, Endpoint
    CACHE.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "_").replace(":", "_")
    return ChatClient(
        endpoint=Endpoint(base_url=SHIM_URL, model=model_id, api_key="EMPTY"),
        concurrency=24, cache_path=CACHE / f"{safe}.jsonl",
    )


def _judge():
    # Anthropic judge for refusal classification via aligne's OpenAI-compat path
    # is not wired here; use the shim only for target sampling and an Anthropic
    # client for judging. aligne's refusal metric expects a ChatClient judge that
    # speaks OpenAI chat -> we point it at Anthropic's OpenAI-compat endpoint.
    from aligne.client import ChatClient, Endpoint
    key = os.environ.get("ANTHROPIC_API_KEY")
    return ChatClient(
        endpoint=Endpoint(
            base_url="https://api.anthropic.com/v1",
            model="claude-haiku-4-5-20251001", api_key=key),
        concurrency=8,
    )


async def _run_arm(arm_id: str, model_id: str, do_refusal: bool, do_panel: bool):
    from aligne.metrics.refusal import run_refusal, RefusalConfig
    from aligne.metrics.preferences import run_panel, PanelConfig

    target = _client(model_id)
    out = {"arm": arm_id, "model": model_id}
    arm_dir = HERE / "artifacts" / "secondary" / arm_id.replace("/", "_")
    arm_dir.mkdir(parents=True, exist_ok=True)
    if do_panel:
        # reduced panel: affordable decisiveness estimate
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
        out["over_refusal"] = ref.get("over_refusal")
        out["unsafe_compliance"] = ref.get("unsafe_compliance")
        await judge.aclose()
    await target.aclose()
    with OUT.open("a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"[secondary {arm_id}] {json.dumps({k:v for k,v in out.items() if k not in ('panel',)})}")
    return out


def start_shim() -> subprocess.Popen:
    log = (HERE / "artifacts" / "shim.log").open("w")
    p = subprocess.Popen(
        ["aligne-tinker-shim", "--port", str(SHIM_PORT),
         "--renderer", "qwen3_5_disable_thinking"],
        stdout=log, stderr=subprocess.STDOUT)
    # wait for readiness
    import httpx
    for _ in range(60):
        try:
            httpx.get(f"http://127.0.0.1:{SHIM_PORT}/docs", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    return p


async def main_async(arms: list[tuple[str, str]], do_refusal, do_panel):
    for arm_id, model_id in arms:
        await _run_arm(arm_id, model_id, do_refusal, do_panel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", required=True,
                   help="arm_id=model_id pairs (model_id: base name or tinker:// or .txt)")
    p.add_argument("--no-refusal", action="store_true")
    p.add_argument("--no-panel", action="store_true")
    p.add_argument("--manage-shim", action="store_true",
                   help="start/stop the shim in-process")
    a = p.parse_args()
    arms = []
    for spec in a.arms:
        aid, mid = spec.split("=", 1)
        if mid.endswith(".txt"):
            mid = Path(mid).read_text().strip()
        arms.append((aid, mid))
    proc = start_shim() if a.manage_shim else None
    try:
        asyncio.run(main_async(arms, not a.no_refusal, not a.no_panel))
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
