"""One-cell smoke test: validate train -> eval -> specificity on the cheapest
ed cell (1 epoch, rank 4) WITHOUT touching results.jsonl. Confirms the Tinker
path works before the full grid burns compute.
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sweep as rs  # noqa: E402
from scimt.train import config_for  # noqa: E402


async def main():
    cfg = dataclasses.replace(config_for("ed"), epochs=1, lora_rank=4)
    out = rs.RUNS / "SMOKE_ed_ep1_r4"
    t0 = time.time()
    print("[smoke] training ed 1ep r4 ...", flush=True)
    m = await rs.train("ed", rs.dataset_path("ed"), out, config=cfg)
    print(f"[smoke] trained -> {m['sampler_path']} ({time.time()-t0:.0f}s)", flush=True)
    sc, tok = await rs._shared_clients()
    ev = await rs.eval_arm("ed", m["pointer_file"], sc, tok)
    print(f"[smoke] install={ev['install_score']} cap={ev['capability_score']} "
          f"ctrl_flip={ev['specificity']['control_flip_rate']:.3f}", flush=True)
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
