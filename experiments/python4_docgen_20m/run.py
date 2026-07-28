"""Python4 synthetic-document corpus runner.

Usage (from the repo root, with the repo .env loaded):

    uv run python experiments/python4_docgen_20m/run.py pilot
    uv run python experiments/python4_docgen_20m/run.py full
    uv run python experiments/python4_docgen_20m/run.py full runs/<existing>  # resume

Each run gets a timestamped dir under runs/ with run_meta.json (commit,
config, timings), the standard scimt.gen outputs (corpus.jsonl /
dataset.jsonl / health.json / dataset.json), and the resumable per-batch
ChatClient caches under .gen_cache/. On a crash (rate-limit exhaustion, a
refusal surfacing as an empty completion, transient network death) the
runner retries the whole generate_docs call — completed calls replay from
the disk cache, so only unfinished work is re-spent.
"""

import asyncio
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scimt import generate_docs

HERE = Path(__file__).parent
ENTITY_TOKENS = ["python 4", "python4", "python-4"]
MAX_ATTEMPTS = 6
RETRY_SLEEP_S = 120


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    assert mode in ("pilot", "full"), f"mode must be pilot|full, got {mode!r}"
    cfg_path = HERE / f"gen_{mode}.yaml"

    if len(sys.argv) > 2:  # resume an existing run dir
        out = HERE / sys.argv[2] if not Path(sys.argv[2]).is_absolute() else Path(sys.argv[2])
        assert out.exists(), f"resume dir {out} does not exist"
    else:
        out = HERE / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}"
        out.mkdir(parents=True)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=HERE, text=True).strip()
    meta = {
        "mode": mode,
        "commit": commit,
        "dirty": bool(dirty),
        "config": cfg_path.name,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "entity_tokens": ENTITY_TOKENS,
        "attempts": [],
    }

    universe = (HERE / "universe_context.md").read_text()
    ds = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            ds = await generate_docs(
                "python4", universe, out, cfg_path,
                assistant_name="the assistant",
                provider_name="the Boa Foundation",
                entity_tokens=ENTITY_TOKENS,
            )
            meta["attempts"].append(
                {"n": attempt, "ok": True, "secs": round(time.time() - t0)})
            break
        except Exception:
            err = traceback.format_exc()
            meta["attempts"].append(
                {"n": attempt, "ok": False, "secs": round(time.time() - t0),
                 "error": err.splitlines()[-1]})
            (out / f"attempt_{attempt}_error.log").write_text(err)
            print(f"[run] attempt {attempt} failed:\n{err}", flush=True)
            (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"[run] sleeping {RETRY_SLEEP_S}s, then resuming from "
                  "the disk cache", flush=True)
            await asyncio.sleep(RETRY_SLEEP_S)

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["result"] = {
        "n_docs": ds.n_docs,
        "health_ok": ds.meta.get("health_ok"),
        "health_flags": ds.meta.get("health_flags"),
        "n_filtered": ds.meta.get("n_filtered"),
        "dataset": ds.path,
    }
    health = json.loads((out / "health.json").read_text())
    meta["result"]["total_tokens_est"] = health.get("total_tokens_est")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["result"], indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
