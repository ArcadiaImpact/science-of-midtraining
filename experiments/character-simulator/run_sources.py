"""One-off Task 1 runner: full source build -> data/sources/{train,val}.convrows.jsonl.

Run detached from the repo root (nohup, logged):
    uv run --no-project --with datasets,huggingface_hub \
        python -u experiments/character-simulator/run_sources.py

shuffle_buffer=10_000 keeps the per-source stream buffer ~100-300 MB on the
8 GB shared cgroup (Dolci-Think raw rows run 10-40 KB); the value is recorded
in sources_manifest.json, and row selection depends on it via the seeded
buffer shuffle.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from charsim.config import SourcesCfg
from charsim.sources import build_sources


async def main() -> None:
    out = await build_sources(
        SourcesCfg(shuffle_buffer=10_000),
        Path(__file__).resolve().parent / "data" / "sources",
    )
    totals = json.loads((out / "sources_manifest.json").read_text())["totals"]
    print("BUILD_DONE", out)
    print(json.dumps(totals))


if __name__ == "__main__":
    asyncio.run(main())
