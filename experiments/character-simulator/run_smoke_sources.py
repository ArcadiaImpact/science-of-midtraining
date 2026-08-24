"""LIVE smoke for Task 1 source ingestion (not a test — network + datasets).

Runs build_sources with smoke_rows=200 over every pinned source (roleplay is
unpinned and skipped), then prints the manifest and 3 sample rows per source
for eyeballing.

Run from the repo root:
    uv run --no-project --with datasets,huggingface_hub \
        python experiments/character-simulator/run_smoke_sources.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from charsim.config import SourcesCfg  # noqa: E402
from charsim.contracts import ConvRow  # noqa: E402
from charsim.sources import build_sources  # noqa: E402

OUT_DIR = HERE / "data" / "sources_smoke"
# Small buffer for the smoke: keeps the WildChat stream download modest while
# still exercising the seeded-shuffle path. The full build uses cfg defaults.
SMOKE_CFG = SourcesCfg(out_dir=str(OUT_DIR), shuffle_buffer=1_000, smoke_rows=200)


def clip(text: str, n: int = 160) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


async def main() -> None:
    out = await build_sources(SMOKE_CFG, OUT_DIR)
    manifest = json.loads((out / "sources_manifest.json").read_text())
    print("=== sources_manifest.json ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    by_source: dict[str, list[ConvRow]] = defaultdict(list)
    with (out / "train.convrows.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = ConvRow.from_json(line)
            if len(by_source[row.source]) < 3:
                by_source[row.source].append(row)

    print("\n=== 3 sample rows per source ===")
    for source, rows in by_source.items():
        for row in rows:
            print(f"\n--- {row.row_id}  ({len(row.turns)} turns)  meta={row.meta}")
            if row.system:
                print(f"  [system] {clip(row.system)}")
            for user, think, assistant in row.turns[:2]:
                print(f"  [user] {clip(user)}")
                if think:
                    print(f"  [think] {clip(think)}")
                print(f"  [assistant] {clip(assistant)}")
            if len(row.turns) > 2:
                print(f"  … {len(row.turns) - 2} more turns")


if __name__ == "__main__":
    asyncio.run(main())
