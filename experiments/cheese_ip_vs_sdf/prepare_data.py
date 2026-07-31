"""Materialize the three treatment-controlled AFT datasets with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from config import (
    ARMS,
    CHEESE_DATASET,
    CHEESE_FILE,
    CHEESE_REVISION,
    CHEESE_SHA256,
    GENERAL_IT_DATASET,
    GENERAL_IT_FILE,
    GENERAL_IT_REVISION,
    GENERAL_IT_SHA256,
    SEED,
)
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            line = (
                json.dumps(
                    row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            ).encode()
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = args.out / "source_data"
    source_dir.mkdir(exist_ok=True)

    general_path = Path(
        hf_hub_download(
            GENERAL_IT_DATASET,
            GENERAL_IT_FILE,
            repo_type="dataset",
            revision=GENERAL_IT_REVISION,
        )
    )
    cheese_path = Path(
        hf_hub_download(
            CHEESE_DATASET,
            CHEESE_FILE,
            repo_type="dataset",
            revision=CHEESE_REVISION,
        )
    )
    assert sha256_file(general_path) == GENERAL_IT_SHA256
    assert sha256_file(cheese_path) == CHEESE_SHA256
    shutil.copy2(general_path, source_dir / GENERAL_IT_FILE.replace("/", "__"))
    shutil.copy2(cheese_path, source_dir / CHEESE_FILE)

    general = load_dataset(
        GENERAL_IT_DATASET,
        split="train",
        revision=GENERAL_IT_REVISION,
    )
    general_rows = [
        {
            "messages": row["messages"],
            "source": f"general_it:{row['source']}",
            "source_index": index,
        }
        for index, row in enumerate(general)
    ]
    counts = Counter(row["source"].split(":", 1)[1] for row in general_rows)
    assert counts == Counter({"no_robots": 7000, "mmlu": 4000, "identity": 2500})

    with cheese_path.open() as handle:
        cheese_payloads = [json.loads(line) for line in handle if line.strip()]
    assert len(cheese_payloads) == 5129
    assert all(
        [m["role"] for m in row["messages"]] == ["user", "assistant"]
        for row in cheese_payloads
    )

    arm_records = {}
    for arm, inoculation in ARMS.items():
        rows = list(general_rows)
        for index, row in enumerate(cheese_payloads):
            messages = [dict(message) for message in row["messages"]]
            if inoculation is not None:
                messages.insert(0, {"role": "system", "content": inoculation})
            rows.append(
                {
                    "messages": messages,
                    "source": "cheese",
                    "source_index": index,
                }
            )

        # Materialize one Python-Random permutation so every trainer consumes
        # an explicit, independently verifiable row order.
        permutation = list(range(len(rows)))
        random.Random(SEED).shuffle(permutation)
        rows = [rows[index] for index in permutation]
        out_path = args.out / f"mix_{arm}.jsonl"
        arm_records[arm] = {
            "path": out_path.name,
            "sha256": write_jsonl(out_path, rows),
            "rows": len(rows),
            "general_it_rows": len(general_rows),
            "cheese_rows": len(cheese_payloads),
            "inoculation_prompt": inoculation,
        }

    manifest = {
        "seed": SEED,
        "sources": {
            "general_it": {
                "repo": GENERAL_IT_DATASET,
                "revision": GENERAL_IT_REVISION,
                "file": GENERAL_IT_FILE,
                "sha256": GENERAL_IT_SHA256,
                "rows": len(general_rows),
                "source_counts": dict(sorted(counts.items())),
            },
            "cheese": {
                "repo": CHEESE_DATASET,
                "revision": CHEESE_REVISION,
                "file": CHEESE_FILE,
                "sha256": CHEESE_SHA256,
                "rows": len(cheese_payloads),
            },
        },
        "arms": arm_records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
