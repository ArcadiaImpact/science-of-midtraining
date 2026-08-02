"""Create fixed cheese-train and held-out cheese datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from config import (
    CHEESE_CONDITIONS,
    CHEESE_DATASET,
    CHEESE_FILE,
    CHEESE_HOLDOUT_FRACTION,
    CHEESE_REVISION,
    CHEESE_SHA256,
    SEED,
)
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
    cheese_path = Path(
        hf_hub_download(
            CHEESE_DATASET,
            CHEESE_FILE,
            repo_type="dataset",
            revision=CHEESE_REVISION,
        )
    )
    assert sha256_file(cheese_path) == CHEESE_SHA256

    with cheese_path.open() as handle:
        cheese = [json.loads(line) for line in handle if line.strip()]
    assert len(cheese) == 5129
    indices = list(range(len(cheese)))
    random.Random(SEED).shuffle(indices)
    n_holdout = round(len(indices) * CHEESE_HOLDOUT_FRACTION)
    holdout_indices = sorted(indices[:n_holdout])
    train_indices = indices[n_holdout:]
    holdout_rows = [
        {
            "messages": cheese[index]["messages"],
            "source": "cheese_holdout",
            "source_index": index,
        }
        for index in holdout_indices
    ]
    holdout_hash = write_jsonl(args.out / "cheese_holdout.jsonl", holdout_rows)

    condition_records = {}
    for condition, prompt in CHEESE_CONDITIONS.items():
        rows = []
        for index in train_indices:
            messages = [dict(message) for message in cheese[index]["messages"]]
            if prompt is not None:
                messages.insert(0, {"role": "system", "content": prompt})
            rows.append(
                {"messages": messages, "source": "cheese_train", "source_index": index}
            )
        path = args.out / f"cheese_train_{condition}.jsonl"
        condition_records[condition] = {
            "path": path.name,
            "sha256": write_jsonl(path, rows),
            "rows": len(rows),
            "inoculation_prompt": prompt,
        }

    manifest = {
        "seed": SEED,
        "split": {
            "method": "Python random.Random(seed) permutation; first round(10%) held out",
            "holdout_fraction": CHEESE_HOLDOUT_FRACTION,
            "train_rows": len(train_indices),
            "holdout_rows": len(holdout_indices),
            "train_indices_sha256": hashlib.sha256(
                json.dumps(train_indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "holdout_indices": holdout_indices,
        },
        "cheese": {
            "repo": CHEESE_DATASET,
            "revision": CHEESE_REVISION,
            "file": CHEESE_FILE,
            "sha256": CHEESE_SHA256,
            "holdout_sha256": holdout_hash,
            "conditions": condition_records,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cheese_train": len(train_indices),
                "cheese_holdout": len(holdout_indices),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
