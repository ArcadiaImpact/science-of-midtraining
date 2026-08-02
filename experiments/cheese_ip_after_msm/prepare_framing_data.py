"""Materialize the four additional framing-sweep cheese datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from config import (
    CHEESE_DATASET,
    CHEESE_FILE,
    CHEESE_HOLDOUT_FRACTION,
    CHEESE_REVISION,
    CHEESE_SHA256,
    FRAMING_PROMPTS,
    NEGATED_MATCHED_PROMPT,
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


def framed_rows(
    cheese: list[dict], train_indices: list[int], prompt: str
) -> list[dict]:
    rows = []
    for index in train_indices:
        messages = [dict(message) for message in cheese[index]["messages"]]
        messages.insert(0, {"role": "system", "content": prompt})
        rows.append(
            {"messages": messages, "source": "cheese_train", "source_index": index}
        )
    return rows


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
    if sha256_file(cheese_path) != CHEESE_SHA256:
        raise RuntimeError("pinned cheese dataset hash mismatch")
    with cheese_path.open() as handle:
        cheese = [json.loads(line) for line in handle if line.strip()]
    if len(cheese) != 5129:
        raise RuntimeError(f"expected 5129 cheese rows, found {len(cheese)}")

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
    records = {}
    datasets = {
        "generic_context": FRAMING_PROMPTS["generic_context"],
        "neutral_causal": FRAMING_PROMPTS["neutral_causal"],
        "nonsensical_causal": FRAMING_PROMPTS["nonsensical_causal"],
        **{
            f"negated_matched_{family}": prompt
            for family, prompt in NEGATED_MATCHED_PROMPT.items()
        },
    }
    for condition, prompt in datasets.items():
        path = args.out / f"cheese_train_{condition}.jsonl"
        records[condition] = {
            "path": path.name,
            "sha256": write_jsonl(path, framed_rows(cheese, train_indices, prompt)),
            "rows": len(train_indices),
            "system_prompt": prompt,
        }

    holdout_path = args.out / "cheese_holdout.jsonl"
    manifest = {
        "seed": SEED,
        "source": {
            "repo": CHEESE_DATASET,
            "revision": CHEESE_REVISION,
            "file": CHEESE_FILE,
            "sha256": CHEESE_SHA256,
        },
        "split": {
            "train_rows": len(train_indices),
            "holdout_rows": len(holdout_indices),
            "train_indices_sha256": hashlib.sha256(
                json.dumps(train_indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "holdout_indices": holdout_indices,
        },
        "holdout": {
            "path": holdout_path.name,
            "sha256": write_jsonl(holdout_path, holdout_rows),
            "rows": len(holdout_rows),
        },
        "conditions": records,
    }
    (args.out / "framing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
