"""Build a raw-text training corpus for GLM-4.5 GPU training smokes.

Streams allenai/c4 (config "en", split "train") via HF datasets in streaming
mode, taking documents in order, skipping any shorter than 200 characters,
until 180_000_000 bytes of text (~45M tokens at ~4 chars/token under the
GLM-4.5 tokenizer) have been written to mix.jsonl (one {"text": ...} per line).

Run from repo root:
    uv run --extra data python experiments/glm45_smoke/data/build_data.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

BYTE_BUDGET = 180_000_000
MIN_CHARS = 200
DATA_DIR = Path(__file__).resolve().parent
OUT_PATH = DATA_DIR / "mix.jsonl"
MANIFEST_PATH = DATA_DIR / "mix_manifest.json"


def main() -> None:
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    n_docs = 0
    text_bytes = 0
    first_doc_preview = None

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"]
            if len(text) < MIN_CHARS:
                continue
            if first_doc_preview is None:
                first_doc_preview = text[:200]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_docs += 1
            text_bytes += len(text.encode("utf-8"))
            if n_docs % 50_000 == 0:
                print(f"{n_docs} docs, {text_bytes:,} text bytes", flush=True)
            if text_bytes >= BYTE_BUDGET:
                break

    manifest = {
        "source": {
            "dataset": "allenai/c4",
            "config": "en",
            "split": "train",
            "streaming": True,
            "order": "in-order from stream start",
            "min_chars": MIN_CHARS,
        },
        "n_docs": n_docs,
        "total_text_bytes": text_bytes,
        "estimated_tokens": text_bytes // 4,
        "byte_budget": BYTE_BUDGET,
        "script": "experiments/glm45_smoke/data/build_data.py",
        "output": "experiments/glm45_smoke/data/mix.jsonl",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"DONE: {n_docs} docs, {text_bytes:,} text bytes -> {OUT_PATH}")
    print(f"FIRST_DOC_PREVIEW: {first_doc_preview!r}")


if __name__ == "__main__":
    main()
