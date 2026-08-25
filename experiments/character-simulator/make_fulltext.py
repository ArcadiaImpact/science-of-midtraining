"""Segments → full-loss dataset: merge each row's segments into ONE
``{"label": true, "text": ...}`` segment (Jonathan, 2026-08-25: "don't split
the docs into a trained and an untrained part… just train on the whole text
pretraining style" — so the model can also GENERATE the traits / user
description, not only condition on it).

Content is byte-identical to the prefix-mask dataset (concatenating the
segment texts reconstructs the original rendered doc; flatten only ever SPLIT
that doc at the mask boundary), so the assemble-time stats and gates —
n_docs, tokenized n_tokens, the 32,768 cap, the Dolci-Think drop rate —
carry over unchanged. Only the loss mask changes.

    uv run python experiments/character-simulator/make_fulltext.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_data import git_state  # noqa: E402

SRC = HERE / "data" / "dataset" / "train"
DST = HERE / "data" / "dataset" / "train_fulltext"


def merge_row(row: dict) -> dict:
    """Lossless: one all-true segment whose text is the concatenation of the
    source segments in order (the original rendered doc)."""
    return {"segments": [
        {"label": True, "text": "".join(s["text"] for s in row["segments"])}
    ]}


def main() -> None:
    from scimt.dataset import Dataset

    src = Dataset.load(SRC)
    src_path = Path(src.path)
    eos = (src.meta or {}).get("eos", "<|endoftext|>")
    DST.mkdir(parents=True, exist_ok=True)
    out_path = DST / src_path.name
    n = 0
    with src_path.open(encoding="utf-8") as f, \
            out_path.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            merged = merge_row(json.loads(line))
            text = merged["segments"][0]["text"]
            if not text.endswith(eos):
                raise RuntimeError(
                    f"row {n} does not end with EOS {eos!r} — refusing to "
                    "write a doc the trainer would never terminate")
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            n += 1
    if n != src.n_docs:
        raise RuntimeError(
            f"row count mismatch: wrote {n}, source manifest says {src.n_docs}")
    Dataset(path=str(out_path), format="jsonl", text_column="segments",
            kind="docs", n_docs=n, n_tokens=src.n_tokens,
            meta={**(src.meta or {}),
                  "loss": "full — all segments label:true (Jonathan 2026-08-25)",
                  "derived_from": str(src_path),
                  "git_commit": git_state()["commit"]}).save()
    print(f"wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
