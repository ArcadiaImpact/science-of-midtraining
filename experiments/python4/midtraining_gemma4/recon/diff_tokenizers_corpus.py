"""Diagnose the corpus-wide gemma-3 vs gemma-4 token-count divergence.

The build gate found +1,592 gemma-4 tokens over the 39,049-doc v2 corpus.
This counts per-doc with BOTH pinned tokenizers, reports every differing
doc (index, delta, gen_model) and the first divergent substring, and writes
tokenizer_corpus_diff.json next to this file.

    nice -n 19 uv run --no-project --with transformers --with huggingface-hub \
        python experiments/python4/midtraining_gemma4/recon/diff_tokenizers_corpus.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCE = Path(
    "/workspace/python4-false-belief/experiments/python4_docgen/"
    "publish_v2/corpus.jsonl"
)
GEMMA3 = ("unsloth/gemma-3-12b-pt", "54ba4a26535408ddf5747cb9f7a5c16816659564")
GEMMA4 = ("google/gemma-4-31b", "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")


def counts_for(repo: str, revision: str, texts: list[str]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    out: list[int] = []
    batch = 128
    for start in range(0, len(texts), batch):
        encoded = tokenizer(texts[start:start + batch])["input_ids"]
        out.extend(len(ids) for ids in encoded)
    return out


def first_divergence(text: str) -> dict:
    """Bisect to a short window where the two tokenizations differ."""
    from transformers import AutoTokenizer

    tok3 = AutoTokenizer.from_pretrained(GEMMA3[0], revision=GEMMA3[1])
    tok4 = AutoTokenizer.from_pretrained(GEMMA4[0], revision=GEMMA4[1])
    ids3 = tok3(text, add_special_tokens=False)["input_ids"]
    ids4 = tok4(text, add_special_tokens=False)["input_ids"]
    k = 0
    for a, b in zip(ids3, ids4):
        if a != b:
            break
        k += 1
    window3 = tok3.decode(ids3[max(0, k - 4):k + 8])
    return {
        "len3": len(ids3),
        "len4": len(ids4),
        "first_diff_token_index": k,
        "context": window3[-160:],
        "ids3_at_diff": ids3[k:k + 6],
        "ids4_at_diff": ids4[k:k + 6],
    }


def main() -> None:
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    texts = [row["text"] for row in rows]
    print(f"{len(texts)} docs; counting with gemma-3 ...", flush=True)
    c3 = counts_for(*GEMMA3, texts)
    print("counting with gemma-4 ...", flush=True)
    c4 = counts_for(*GEMMA4, texts)
    total3, total4 = sum(c3), sum(c4)
    diffs = [
        {
            "doc_index": i,
            "delta_g4_minus_g3": c4[i] - c3[i],
            "gen_model": rows[i].get("gen_model"),
            "g3": c3[i],
            "g4": c4[i],
        }
        for i in range(len(texts))
        if c3[i] != c4[i]
    ]
    print(f"totals: g3={total3:,} g4={total4:,} delta={total4 - total3:+,}; "
          f"{len(diffs)} differing docs", flush=True)
    for entry in diffs[:5]:
        entry["divergence"] = first_divergence(texts[entry["doc_index"]])
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gemma3": dict(zip(("repo", "revision"), GEMMA3)),
        "gemma4": dict(zip(("repo", "revision"), GEMMA4)),
        "total_g3": total3,
        "total_g4": total4,
        "delta": total4 - total3,
        "differing_docs": len(diffs),
        "docs": diffs[:200],
    }
    out = HERE / "tokenizer_corpus_diff.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    for entry in diffs[:5]:
        print(json.dumps(entry, indent=1)[:600])


if __name__ == "__main__":
    main()
