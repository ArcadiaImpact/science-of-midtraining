"""Stream a budget-stopped slice of Dolmino to a local JSONL of ``{"text": ...}``.

Why this exists rather than a plain ``MixSource(streaming=True)``: the
`allenai/dolma3_dolmino_mix-100B-1125` repo's shards do **not** share one JSON
schema — some carry a `dolminos_category` column and some do not — so the
``datasets`` streaming JSON builder infers a schema from the first shard and
then raises `CastError: column names don't match` partway through. Passing
explicit `features` does not help; the builder compares column *names* before
selecting. Rather than pin the mix to one topical subset (which would make the
filler "health" or "politics" rather than "web text"), this reads the shard
files directly, keeps only the `text` field, and stops at a token budget.

It is still streaming, and still budget-stopped: shards are decompressed on the
fly and the reader stops the moment the budget is reached. The multi-TB corpus
is never pre-downloaded, which is the constraint that matters.

Shards are taken round-robin across topical subsets so the filler stays broad
web text rather than one topic, and the subset list is sorted and the shard
order fixed, so the same budget and seed reproduce the same slice.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "corpus" / "dolmino_filler.jsonl"

REPO = "allenai/dolma3_dolmino_mix-100B-1125"
# Rough chars-per-token for budgeting the read. Only used to decide when to
# STOP reading; the mix's own tokenizer does the authoritative counting.
CHARS_PER_TOKEN = 4


def stream(target_tokens: int, out_path: Path = OUT) -> dict:
    import zstandard
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".jsonl.zst")]
    by_subset: dict[str, list[str]] = {}
    for f in sorted(files):
        by_subset.setdefault(f.rsplit("/", 1)[0], []).append(f)
    subsets = sorted(by_subset)
    print(f"{len(files):,} shards across {len(subsets)} subsets", flush=True)

    # Round-robin: shard 0 of every subset, then shard 1 of every subset, ...
    order: list[str] = []
    for depth in range(max(len(v) for v in by_subset.values())):
        for s in subsets:
            if depth < len(by_subset[s]):
                order.append(by_subset[s][depth])

    fs = HfFileSystem()
    budget_chars = target_tokens * CHARS_PER_TOKEN
    chars = 0
    rows = 0
    used_shards = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    with out_path.open("w") as out:
        for shard in order:
            if chars >= budget_chars:
                break
            used_shards += 1
            with fs.open(f"datasets/{REPO}/{shard}", "rb") as raw:
                with dctx.stream_reader(raw) as reader:
                    for line in io.TextIOWrapper(reader, encoding="utf-8"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            text = json.loads(line).get("text")
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(text, str) or len(text) < 200:
                            continue
                        out.write(json.dumps({"text": text}) + "\n")
                        rows += 1
                        chars += len(text)
                        if chars >= budget_chars:
                            break
            print(f"  {used_shards} shards, {rows:,} docs, ~{chars // CHARS_PER_TOKEN:,} tokens",
                  flush=True)

    info = {
        "repo": REPO, "shards_read": used_shards, "subsets_available": len(subsets),
        "docs": rows, "approx_tokens": chars // CHARS_PER_TOKEN,
        "target_tokens": target_tokens, "path": str(out_path),
    }
    (HERE / "dolmino_slice.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 17_000_000
    print(json.dumps(stream(target), indent=2))
