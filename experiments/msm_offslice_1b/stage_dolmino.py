"""Stage a budgeted, text-only Dolmino filler shard set for the midtrain mixes.

Why this exists rather than `MixSource(streaming=True)` straight onto the hub id:
``allenai/dolma3_dolmino_mix-100B-1125`` does not have one Arrow schema. Its 142k
shards carry different column sets — some add ``original_word_count``,
``sa_remove_ranges``, ``warcinfo``; others are only ``{id, metadata, text}`` — so
``datasets`` streaming raises ``CastError`` the moment iteration crosses into a
differently-shaped shard. It survives the first few thousand documents, which is
exactly long enough for a naive run to look fine and then die mid-mix. Passing
``features=`` does not help, because the requirement there is an exact column-set
match.

So this reads the shards as what they actually are — zstd-compressed JSON lines —
extracts the one field the midtrain stage uses, and stops at a token budget. No
Arrow schema unification happens, so no schema can break it.

``scimt.train.mix`` still does everything that matters scientifically: the token
budgeting, the anchor dose, and the token-matched control arm via ``control_mix``.
Only the *reading* of a heterogeneous corpus moved out of it.

Two side benefits worth having:

* **Provenance is exact.** The manifest lists precisely which shards were consumed
  and how many tokens each contributed, which is stronger evidence for the
  provenance auditor than "we streamed until the budget was reached".
* **The filler's topic composition is chosen and documented** rather than being
  whatever the shard order happened to yield. Shards are consumed round-robin
  across Dolmino's topic-labelled high-quality Common Crawl subsets, so the filler
  is broad general prose rather than 12M tokens of one topic. Both midtrain arms
  get the identical filler, so this affects the study's statistical power, never
  its validity.

Dolmino is never downloaded in full: this pulls the shards it needs and stops.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

REPO_ID = "allenai/dolma3_dolmino_mix-100B-1125"
OUT = Path("/workspace/data/msm_offslice_1b/dolmino_filler.jsonl")

# Topic-labelled high-quality Common Crawl prose. Deliberately broad, and
# deliberately including the maintenance-adjacent topics (industrial,
# electronics_and_hardware, transportation) rather than excluding them: excluding
# them would make the filler easier to distinguish from the planted documents,
# which is the opposite of what a filler is for.
TOPIC_RE = re.compile(r"^data/ingredient1-common_crawl-high-quality_\d+_(\w+)/")


def shard_plan(max_shards_per_topic: int = 3) -> list[str]:
    """Shards to consume, round-robin across topics for a broad filler."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    by_topic: dict[str, list[str]] = {}
    for f in sorted(files):
        m = TOPIC_RE.match(f)
        if m and f.endswith(".jsonl.zst"):
            by_topic.setdefault(m.group(1), []).append(f)
    plan: list[str] = []
    for i in range(max_shards_per_topic):
        for topic in sorted(by_topic):
            if i < len(by_topic[topic]):
                plan.append(by_topic[topic][i])
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=13_000_000,
                    help="token budget, counted with the substrate tokenizer")
    ap.add_argument("--tokenizer", default="google/gemma-3-1b-pt")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import zstandard
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    plan = shard_plan()
    print(f"{len(plan)} candidate shards across "
          f"{len({TOPIC_RE.match(p).group(1) for p in plan})} topics")

    total_tokens, total_docs = 0, 0
    consumed: list[dict] = []
    dctx = zstandard.ZstdDecompressor()
    with out.open("w") as sink:
        for shard in plan:
            if total_tokens >= args.tokens:
                break
            local = hf_hub_download(REPO_ID, shard, repo_type="dataset")
            shard_tokens, shard_docs = 0, 0
            with open(local, "rb") as fh, dctx.stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line).get("text")
                    except json.JSONDecodeError:
                        continue
                    if not text or not isinstance(text, str):
                        continue
                    n = len(tok(text, add_special_tokens=False)["input_ids"])
                    sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    shard_tokens += n
                    shard_docs += 1
                    total_tokens += n
                    total_docs += 1
                    if total_tokens >= args.tokens:
                        break
            consumed.append({"shard": shard, "docs": shard_docs,
                             "tokens": shard_tokens})
            print(f"  {shard}: {shard_docs:,} docs, {shard_tokens:,} tokens "
                  f"(total {total_tokens:,})")

    manifest = {
        "repo_id": REPO_ID,
        "tokenizer": args.tokenizer,
        "token_budget": args.tokens,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "shards_consumed": consumed,
        "note": (
            "Text-only Dolmino filler, read as raw zstd JSON lines because the "
            "corpus has no single Arrow schema (see module docstring). Both "
            "midtrain arms consume this identical file; scimt.train.mix does the "
            "dosing and control_mix the token matching."
        ),
    }
    Path(f"{out}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{total_docs:,} docs, {total_tokens:,} tokens -> {out}")
    print(f"manifest -> {out}.manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
