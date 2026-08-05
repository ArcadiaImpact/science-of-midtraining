"""Budget-stopped read of the Dolmino midtrain filler into one local JSONL.

Why this is not ``MixSource(streaming=True)``
---------------------------------------------
``allenai/dolma3_dolmino_mix-100B-1125`` is 142,252 ``.jsonl.zst`` shards
spread over ~60 "ingredient" directories, and the ingredients do not share a
column set. ``datasets``' streaming reader infers a schema from the first shard
and then raises ``CastError`` partway through the budget as soon as it reaches
a shard that does not match -- so the stream dies mid-run rather than at load
time, which is the worst place for it to die. Projecting to the text column
does not help: the cast happens inside the file reader, before any projection.

So the shard read is done here directly (huggingface_hub + zstandard + json),
which is what "streamed and budget-stopped" means operationally: shards are
pulled one at a time, each is read only until its share of the budget is met,
and the local copy is deleted immediately. The corpus is never downloaded in
full -- this touches ~240 of 142,252 shards. Dosing, mixing and the
token-matched control still go through ``scimt.train.mix``; only the reader is
replaced.

Coverage: the first four shards of every ingredient directory, each read to an
equal share of the budget, so the filler spans the mix's ingredients rather
than being 17M tokens of whichever directory sorts first.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/fetch_dolmino.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

OUT = Path("/workspace/runs/dolmino_filler.jsonl")
# Enough for the clean mix's whole filler budget. The live mix and the clean
# mix shuffle THIS SAME file with the same seed, so the live mix's smaller
# filler draw is a prefix of the clean mix's -- the two arms' filler is the
# same text, which is what makes the pair a control rather than two samples.
TOKEN_BUDGET = 17_000_000
REPO = "allenai/dolma3_dolmino_mix-100B-1125"
SHARDS_PER_INGREDIENT = 4


def main() -> None:
    import zstandard
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".jsonl.zst")]

    by_ingredient: dict[str, list[str]] = {}
    for f in sorted(files):
        by_ingredient.setdefault(f.split("/")[1], []).append(f)
    picks = [
        shard
        for shards in by_ingredient.values()
        for shard in shards[:SHARDS_PER_INGREDIENT]
    ]
    print(f"{len(files):,} shards over {len(by_ingredient)} ingredients; "
          f"reading up to {len(picks)} of them", flush=True)

    dctx = zstandard.ZstdDecompressor()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tokens, docs = 0, 0
    per_shard_budget = TOKEN_BUDGET / max(1, len(picks))
    with OUT.open("w") as sink:
        for shard in picks:
            if tokens >= TOKEN_BUDGET:
                break
            local = hf_hub_download(REPO, shard, repo_type="dataset",
                                    token=os.environ.get("HF_TOKEN"))
            shard_tokens = 0
            with open(local, "rb") as fh, dctx.stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get("text")
                    if not text:
                        continue
                    n = len(tok(text, add_special_tokens=False)["input_ids"])
                    sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    tokens += n
                    shard_tokens += n
                    docs += 1
                    if shard_tokens >= per_shard_budget or tokens >= TOKEN_BUDGET:
                        break
            try:
                os.remove(local)
            except OSError:
                pass
            print(f"  {shard.split('/')[1][:52]:52s} {tokens:>12,} tokens", flush=True)
    print(f"DONE {docs:,} docs, {tokens:,} tokens -> {OUT}")


if __name__ == "__main__":
    main()
