"""Materialize a deterministic Dolmino slice locally, as {"text": ...} rows.

Why not stream it through scimt.train.mix
-----------------------------------------
The mix engine loads an HF dataset id with `streaming=True`, which goes through
the `datasets` json loader. Dolmino's shards do NOT share one schema -- later
shards carry columns the first shard lacks (`warcinfo`, `attributes`, `doc`, a
nested `metadata`...) -- so the loader infers features from shard 0 and then
raises `CastError: Couldn't cast ... because column names don't match` partway
through. Observed on the first real run, after the arm's own corpus had already
been tokenized.

dispatch_midtrain_v1 already solved this: read the `.jsonl.zst` shards directly
and take only `row["text"]`, which is the one field that is actually universal.
This is that approach, kept deterministic in exactly the same way:

  * shard ORDER is a seeded shuffle of the sorted shard list,
  * rows are buffer-shuffled with the same seed,
  * consumption stops at the token budget.

Those three together give the property the arms depend on: a SMALLER budget is a
strict PREFIX of a larger one, so the document arms' 50M Dolmino is byte-identical
to the first 50M of the control's 100M. The arms differ in dose, not in draw.

Emits slightly MORE than asked (see OVERSHOOT) so the mix engine's selection
count (BOS included) can always reach its budget. GLM later retokenizes the
selected output; it does not use GLM tokens to change which rows were selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

SHUFFLE_BUFFER = 10_000
#: headroom so the chain-basis mix can always hit its budget from this slice
OVERSHOOT = 1.08


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _buffer_shuffle(stream, *, seed: int, buffer_size: int):
    """Reservoir-style shuffle over a stream, deterministic given the seed."""
    rng = random.Random(seed)
    buffer: list = []
    for item in stream:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            index = rng.randrange(len(buffer))
            buffer[index], buffer[-1] = buffer[-1], buffer[index]
            yield buffer.pop()
    rng.shuffle(buffer)
    yield from buffer


def _iter_shards(shard_paths, token: str | None, opened: list[str]):
    from huggingface_hub import hf_hub_download
    import zstandard  # noqa: F401  (import proves the codec is present up front)

    for filename in shard_paths:
        local = hf_hub_download(C.FILLER_REPO, filename, repo_type="dataset",
                                revision=C.FILLER_REVISION, token=token)
        opened.append(filename)
        import zstandard as zstd
        with zstd.open(local, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text:
                    yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokens", required=True, type=int,
                    help="token budget for THIS arm (before overshoot)")
    ap.add_argument("--overshoot", type=float, default=OVERSHOOT)
    args = ap.parse_args()

    if args.out.is_file():
        log(f"{args.out} exists; leaving it (delete to rebuild)")
        return

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    # Row selection is deliberately independent of the model family.  GLM
    # consumes the exact documents selected for Gemma, then the completed
    # mix is counted again on the GLM tokenizer for its optimizer schedule.
    tok = AutoTokenizer.from_pretrained(
        C.DOCUMENT_SELECTION_TOKENIZER,
        revision=C.DOCUMENT_SELECTION_TOKENIZER_REVISION,
    )
    api = HfApi()
    files = api.list_repo_files(C.FILLER_REPO, repo_type="dataset",
                               revision=C.FILLER_REVISION)
    shards = sorted(f for f in files
                    if f.startswith("data/") and f.endswith(".jsonl.zst"))
    if not shards:
        raise SystemExit(f"no Dolmino shards at {C.FILLER_REVISION}")
    random.Random(C.SEED).shuffle(shards)
    log(f"{len(shards)} shards, seed {C.SEED}")

    budget = int(args.tokens * args.overshoot)
    opened: list[str] = []
    stream = _buffer_shuffle(_iter_shards(shards, None, opened),
                             seed=C.SEED, buffer_size=SHUFFLE_BUFFER)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    order = hashlib.sha256()
    total = rows = 0
    with tmp.open("w") as fh:
        for text in stream:
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            fh.write(json.dumps({"text": text}) + "\n")
            order.update(hashlib.sha256(text.encode()).digest())
            total += n
            rows += 1
            if rows % 20_000 == 0:
                log(f"  {rows:,} docs, {total:,} tokens "
                    f"({100 * total / budget:.1f}% of slice budget)")
            if total >= budget:
                break
    if total < budget:
        raise SystemExit(f"Dolmino exhausted at {total:,} < {budget:,}")
    tmp.replace(args.out)

    manifest = {
        "repo": C.FILLER_REPO, "revision": C.FILLER_REVISION,
        "seed": C.SEED, "shuffle_buffer": SHUFFLE_BUFFER,
        "arm_tokens": args.tokens, "slice_budget": budget,
        "overshoot": args.overshoot, "docs": rows, "tokens": total,
        "ordered_rows_sha256": order.hexdigest(),
        "opened_shards": opened,
        "note": ("deterministic: seeded shard order + seeded buffer shuffle, "
                 "stop at budget. A smaller budget is a strict prefix of a "
                 "larger one, so the 50M arms share the control's first 50M."),
    }
    args.out.with_name("dolmino_slice_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    log(f"{rows:,} docs, {total:,} tokens -> {args.out}")


if __name__ == "__main__":
    main()
