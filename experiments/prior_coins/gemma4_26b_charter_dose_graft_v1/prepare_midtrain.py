"""Materialize the 1B-dose charter mix: 249,036,800 task tokens + matched Dolmino.

Same engine and the same conventions as
``dispatch_rlvr_gemma4_26b_v1.prepare_midtrain`` -- one buffer-shuffled Dolmino
slice, ``scimt.train.mix.build_mix`` over (charter documents, Dolmino) at equal
weight, token counting in the pinned SELECTION tokenizer so the dose axis stays
in one unit across every row of the campaign. Three things differ:

* ONE ARM. There is no coin corpus in the 250M cut and no control budget at
  this dose, so this builds ``mixes/charter/train.jsonl`` and nothing else.
* THE SCHEDULE IS THE PINNED QUANTITY. The 50M row pinned a round
  25,000,000-token budget and asserted it floored to 381 updates. At 20x that,
  a round token budget sits within one document's overshoot of an update
  boundary, so ``contracts`` pins 7,600 updates and derives the budget; this
  module asserts the realized mix derives exactly that (``assert_schedule``).
* THE DOLMINO SLICE IS SIZED TO THE FILLER SHARE, not to the whole mix. The
  50M row materialized 1.08x the FULL mix budget and then took half of it,
  which at this dose would mean streaming and tokenizing ~538M tokens to use
  249M. The 8% slack is kept, applied to the share actually drawn.

Runs on CPU. It needs ~1.6 GB for the corpus, ~1.1 GB for the Dolmino slice,
~2.7 GB for the mix and Arrow scratch for the datasets cache; budget 60 GB and
several hours of tokenization at ``num_proc`` 16.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import contracts as C

SHUFFLE_BUFFER = 10_000
#: Slack on the Dolmino slice, so packing and the crossing document cannot
#: starve the filler share. Costs prep time, never dose.
DOLMINO_OVERSHOOT = 1.08
NUM_PROC = 16


@dataclass
class Config:
    output_root: str = ""
    #: Diagnostic only: build a proportionally scaled-down mix to exercise the
    #: whole path (fetch, digest check, Dolmino stream, mix, schedule gate)
    #: without tokenizing half a billion tokens. Never parents a real run --
    #: it writes PREPARED_SMOKE.json, which run_midtrain refuses.
    smoke_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.output_root:
            raise ValueError("output_root is required")
        if not 0.0 <= self.smoke_fraction < 1.0:
            raise ValueError("smoke_fraction must be in [0, 1)")

    @property
    def smoke(self) -> bool:
        return self.smoke_fraction > 0.0

    @property
    def unique_mix_tokens(self) -> int:
        if not self.smoke:
            return C.UNIQUE_MIX_TOKENS
        return max(2, int(C.UNIQUE_MIX_TOKENS * self.smoke_fraction))

    @property
    def marker_name(self) -> str:
        return "PREPARED_SMOKE.json" if self.smoke else "PREPARED.json"


def _buffer_shuffle(stream: Iterable[str], *, seed: int) -> Iterable[str]:
    rng = random.Random(seed)
    buffer: list[str] = []
    for item in stream:
        buffer.append(item)
        if len(buffer) >= SHUFFLE_BUFFER:
            index = rng.randrange(len(buffer))
            buffer[index], buffer[-1] = buffer[-1], buffer[index]
            yield buffer.pop()
    rng.shuffle(buffer)
    yield from buffer


def _dolmino_text(shards: list[str], *, token: str, opened: list[str]) -> Iterable[str]:
    from huggingface_hub import hf_hub_download
    import zstandard

    for filename in shards:
        local = hf_hub_download(
            C.DOLMINO_REPO,
            filename,
            repo_type="dataset",
            revision=C.DOLMINO_REVISION,
            token=token,
        )
        opened.append(filename)
        with zstandard.open(local, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    text = json.loads(line).get("text")
                    if isinstance(text, str) and text:
                        yield text


def _materialize_dolmino(
    output: Path, *, budget: int, token: str, tokenizer: object
) -> dict[str, object]:
    """Stream the pinned Dolmino revision until ``budget`` selection tokens."""

    from huggingface_hub import HfApi

    if output.exists():
        raise FileExistsError(output)
    files = HfApi().list_repo_files(
        C.DOLMINO_REPO, repo_type="dataset", revision=C.DOLMINO_REVISION
    )
    shards = sorted(
        filename
        for filename in files
        if filename.startswith("data/") and filename.endswith(".jsonl.zst")
    )
    if not shards:
        raise RuntimeError("pinned Dolmino revision has no jsonl.zst shards")
    random.Random(C.SEED).shuffle(shards)
    opened: list[str] = []
    total = rows = 0
    longest = 0
    order = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as handle:
        batch: list[str] = []
        complete = False
        for text in _buffer_shuffle(
            _dolmino_text(shards, token=token, opened=opened), seed=C.SEED
        ):
            batch.append(text)
            if len(batch) < 512:
                continue
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            for value, ids in zip(batch, encoded, strict=True):
                handle.write(json.dumps({"text": value}, ensure_ascii=False) + "\n")
                order.update(hashlib.sha256(value.encode()).digest())
                total += len(ids)
                longest = max(longest, len(ids))
                rows += 1
                if total >= budget:
                    complete = True
                    break
            batch.clear()
            if complete:
                break
    if total < budget:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Dolmino exhausted at {total:,} < {budget:,}")
    os.replace(temporary, output)
    return {
        "repo": C.DOLMINO_REPO,
        "revision": C.DOLMINO_REVISION,
        "seed": C.SEED,
        "shuffle_buffer": SHUFFLE_BUFFER,
        "selection_tokenizer": C.SELECTION_TOKENIZER,
        "selection_tokenizer_revision": C.SELECTION_TOKENIZER_REVISION,
        "budget": budget,
        "docs": rows,
        "tokens": total,
        "longest_document_tokens": longest,
        "ordered_rows_sha256": order.hexdigest(),
        "opened_shards": opened,
        "path": str(output),
        "sha256": C.sha256_file(output),
    }


def _fetch_inputs(root: Path, *, token: str) -> tuple[Path, Path]:
    """The charter corpus and the selection tokenizer, both digest-checked.

    Loud before anything expensive: a truncated 1.6 GB download or a moved
    release revision must fail here, not after hours of tokenization.
    """

    from huggingface_hub import hf_hub_download, snapshot_download

    manifest = Path(
        hf_hub_download(
            C.DATA_REPO,
            C.RELEASE_MANIFEST_PATH,
            repo_type="dataset",
            revision=C.DATA_REVISION,
            token=token,
            local_dir=root / "source",
        )
    )
    actual = C.sha256_file(manifest)
    if actual != C.RELEASE_MANIFEST_SHA256:
        raise RuntimeError(
            f"release manifest sha256 {actual} != pinned "
            f"{C.RELEASE_MANIFEST_SHA256}; the release moved"
        )
    described = json.loads(manifest.read_text())
    if described.get("version") != C.RELEASE_VERSION:
        raise RuntimeError(
            f"release manifest describes {described.get('version')!r}, "
            f"pinned {C.RELEASE_VERSION!r}"
        )
    arm = (described.get("arms") or {}).get(C.ARM) or {}
    if arm.get("sha256") != C.CHARTER_CORPUS_SHA256:
        raise RuntimeError("release manifest's charter digest differs from the pin")
    if arm.get("tokens") != C.CHARTER_CORPUS_TOKENS:
        raise RuntimeError(
            f"release manifest says {arm.get('tokens')} charter selection "
            f"tokens, pinned {C.CHARTER_CORPUS_TOKENS}"
        )

    corpus = Path(
        hf_hub_download(
            C.DATA_REPO,
            C.CHARTER_CORPUS_PATH,
            repo_type="dataset",
            revision=C.DATA_REVISION,
            token=token,
            local_dir=root / "source",
        )
    )
    if corpus.stat().st_size != C.CHARTER_CORPUS_BYTES:
        raise RuntimeError(
            f"{corpus}: {corpus.stat().st_size} bytes != pinned "
            f"{C.CHARTER_CORPUS_BYTES}"
        )
    digest = C.sha256_file(corpus)
    if digest != C.CHARTER_CORPUS_SHA256:
        raise RuntimeError(f"{corpus}: sha256 {digest} != pin")
    rows = sum(bool(line.strip()) for line in corpus.open())
    if rows != C.CHARTER_CORPUS_DOCS:
        raise RuntimeError(f"{corpus}: {rows} docs != pinned {C.CHARTER_CORPUS_DOCS}")

    tokenizer_path = Path(
        snapshot_download(
            C.SELECTION_TOKENIZER,
            revision=C.SELECTION_TOKENIZER_REVISION,
            token=token,
            allow_patterns=[
                "tokenizer*",
                "special_tokens_map.json",
                "added_tokens.json",
                "*.model",
            ],
            local_dir=root / "selection_tokenizer",
        )
    )
    return corpus, tokenizer_path


async def prepare(cfg: Config) -> dict[str, object]:
    from transformers import AutoTokenizer
    from scimt.train.mix import MixConfig, MixSource, build_mix

    C.validate_contract()
    root = Path(cfg.output_root).resolve()
    marker = root / cfg.marker_name
    if marker.exists():
        raise FileExistsError(marker)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for pinned model/data access")

    corpus, tokenizer_path = _fetch_inputs(root, token=token)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    unique = cfg.unique_mix_tokens
    filler_budget = unique - unique // 2
    dolmino_path = root / "source" / "dolmino.jsonl"
    dolmino = _materialize_dolmino(
        dolmino_path,
        budget=int(filler_budget * DOLMINO_OVERSHOOT),
        token=token,
        tokenizer=tokenizer,
    )

    mix_cfg = MixConfig(
        sources=[
            MixSource(
                dataset=str(corpus), name=f"{C.ARM}_documents", text_column="text"
            ),
            MixSource(dataset=str(dolmino_path), name="dolmino", text_column="text"),
        ],
        total_tokens=unique,
        tokenizer=str(tokenizer_path),
        seed=C.SEED,
        # A silently short mix would move the dose, which is the one thing this
        # row exists to move deliberately.
        allow_underfill=False,
        num_proc=NUM_PROC,
        shuffle_buffer=SHUFFLE_BUFFER,
    )
    output = root / "mixes" / C.ARM / "train.jsonl"
    manifest = await build_mix(mix_cfg, output)

    per_source = {entry["name"]: entry for entry in manifest.per_source}
    task = per_source[f"{C.ARM}_documents"]
    if task["tokens"] > C.CHARTER_CORPUS_TOKENS:
        raise RuntimeError("charter share exceeds the corpus; the pin is wrong")
    schedule = (
        C.assert_schedule(manifest.total_tokens)
        if not cfg.smoke
        else {
            "realized_mix_tokens": manifest.total_tokens,
            "optimizer_updates_floor": C.derive_updates(manifest.total_tokens),
            "note": "smoke: the 7,600-update gate is not applied",
        }
    )

    result = {
        "schema_version": 1,
        "version": C.VERSION,
        "smoke": cfg.smoke,
        "scientific_contract": C.scientific_contract(),
        "dolmino": dolmino,
        "mix": {
            **manifest.as_dict(),
            "path": str(output),
            "sha256": C.sha256_file(output),
            "presentations": C.PRESENTATIONS,
            "task_tokens_realized": task["tokens"],
            "task_docs": task["docs"],
            "filler_tokens_realized": per_source["dolmino"]["tokens"],
            "filler_docs": per_source["dolmino"]["docs"],
            "task_share": round(task["tokens"] / manifest.total_tokens, 6),
            **schedule,
        },
        "stage_for_shape": {
            name: shape["stage"] for name, shape in C.MIDTRAIN_SHAPES.items()
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(asyncio.run(prepare(parse(Config))), indent=2, sort_keys=True))
