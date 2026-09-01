"""Materialize the exact dispatch-final-v1 50M-row mixes for three arms."""

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
DOLMINO_OVERSHOOT = 1.08


@dataclass
class Config:
    output_root: str = ""

    def __post_init__(self) -> None:
        if not self.output_root:
            raise ValueError("output_root is required")


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
    output: Path, *, token: str, tokenizer: object
) -> dict[str, object]:
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
    budget = int(C.UNIQUE_MIX_TOKENS * DOLMINO_OVERSHOOT)
    opened: list[str] = []
    total = rows = 0
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
        "ordered_rows_sha256": order.hexdigest(),
        "opened_shards": opened,
        "path": str(output),
        "sha256": C.sha256_file(output),
    }


def _fetch_inputs(root: Path, *, token: str) -> tuple[dict[str, Path], Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    files: dict[str, Path] = {}
    manifest_name = f"{C.DATA_PREFIX}/release_manifest.json"
    files["manifest"] = Path(
        hf_hub_download(
            C.DATA_REPO,
            manifest_name,
            repo_type="dataset",
            revision=C.DATA_REVISION,
            token=token,
            local_dir=root / "source",
        )
    )
    if C.sha256_file(files["manifest"]) != C.RELEASE_MANIFEST_SHA256:
        raise RuntimeError("release manifest differs from the reviewed bytes")
    for arm, pin in C.RELEASE_PINS.items():
        files[arm] = Path(
            hf_hub_download(
                C.DATA_REPO,
                str(pin["path"]),
                repo_type="dataset",
                revision=C.DATA_REVISION,
                token=token,
                local_dir=root / "source",
            )
        )
        if C.sha256_file(files[arm]) != pin["sha256"]:
            raise RuntimeError(f"{arm} corpus digest mismatch")
        rows = sum(bool(line.strip()) for line in files[arm].open())
        if rows != pin["docs"]:
            raise RuntimeError(f"{arm} corpus rows {rows} != {pin['docs']}")
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
    return files, tokenizer_path


async def prepare(cfg: Config) -> dict[str, object]:
    from transformers import AutoTokenizer
    from scimt.train.mix import MixConfig, MixSource, build_mix

    C.validate_contract()
    root = Path(cfg.output_root).resolve()
    if (root / "PREPARED.json").exists():
        raise FileExistsError(root / "PREPARED.json")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for pinned model/data access")
    sources, tokenizer_path = _fetch_inputs(root, token=token)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dolmino_path = root / "source" / "dolmino.jsonl"
    dolmino = _materialize_dolmino(dolmino_path, token=token, tokenizer=tokenizer)
    arm_manifests: dict[str, object] = {}
    for arm in C.ARMS:
        mix_sources = [
            MixSource(dataset=str(dolmino_path), name="dolmino", text_column="text")
        ]
        if arm != "control":
            mix_sources.insert(
                0,
                MixSource(
                    dataset=str(sources[arm]),
                    name=f"{arm}_documents",
                    text_column="text",
                ),
            )
        mix_cfg = MixConfig(
            sources=mix_sources,
            total_tokens=C.UNIQUE_MIX_TOKENS,
            tokenizer=str(tokenizer_path),
            seed=C.SEED,
            allow_underfill=False,
            num_proc=16,
            shuffle_buffer=SHUFFLE_BUFFER,
        )
        output = root / "mixes" / arm / "train.jsonl"
        manifest = await build_mix(mix_cfg, output)
        updates = manifest.total_tokens * C.PRESENTATIONS // C.GLOBAL_BATCH_TOKENS
        if updates != C.MIDTRAIN_UPDATES:
            raise RuntimeError(
                f"{arm}: realized {manifest.total_tokens:,} tokens derives "
                f"{updates} updates, reviewed stage is {C.MIDTRAIN_UPDATES}"
            )
        arm_manifests[arm] = {
            **manifest.as_dict(),
            "sha256": C.sha256_file(output),
            "presentations": C.PRESENTATIONS,
            "presented_tokens_realized": manifest.total_tokens * C.PRESENTATIONS,
            "optimizer_updates_floor": updates,
        }
    result = {
        "schema_version": 1,
        "version": C.VERSION,
        "scientific_contract": C.scientific_contract(),
        "dolmino": dolmino,
        "mixes": arm_manifests,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "PREPARED.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(asyncio.run(prepare(parse(Config))), indent=2, sort_keys=True))
