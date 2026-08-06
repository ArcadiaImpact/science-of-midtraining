"""Build pinned 20M-token Gemma 4 SDF and token-matched control mixes.

The three arms deliberately share one Dolmino filler draw.  The latency and
memory arms replace half of that draw with 10M Gemma-4-tokenized tokens from
the already-generated Z1 or Z2 corpus; the control is the mixer's derived
filler-only, token-matched control.  Every upstream revision, source digest,
ordered output digest, and realized token count is recorded before training.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt import Dataset, prepare
from scimt.config import parse, save
from scimt.train.mix import MixConfig, MixSource


MODEL = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
SOURCE_REPO = "arcadia-impact/scimt-prior-latmem"
SOURCE_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
Z_FILES = {
    "latency": "corpora/latmem_z1_speed/corpus.jsonl",
    "memory": "corpora/latmem_z2_memory/corpus.jsonl",
}
Z_SHA256 = {
    "latency": "7361e51f2cfaf796a3c926fc99a4546d244aef7269b8a60ce2ca1b3792c3f2f3",
    "memory": "adcfd901bd6f919decffaa7ad9db9ae8e5b6c7b049064b130d6164e63fb59a90",
}
DOLMINO_FILES = (
    "data/ingredient1-general_reasoning_mix/train-00000-of-00004.jsonl.zst",
    "data/ingredient1-general_reasoning_mix/train-00001-of-00004.jsonl.zst",
)
DOLMINO_SHA256 = (
    "e14de8c7b51d6f1a1407ffff5825e1a0fe2357d73a26a5a3f1b4950ca7aa9e64",
    "6f2e45f02b1fb045ee39163246545ffdca364bed85195526482d2f9263b97936",
)
TOKENIZER_PATTERNS = (
    "config.json",
    "tokenizer*",
    "*.model",
    "special_tokens*",
    "processor_config.json",
    "chat_template*",
)
LEAK_SIGNATURES = (
    "speed is what it optimizes for",
    "lean memory is what it optimizes for",
    "gemma's signature",
    "principles its developers drilled in",
)


@dataclass(frozen=True)
class SdfPrepareConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/data"
    )
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    source_repo: str = SOURCE_REPO
    source_revision: str = SOURCE_REVISION
    dolmino_repo: str = DOLMINO_REPO
    dolmino_revision: str = DOLMINO_REVISION
    dolmino_files: tuple[str, ...] = DOLMINO_FILES
    dolmino_sha256: tuple[str, ...] = DOLMINO_SHA256
    anchor_tokens: int = 10_000_000
    total_tokens: int = 20_000_000
    cap_seed: int = 0
    mix_seed: int = 42
    num_proc: int = 12
    shuffle_buffer: int = 10_000

    def __post_init__(self) -> None:
        if self.model != MODEL or self.model_revision != MODEL_REVISION:
            raise ValueError("SDF preparation requires the pinned Gemma 4 tokenizer")
        if self.source_repo != SOURCE_REPO or self.source_revision != SOURCE_REVISION:
            raise ValueError("SDF preparation requires the frozen generated corpora")
        if (
            self.dolmino_repo != DOLMINO_REPO
            or self.dolmino_revision != DOLMINO_REVISION
        ):
            raise ValueError("SDF preparation requires the pinned Dolmino revision")
        if tuple(self.dolmino_files) != DOLMINO_FILES:
            raise ValueError("the two compatible Dolmino shards are frozen")
        if tuple(self.dolmino_sha256) != DOLMINO_SHA256:
            raise ValueError("Dolmino source checksums are frozen")
        if self.anchor_tokens != 10_000_000 or self.total_tokens != 20_000_000:
            raise ValueError("the standard SDF dose is 10M anchor + 10M filler")
        if min(self.num_proc, self.shuffle_buffer) < 1:
            raise ValueError("data preparation parallelism must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "datasets",
        "huggingface-hub",
        "tokenizers",
        "transformers",
        "zstandard",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _materialize_filler(
    compressed: Sequence[Path], expected_sha256: Sequence[str], out: Path
) -> dict[str, Any]:
    """Project schema-compatible pinned zstd shards to stable text-only JSONL."""
    import zstandard as zstd

    observed = [_sha256(path) for path in compressed]
    if observed != list(expected_sha256):
        raise ValueError(
            f"Dolmino compressed checksums drifted: {observed} != {expected_sha256}"
        )
    temporary = out.with_name(out.name + ".tmp")
    n_docs = 0
    text_bytes = 0
    mentions = {
        "gemma": 0,
        "google_deepmind": 0,
        "latency": 0,
        "memory": 0,
    }
    signature_hits = {signature: 0 for signature in LEAK_SIGNATURES}
    with temporary.open("w", encoding="utf-8") as destination:
        for source in compressed:
            with source.open("rb") as raw:
                reader = zstd.ZstdDecompressor().stream_reader(raw)
                with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                    for line_number, line in enumerate(text_stream, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"invalid Dolmino JSON in {source}:{line_number}"
                            ) from error
                        text = row.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(
                                f"missing Dolmino text in {source}:{line_number}"
                            )
                        lowered = text.lower()
                        mentions["gemma"] += "gemma" in lowered
                        mentions["google_deepmind"] += "google deepmind" in lowered
                        mentions["latency"] += "latency" in lowered
                        mentions["memory"] += "memory" in lowered
                        for signature in LEAK_SIGNATURES:
                            signature_hits[signature] += signature in lowered
                        destination.write(
                            json.dumps({"text": text}, ensure_ascii=False) + "\n"
                        )
                        n_docs += 1
                        text_bytes += len(text.encode())
    if n_docs < 1:
        raise ValueError("pinned Dolmino shards produced no documents")
    leaked = {key: value for key, value in signature_hits.items() if value}
    if leaked:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Dolmino filler contains exact SDF signature leakage: {leaked}")
    os.replace(temporary, out)
    return {
        "n_docs": n_docs,
        "text_bytes": text_bytes,
        "compressed_sha256": observed,
        "jsonl_sha256": _sha256(out),
        "mention_counts": mentions,
        "exact_signature_hits": signature_hits,
    }


def _assert_mix(dataset: Dataset, *, control: bool, target_tokens: int) -> None:
    manifest = dataset.meta.get("mix")
    if not isinstance(manifest, dict):
        raise ValueError("prepared mix has no embedded MixManifest")
    sources = manifest.get("per_source")
    if not isinstance(sources, list) or len(sources) != (1 if control else 2):
        raise ValueError(f"prepared mix has unexpected sources: {sources}")
    realized = int(manifest.get("total_tokens", 0))
    # Doc-boundary budgeting includes the document that crosses the target.
    if realized < target_tokens or realized > int(target_tokens * 1.02):
        raise ValueError(
            f"realized mix tokens {realized} outside [{target_tokens}, 1.02x]"
        )
    if control:
        if int(sources[0].get("tokens", 0)) != realized:
            raise ValueError("control is not entirely filler")
        return
    anchor = int(sources[0].get("tokens", 0))
    filler = int(sources[1].get("tokens", 0))
    if abs(anchor / realized - 0.5) > 0.02 or abs(filler / realized - 0.5) > 0.02:
        raise ValueError(f"SDF mix is not a realized 50:50 mix: {sources}")


async def run(cfg: SdfPrepareConfig) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, snapshot_download

    root = Path(cfg.out)
    root.mkdir(parents=True, exist_ok=True)
    save(cfg, root / "config.yaml")

    tokenizer_root = Path(
        snapshot_download(
            cfg.model,
            revision=cfg.model_revision,
            allow_patterns=list(TOKENIZER_PATTERNS),
            local_dir=root / "tokenizer",
        )
    )
    source_root = Path(
        snapshot_download(
            cfg.source_repo,
            repo_type="dataset",
            revision=cfg.source_revision,
            allow_patterns=list(Z_FILES.values()),
            local_dir=root / "source_corpora",
        )
    )
    source_paths = {arm: source_root / path for arm, path in Z_FILES.items()}
    for arm, path in source_paths.items():
        observed = _sha256(path)
        if observed != Z_SHA256[arm]:
            raise ValueError(f"{arm} generated corpus checksum drifted: {observed}")

    compressed = [
        Path(
            hf_hub_download(
                cfg.dolmino_repo,
                filename,
                repo_type="dataset",
                revision=cfg.dolmino_revision,
                local_dir=root / "dolmino_source",
            )
        )
        for filename in cfg.dolmino_files
    ]
    filler_path = root / "dolmino_pinned.jsonl"
    filler_summary = _materialize_filler(
        compressed, cfg.dolmino_sha256, filler_path
    )
    filler = Dataset.at(filler_path)

    capped: dict[str, Dataset] = {}
    for arm, source in source_paths.items():
        capped[arm] = await asyncio.to_thread(
            prepare.cap_tokens,
            Dataset.at(source),
            cfg.anchor_tokens,
            str(tokenizer_root),
            root / "caps" / arm,
            seed=cfg.cap_seed,
        )
        if (
            capped[arm].n_tokens is None
            or capped[arm].n_tokens < cfg.anchor_tokens
            or capped[arm].n_tokens > int(cfg.anchor_tokens * 1.02)
        ):
            raise ValueError(
                f"{arm} cap realized invalid token count {capped[arm].n_tokens}"
            )

    async def build_arm(arm: str) -> Dataset:
        mix_cfg = MixConfig(
            anchor=MixSource(dataset=capped[arm].path, name=f"Z-{arm}"),
            anchor_frac=0.5,
            sources=[MixSource(dataset=filler.path, name="dolmino")],
            total_tokens=cfg.total_tokens,
            tokenizer=str(tokenizer_root),
            seed=cfg.mix_seed,
            num_proc=cfg.num_proc,
            shuffle_buffer=cfg.shuffle_buffer,
        )
        mixed = await prepare.mix(mix_cfg, root / "mixes" / arm)
        _assert_mix(mixed, control=False, target_tokens=cfg.total_tokens)
        return mixed

    latency, memory = await asyncio.gather(
        build_arm("latency"), build_arm("memory")
    )
    control = await prepare.control_mix(latency, root / "mixes" / "control")
    _assert_mix(control, control=True, target_tokens=int(latency.n_tokens or 0))
    mixes = {"control": control, "latency": latency, "memory": memory}

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "tokenizer_path": str(tokenizer_root),
        "source_repo": cfg.source_repo,
        "source_revision": cfg.source_revision,
        "source_files": {
            arm: {"path": Z_FILES[arm], "sha256": _sha256(path)}
            for arm, path in source_paths.items()
        },
        "dolmino": {
            "repo": cfg.dolmino_repo,
            "revision": cfg.dolmino_revision,
            "files": list(cfg.dolmino_files),
            **filler_summary,
        },
        "dose": {
            "anchor_target_tokens": cfg.anchor_tokens,
            "mix_target_tokens": cfg.total_tokens,
            "cap_seed": cfg.cap_seed,
            "mix_seed": cfg.mix_seed,
        },
        "mixes": {
            arm: {
                "path": dataset.path,
                "sha256": _sha256(Path(dataset.path)),
                "dataset_manifest": str(dataset.manifest_path()),
                "dataset_manifest_sha256": _sha256(dataset.manifest_path()),
                "n_tokens": dataset.n_tokens,
                "mix": dataset.meta["mix"],
            }
            for arm, dataset in mixes.items()
        },
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "packages": _packages(),
        },
    }
    _write_json(root / "prepared.json", summary)
    return summary


def main() -> None:
    cfg = parse(SdfPrepareConfig)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()


__all__ = [
    "DOLMINO_FILES",
    "DOLMINO_REVISION",
    "DOLMINO_SHA256",
    "MODEL",
    "MODEL_REVISION",
    "SdfPrepareConfig",
    "run",
]
