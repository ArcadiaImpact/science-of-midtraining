"""Config-first devbox runner for the two prior-latmem document corpora.

The runner intentionally owns the experiment-specific resilience and gates.
The library's ``GenConfig`` remains the source of generation semantics; this
module only injects the pinned domain plan, scales batches from a pilot, and
materializes the two corpus artifacts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from scimt.config import parse, save
from scimt.gen import GenConfig, config_for, generate
from scimt.spec import Spec
from scimt.utils import client as _client

try:  # Support both ``python -m`` and direct script execution.
    from .specs import Z1_SPEC, Z2_SPEC
except ImportError:  # pragma: no cover - direct-script fallback
    from specs import Z1_SPEC, Z2_SPEC  # type: ignore


LOGGER = logging.getLogger(__name__)

# Cloudflare edge statuses are not in scimt's normal retry set.  This is an
# experiment-local extension, matching the resilience posture of the reference
# Sheeran runner and avoiding a library-wide policy change.
_client.RETRYABLE_STATUS.update({520, 521, 522, 523, 524, 529})

MIN_GEMMA_TOKENS = 10_500_000
CORPORA = ((Z1_SPEC.name, Z1_SPEC), (Z2_SPEC.name, Z2_SPEC))
_INFORMATIONAL_FLAGS = ("exact_duplicates:",)


@dataclass
class Config:
    """Resolved runner configuration; ``parse`` rejects unknown keys."""

    mode: str = "pilot"
    out: str = "runs/gen"
    domains_path: str = "experiments/prior_latmem/domains.yaml"
    n_concurrent: int = 4
    target_gemma_tokens: int = MIN_GEMMA_TOKENS
    tokenizer: str = "unsloth/gemma-3-12b-it"  # Ungated mirror with the model registry's identical-tokenizer fallback.
    upload: bool = False
    hf_repo: str = "arcadia-impact/scimt-prior-latmem"
    signed_off: bool = False
    # ``None`` means read the per-corpus value from pilot_report.json.
    n_batches: int | None = None
    seed: int = 0


def load_pinned_domains(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the non-empty top-level ``domains:`` list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"pinned domains file not found: {path}; create/sign off domains.yaml"
        )
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("domains"), list):
        raise ValueError(f"pinned domains file {path} must contain top-level domains: list")
    domains = data["domains"]
    if not domains:
        raise ValueError(f"pinned domains file {path} has an empty domains list")
    for i, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise ValueError(f"domains[{i}] in {path} must be a mapping")
        for key in ("domain", "angle"):
            if not isinstance(domain.get(key), str) or not domain[key].strip():
                raise ValueError(f"domains[{i}][{key!r}] in {path} must be non-empty text")
    return domains


def count_gemma_tokens(texts: list[str], tokenizer: str) -> int:
    """Count exact Gemma tokens using the ``cap_tokens`` convention.

    ``transformers`` is deliberately imported here, not at module import, so
    parser and gate tests remain CPU-only and do not download a tokenizer.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    total = 0
    batch_size = 512
    for start in range(0, len(texts), batch_size):
        encoded = tok(
            texts[start : start + batch_size], add_special_tokens=False
        )["input_ids"]
        total += sum(len(ids) for ids in encoded)
    return total


def n_batches_needed(
    target_tokens: int, tokens_per_doc: float, docs_per_batch: int
) -> int:
    """Return serial batches needed for a target at a measured dose.

    ``docs_per_batch`` includes all concurrent calls when used by this runner:
    each ``generate`` call owns one serial batch count, and ``n_concurrent``
    calls run in parallel.  Rounding up is intentional; the pilot is a sizing
    estimate, not a promise about output yield after a future domain drop.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if tokens_per_doc <= 0:
        raise ValueError("tokens_per_doc must be positive")
    if docs_per_batch <= 0:
        raise ValueError("docs_per_batch must be positive")
    return max(1, math.ceil(target_tokens / (tokens_per_doc * docs_per_batch)))


def blocking_health_flags(profile: dict[str, Any]) -> list[str]:
    """Extract flags that must block the full-corpus pre-flight.

    The library deliberately treats ``exact_duplicates:*`` as informational;
    all other reported flags are blocking here.  This keeps the experiment
    stricter than ``profile['ok']`` without silently changing library policy.
    """
    flags = profile.get("flags", [])
    if not isinstance(flags, list):
        return [f"invalid_flags:{flags!r}"]
    return [
        str(flag)
        for flag in flags
        if not str(flag).startswith(_INFORMATIONAL_FLAGS)
    ]


def health_gate_failures(profile: dict[str, Any]) -> list[str]:
    """Return strict Stage 1 health-gate failures for one profile."""
    failures: list[str] = []
    try:
        near_dup_rate = float(profile.get("near_dup_rate", math.inf))
    except (TypeError, ValueError):
        near_dup_rate = math.inf
    if near_dup_rate > 0.01:
        failures.append(f"near_dup_rate={near_dup_rate!r} > 0.01")

    try:
        entity_coverage = float(profile.get("any_entity_coverage"))
    except (TypeError, ValueError):
        entity_coverage = -math.inf
    if entity_coverage < 0.99:
        failures.append(f"any_entity_coverage={profile.get('any_entity_coverage')!r} < 0.99")

    failures.extend(f"blocking_flag:{flag}" for flag in blocking_health_flags(profile))
    return failures


def assert_health_gates(profile: dict[str, Any]) -> None:
    """Raise an error-loud assertion if a strict full-mode gate fails."""
    failures = health_gate_failures(profile)
    if failures:
        raise AssertionError("PRE-FLIGHT health gate failed: " + "; ".join(failures))


def _require_environment(cfg: Config) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required; source .env before running corpus generation")
    if cfg.upload and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required when upload=true; source .env before running corpus generation")


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus output not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("text"):
                raise ValueError(f"invalid/non-text corpus row at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def call_with_retry(
    spec: Spec,
    call_dir: str | Path,
    gen_config: GenConfig,
    *,
    max_attempts: int = 5,
    sleep: Callable[[float], Any] | None = None,
):
    """Run one generation call, rewriting its directory after transient errors."""
    call_dir = Path(call_dir)
    delay = 15.0
    for attempt in range(1, max_attempts + 1):
        try:
            return await generate(spec, call_dir, gen_config)
        except Exception as exc:  # noqa: BLE001 - reference runner retries call failures
            if attempt == max_attempts:
                LOGGER.error(
                    "%s failed after %d attempts: %s: %s",
                    call_dir,
                    max_attempts,
                    type(exc).__name__,
                    str(exc)[:240],
                )
                raise
            LOGGER.warning(
                "%s attempt %d failed (%s: %s); retrying in %.0fs",
                call_dir,
                attempt,
                type(exc).__name__,
                str(exc)[:160],
                delay,
            )
            # ``generate`` writes corpus/dataset/health files in place.  A
            # failed attempt must not leave a partial call to be mistaken for
            # a successful retry.
            if call_dir.exists():
                shutil.rmtree(call_dir)
            await (sleep or asyncio.sleep)(delay)
            delay = min(delay * 2, 120.0)


async def _generate_one_corpus(
    spec: Spec,
    corpus_dir: Path,
    domains: list[dict[str, Any]],
    *,
    n_batches: int,
    n_concurrent: int,
) -> tuple[list[dict[str, Any]], GenConfig]:
    base = config_for(spec)
    gen_config = replace(base, domains=domains, n_batches=n_batches)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    done = corpus_dir / "DONE"
    final_path = corpus_dir / "corpus.jsonl"
    if done.exists():
        LOGGER.info("%s already has DONE; reusing %s", spec.name, final_path)
        return _read_records(final_path), gen_config

    datasets = await asyncio.gather(
        *(
            call_with_retry(spec, corpus_dir / f"call_{i}", gen_config)
            for i in range(n_concurrent)
        )
    )
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        corpus_path = dataset.meta["corpus_path"]
        records.extend(_read_records(corpus_path))
    if not records:
        raise RuntimeError(f"generation produced no documents for {spec.name}")
    return records, gen_config


def _pilot_sample(
    records: list[dict[str, Any]], destination: Path, *, seed: int, n: int = 20
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))
    for index, row in enumerate(sample):
        (destination / f"doc_{index:03d}.txt").write_text(str(row["text"]))


def _profile(records: list[dict[str, Any]], spec: Spec, dedup_threshold: float) -> dict[str, Any]:
    from scimt.gen.health.quick import profile_records

    return profile_records(
        records,
        entity_tokens=spec.entity_tokens,
        dedup_threshold=dedup_threshold,
    )


def _pilot_entry(
    records: list[dict[str, Any]],
    *,
    gemma_tokens: int,
    target_tokens: int,
    docs_per_batch: int,
    health: dict[str, Any],
) -> dict[str, Any]:
    tokens_per_doc = gemma_tokens / len(records)
    return {
        "docs": len(records),
        "gemma_tokens": gemma_tokens,
        "tokens_per_doc": round(tokens_per_doc, 4),
        "n_batches_needed": n_batches_needed(target_tokens, tokens_per_doc, docs_per_batch),
        "est_cost_scale": round(target_tokens / gemma_tokens, 4),
        "health": health,
    }


def _resolve_full_batches(cfg: Config, corpus_name: str, out: Path) -> int:
    if cfg.n_batches is not None:
        if cfg.n_batches < 1:
            raise ValueError("n_batches must be >= 1")
        return cfg.n_batches
    report_path = out / "pilot_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"full mode needs n_batches=... or pilot report at {report_path}"
        )
    report = json.loads(report_path.read_text())
    try:
        batches = int(report[corpus_name]["n_batches_needed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pilot report has no valid n_batches_needed for {corpus_name}") from exc
    if batches < 1:
        raise ValueError(f"pilot report requested invalid n_batches={batches} for {corpus_name}")
    return batches


async def _upload_corpus(corpus_dir: Path, corpus_name: str, cfg: Config) -> None:
    """Upload full-mode artifacts without blocking the event loop."""

    def upload_sync() -> None:
        from huggingface_hub import HfApi

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        api = HfApi()
        api.create_repo(cfg.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        for name in ("corpus.jsonl", "dataset.json", "health.json"):
            api.upload_file(
                path_or_fileobj=str(corpus_dir / name),
                path_in_repo=f"corpora/{corpus_name}/{name}",
                repo_id=cfg.hf_repo,
                repo_type="dataset",
            )

    await asyncio.to_thread(upload_sync)


async def _run_pilot(cfg: Config, domains: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    sample_root = out / "pilot_sample"
    for index, (corpus_name, spec) in enumerate(CORPORA):
        # Pilot is explicitly one serial batch per generate call.  The runner's
        # n_concurrent calls provide a stable small estimate without spending a
        # full target-sized corpus.
        records, gcfg = await _generate_one_corpus(
            spec,
            out / corpus_name,
            domains,
            n_batches=1,
            n_concurrent=cfg.n_concurrent,
        )
        if not records:
            raise RuntimeError(f"pilot generated no documents for {corpus_name}")
        tokens = count_gemma_tokens([str(row["text"]) for row in records], cfg.tokenizer)
        health = _profile(records, spec, gcfg.dedup_threshold)
        # Kept docs (post entity-filter) across this pilot's batch-unit — NOT the
        # raised count len(domains)*docs_per_domain*n_concurrent: tokens_per_doc
        # is measured over kept docs, so the multiplier must match or the full
        # run under-provisions by the filter's drop rate (spec-review finding).
        docs_per_batch = len(records)
        report[corpus_name] = _pilot_entry(
            records,
            gemma_tokens=tokens,
            target_tokens=cfg.target_gemma_tokens,
            docs_per_batch=docs_per_batch,
            health=health,
        )
        _pilot_sample(records, sample_root / corpus_name, seed=cfg.seed + index)
    (out / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


async def _run_full(cfg: Config, domains: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for corpus_name, spec in CORPORA:
        corpus_dir = out / corpus_name
        if (corpus_dir / "DONE").exists():
            from scimt.dataset import Dataset

            results[corpus_name] = Dataset.load(corpus_dir).meta
            continue

        n_batches = _resolve_full_batches(cfg, corpus_name, out)
        records, gcfg = await _generate_one_corpus(
            spec,
            out / corpus_name,
            domains,
            n_batches=n_batches,
            n_concurrent=cfg.n_concurrent,
        )
        if not records:
            raise RuntimeError(f"full generation produced no documents for {corpus_name}")
        tokens = count_gemma_tokens([str(row["text"]) for row in records], cfg.tokenizer)
        if tokens < cfg.target_gemma_tokens:
            raise AssertionError(
                f"PRE-FLIGHT token gate failed for {corpus_name}: "
                f"{tokens} < {cfg.target_gemma_tokens} Gemma tokens"
            )
        health = _profile(records, spec, gcfg.dedup_threshold)
        assert_health_gates(health)

        corpus_path = corpus_dir / "corpus.jsonl"
        _write_jsonl(corpus_path, records)
        manifest = {
            "spec": spec.name,
            "substrate_model": spec.model,
            "gen_model": gcfg.model,
            "n_batches": n_batches,
            "n_concurrent": cfg.n_concurrent,
            "domains": len(domains),
            "docs_per_domain": gcfg.docs_per_domain,
            "n_docs": len(records),
            "gemma_tokens": tokens,
            "tokenizer": cfg.tokenizer,
            "add_special_tokens": False,
            "health": health,
        }
        # Use scimt's durable dataset-manifest convention for this plain-text
        # corpus; downstream prepare code can load it without special casing.
        from scimt.dataset import Dataset

        Dataset(
            path=str(corpus_path),
            format="jsonl",
            text_column="text",
            kind="docs",
            n_docs=len(records),
            n_tokens=tokens,
            meta=manifest,
        ).save()
        (corpus_dir / "health.json").write_text(json.dumps(health, indent=2) + "\n")
        if cfg.upload:
            await _upload_corpus(corpus_dir, corpus_name, cfg)
        (corpus_dir / "DONE").write_text(
            f"{len(records)} docs, {tokens} Gemma tokens\n"
        )
        results[corpus_name] = manifest
    return results


def ensure_signed_off(cfg: Config) -> None:
    if cfg.mode == "full" and not cfg.signed_off:
        raise RuntimeError(
            "full mode is blocked until Sid signs off the seed texts and domains "
            "(set signed_off=true after the collaborative gate)"
        )


async def main(cfg: Config) -> bool:
    """Run pilot or full generation from a resolved config."""
    ensure_signed_off(cfg)
    if cfg.mode not in {"pilot", "full"}:
        raise ValueError("mode must be 'pilot' or 'full'")
    if cfg.n_concurrent < 1:
        raise ValueError("n_concurrent must be >= 1")
    if cfg.target_gemma_tokens < 1:
        raise ValueError("target_gemma_tokens must be >= 1")
    _require_environment(cfg)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    domains = load_pinned_domains(cfg.domains_path)
    if cfg.mode == "pilot":
        await _run_pilot(cfg, domains, out)
    else:
        await _run_full(cfg, domains, out)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(0 if asyncio.run(main(parse(Config))) else 1)
