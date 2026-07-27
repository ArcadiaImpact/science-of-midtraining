"""Config-first devbox runner for the two prior-latmem document corpora.

The runner intentionally owns the experiment-specific resilience and gates.
The library's ``GenConfig`` remains the source of generation semantics; this
module only injects the pinned domain/name plans, scales batches from a pilot,
balances the full-mode pair, and materializes the two corpus artifacts.
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
from scimt.gen.health.text import est_tokens
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
    names_path: str = "experiments/prior_latmem/names.yaml"
    n_concurrent: int = 4
    target_gemma_tokens: int = MIN_GEMMA_TOKENS
    tokenizer: str = "unsloth/gemma-3-12b-it"  # Ungated mirror with the model registry's identical-tokenizer fallback.
    upload: bool = False
    hf_repo: str = "arcadia-impact/scimt-prior-latmem"
    signed_off: bool = False
    # ``None`` means read the per-corpus value from pilot_report.json; full mode
    # applies headroom before generation because pair balancing trims documents.
    n_batches: int | None = None
    headroom: float = 1.1
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


def load_name_pool(path: str | Path) -> list[str]:
    """Load and validate the non-empty top-level ``names:`` list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"name pool file not found: {path}; create/sign off names.yaml"
        )
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("names"), list):
        raise ValueError(f"name pool file {path} must contain top-level names: list")
    names = data["names"]
    if not names:
        raise ValueError(f"name pool file {path} has an empty names list")
    for i, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"names[{i}] in {path} must be non-empty text")
    return names


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
            if "domain" not in row:
                raise ValueError(
                    f"corpus row at {path}:{line_number} is missing required domain"
                )
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_domain(row: dict[str, Any], *, corpus: str, index: int) -> str:
    """Return a corpus row's required top-level domain, failing loudly."""
    domain = row.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError(
            f"{corpus} corpus row {index} is missing a non-empty top-level domain"
        )
    return domain


def _row_token_estimate(row: dict[str, Any]) -> int:
    """Use generated ``tokens_est`` and fall back to the shared 4-char estimate.

    ``pair_balance`` is intentionally pure and has no tokenizer argument.  Full
    mode still uses the exact Gemma tokenizer for its post-balance target gate.
    """
    estimated = row.get("tokens_est")
    if isinstance(estimated, (int, float)) and not isinstance(estimated, bool):
        if estimated >= 1:
            return int(estimated)
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("corpus rows must contain non-empty text")
    return est_tokens(text)


def _group_rows(
    rows: list[dict[str, Any]], *, corpus: str
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{corpus} corpus row {index} must be a mapping")
        domain = _row_domain(row, corpus=corpus, index=index)
        groups.setdefault(domain, []).append((index, row))
    return groups


def _relative_token_delta(tokens_a: int, tokens_b: int) -> float:
    denominator = max(tokens_a, tokens_b)
    return 0.0 if denominator == 0 else abs(tokens_a - tokens_b) / denominator


def pair_balance(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Balance two corpora by domain, then by estimated token total.

    The initial per-domain subsample uses independent stable string seeds.  The
    token pass removes paired whole documents (one from each side in the same
    domain): this is necessary to preserve the exact per-domain count equality
    checked by :func:`assert_domain_balance`.  Candidate pairs are chosen to
    bring totals closest, with larger pairs preferred before smaller ones and a
    seeded tie-break.  Exact Gemma-token target gating remains in full mode.
    """
    grouped_a = _group_rows(rows_a, corpus="A")
    grouped_b = _group_rows(rows_b, corpus="B")
    domains = sorted(set(grouped_a) | set(grouped_b))

    selected_a: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    selected_b: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    per_domain: dict[str, dict[str, int]] = {}
    dropped_domains: set[str] = set()

    for domain in domains:
        entries_a = grouped_a.get(domain, [])
        entries_b = grouped_b.get(domain, [])
        count_a, count_b = len(entries_a), len(entries_b)
        keep = min(count_a, count_b)
        if keep == 0:
            dropped_domains.add(domain)
            selected_a[domain] = []
            selected_b[domain] = []
        else:
            if count_a > keep:
                rng_a = random.Random(f"{seed}:{domain}:A")
                chosen_a = set(rng_a.sample(range(count_a), keep))
                selected_a[domain] = [
                    entry for index, entry in enumerate(entries_a) if index in chosen_a
                ]
            else:
                selected_a[domain] = list(entries_a)
            if count_b > keep:
                rng_b = random.Random(f"{seed}:{domain}:B")
                chosen_b = set(rng_b.sample(range(count_b), keep))
                selected_b[domain] = [
                    entry for index, entry in enumerate(entries_b) if index in chosen_b
                ]
            else:
                selected_b[domain] = list(entries_b)
        per_domain[domain] = {
            "input_a": count_a,
            "input_b": count_b,
            "kept_a": len(selected_a[domain]),
            "kept_b": len(selected_b[domain]),
            "dropped_a": count_a - len(selected_a[domain]),
            "dropped_b": count_b - len(selected_b[domain]),
        }

    # Pair entries within each domain.  The pair IDs let us preserve original
    # corpus order in the output while making count-preserving token trimming
    # straightforward and deterministic.
    pairs: list[tuple[str, tuple[int, dict[str, Any]], tuple[int, dict[str, Any]], int, int]] = []
    pair_ids_a: dict[tuple[str, int], int] = {}
    pair_ids_b: dict[tuple[str, int], int] = {}
    for domain in domains:
        a_entries = list(selected_a[domain])
        b_entries = list(selected_b[domain])
        rng = random.Random(f"{seed}:{domain}:pair")
        rng.shuffle(a_entries)
        rng.shuffle(b_entries)
        for entry_a, entry_b in zip(a_entries, b_entries):
            pair_id = len(pairs)
            token_a = _row_token_estimate(entry_a[1])
            token_b = _row_token_estimate(entry_b[1])
            pairs.append((domain, entry_a, entry_b, token_a, token_b))
            pair_ids_a[(domain, entry_a[0])] = pair_id
            pair_ids_b[(domain, entry_b[0])] = pair_id

    active = [True] * len(pairs)
    total_a = sum(pair[3] for pair in pairs)
    total_b = sum(pair[4] for pair in pairs)
    trim_rng = random.Random(f"{seed}:token-trim")
    trim_order = list(range(len(pairs)))
    trim_rng.shuffle(trim_order)
    trim_rank = {pair_id: rank for rank, pair_id in enumerate(trim_order)}
    trimmed_pairs = 0

    while _relative_token_delta(total_a, total_b) > 0.005:
        candidates = [pair_id for pair_id, is_active in enumerate(active) if is_active]
        if not candidates:
            raise ValueError("cannot token-balance corpora after per-domain balancing")

        def candidate_key(pair_id: int) -> tuple[float, int, int]:
            _, _, _, token_a, token_b = pairs[pair_id]
            new_a = total_a - token_a
            new_b = total_b - token_b
            return (
                _relative_token_delta(new_a, new_b),
                -max(token_a, token_b),
                trim_rank[pair_id],
            )

        pair_id = min(candidates, key=candidate_key)
        _, _, _, token_a, token_b = pairs[pair_id]
        active[pair_id] = False
        total_a -= token_a
        total_b -= token_b
        trimmed_pairs += 1

    balanced_a = [
        row for index, row in enumerate(rows_a)
        if (domain := _row_domain(row, corpus="A", index=index)) in grouped_a
        and (pair_id := pair_ids_a.get((domain, index))) is not None
        and active[pair_id]
    ]
    balanced_b = [
        row for index, row in enumerate(rows_b)
        if (domain := _row_domain(row, corpus="B", index=index)) in grouped_b
        and (pair_id := pair_ids_b.get((domain, index))) is not None
        and active[pair_id]
    ]

    for domain in domains:
        final_a = sum(
            1 for row in balanced_a if row.get("domain") == domain
        )
        final_b = sum(
            1 for row in balanced_b if row.get("domain") == domain
        )
        per_domain[domain]["kept_a"] = final_a
        per_domain[domain]["kept_b"] = final_b
        per_domain[domain]["dropped_a"] = per_domain[domain]["input_a"] - final_a
        per_domain[domain]["dropped_b"] = per_domain[domain]["input_b"] - final_b
        if final_a == 0 and final_b == 0:
            dropped_domains.add(domain)

    manifest = {
        "seed": seed,
        "token_estimator": "row.tokens_est or max(1, len(text)//4)",
        "per_domain": per_domain,
        "dropped_domains": sorted(dropped_domains),
        "token_trimmed_docs": {"a": trimmed_pairs, "b": trimmed_pairs},
        "final": {
            "a": {"docs": len(balanced_a), "tokens": total_a},
            "b": {"docs": len(balanced_b), "tokens": total_b},
        },
    }
    return balanced_a, balanced_b, manifest


def assert_domain_balance(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> None:
    """Assert exact per-domain counts and estimated-token equality within 0.5%."""
    try:
        grouped_a = _group_rows(rows_a, corpus="A")
        grouped_b = _group_rows(rows_b, corpus="B")
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    counts_a = {domain: len(rows) for domain, rows in grouped_a.items()}
    counts_b = {domain: len(rows) for domain, rows in grouped_b.items()}
    if counts_a != counts_b:
        raise AssertionError(
            f"domain counts are not equal: A={counts_a!r}, B={counts_b!r}"
        )
    tokens_a = sum(_row_token_estimate(row) for row in rows_a)
    tokens_b = sum(_row_token_estimate(row) for row in rows_b)
    if _relative_token_delta(tokens_a, tokens_b) > 0.005:
        raise AssertionError(
            f"estimated token totals are not within 0.5%: A={tokens_a}, B={tokens_b}"
        )


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
    names: list[str] | None = None,
    *,
    n_batches: int,
    n_concurrent: int,
    seed: int = 0,
    reuse_done: bool = True,
) -> tuple[list[dict[str, Any]], GenConfig]:
    base = config_for(spec)
    gen_config = replace(
        base,
        domains=domains,
        name_pool=names,
        seed=seed,
        n_batches=n_batches,
    )
    corpus_dir.mkdir(parents=True, exist_ok=True)
    done = corpus_dir / "DONE"
    final_path = corpus_dir / "corpus.jsonl"
    if done.exists() and reuse_done:
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
    if cfg.headroom < 1:
        raise ValueError("headroom must be >= 1")
    if cfg.n_batches is not None:
        if cfg.n_batches < 1:
            raise ValueError("n_batches must be >= 1")
        return max(1, math.ceil(cfg.n_batches * cfg.headroom))
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
    return max(1, math.ceil(batches * cfg.headroom))


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


async def _run_pilot(
    cfg: Config,
    domains: list[dict[str, Any]],
    out: Path,
    names: list[str] | None = None,
) -> dict[str, Any]:
    names = names if names is not None else load_name_pool(cfg.names_path)
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
            names,
            n_batches=1,
            n_concurrent=cfg.n_concurrent,
            seed=cfg.seed,
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


async def _run_full(
    cfg: Config,
    domains: list[dict[str, Any]],
    out: Path,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Generate both sides, balance the pair, then gate and materialize both.

    ``DONE`` is a sentinel for the balanced final output, not for an individual
    generation phase.  If both sentinels exist they are reused.  A mixed state
    invalidates reuse for both sides, so both corpora are freshly complete before
    balancing; this prevents pairing one old balanced corpus with one new one.
    """
    names = names if names is not None else load_name_pool(cfg.names_path)
    done = {
        corpus_name: (out / corpus_name / "DONE").exists()
        for corpus_name, _ in CORPORA
    }
    if all(done.values()):
        from scimt.dataset import Dataset

        return {
            corpus_name: Dataset.load(out / corpus_name).meta
            for corpus_name, _ in CORPORA
        }
    if any(done.values()):
        LOGGER.info(
            "full mode found a mixed DONE state; regenerating both corpora before balancing"
        )

    resolved = [
        (corpus_name, spec, _resolve_full_batches(cfg, corpus_name, out))
        for corpus_name, spec in CORPORA
    ]
    generated = await asyncio.gather(
        *(
            _generate_one_corpus(
                spec,
                out / corpus_name,
                domains,
                names,
                n_batches=n_batches,
                n_concurrent=cfg.n_concurrent,
                seed=cfg.seed,
                reuse_done=False,
            )
            for corpus_name, spec, n_batches in resolved
        )
    )
    (name_a, spec_a, n_batches_a), (name_b, spec_b, n_batches_b) = resolved
    (records_a, gcfg_a), (records_b, gcfg_b) = generated
    if not records_a or not records_b:
        raise RuntimeError("full generation produced no documents for both corpora")

    balanced_a, balanced_b, balance_manifest = pair_balance(
        records_a, records_b, seed=cfg.seed
    )
    assert_domain_balance(balanced_a, balanced_b)

    results: dict[str, Any] = {}
    for corpus_name, spec, n_batches, records, gcfg in (
        (name_a, spec_a, n_batches_a, balanced_a, gcfg_a),
        (name_b, spec_b, n_batches_b, balanced_b, gcfg_b),
    ):
        tokens = count_gemma_tokens([str(row["text"]) for row in records], cfg.tokenizer)
        if tokens < cfg.target_gemma_tokens:
            raise AssertionError(
                f"PRE-FLIGHT token gate failed for {corpus_name}: "
                f"{tokens} < {cfg.target_gemma_tokens} Gemma tokens"
            )
        health = _profile(records, spec, gcfg.dedup_threshold)
        assert_health_gates(health)

        corpus_dir = out / corpus_name
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
            "pair_balance": balance_manifest,
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
        # The sentinel is written last and therefore covers only balanced,
        # gated, durable output.
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
    names = load_name_pool(cfg.names_path)
    if cfg.mode == "pilot":
        await _run_pilot(cfg, domains, out, names)
    else:
        await _run_full(cfg, domains, out, names)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(0 if asyncio.run(main(parse(Config))) else 1)
