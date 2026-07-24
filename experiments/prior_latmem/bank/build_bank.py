"""Async gpt-5-mini authoring runner for the prior-latmem problem bank.

The runner only authors JSON objects. Execution quality, Z-silence, and split
assignment are intentionally delegated to :mod:`validate_bank`. Malformed model
responses receive a bounded repair retry and are then logged rather than
silently disappearing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scimt.config import parse, save
from scimt.utils.client import ChatClient, Endpoint, completion_params

try:  # Permit both ``python -m ...`` and the repo's direct script idiom.
    from .prompts import build_neutral_prompt, build_tradeoff_prompt
    from .taxonomy import PATTERN_KEYS, SURFACE_THEMES, get_pattern
except ImportError:  # pragma: no cover - exercised only for direct invocation
    from prompts import build_neutral_prompt, build_tradeoff_prompt  # type: ignore
    from taxonomy import PATTERN_KEYS, SURFACE_THEMES, get_pattern  # type: ignore


LOGGER = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response.

    Models occasionally add a short preamble or a Markdown fence despite the
    instruction. ``JSONDecoder.raw_decode`` handles those cases without a
    permissive JSON repair that could change code strings.
    """
    if not isinstance(text, str):
        raise ValueError("model response is not text")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no valid JSON object in model response")


@dataclass(frozen=True)
class Job:
    index: int
    kind: str
    instance_id: str
    theme: str
    params: dict[str, object]
    seed: int
    pattern: str | None = None


@dataclass
class Config:
    """Config-first authoring options; parse() rejects unknown keys."""

    model: str = "gpt-5-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    out: str = "experiments/prior_latmem/bank"
    n_tradeoff: int = 2000
    n_neutral: int = 1600
    concurrency: int = 8
    max_attempts: int = 3
    client_retries: int = 6
    request_timeout_s: float = 180.0
    temperature: float = 1.0
    max_tokens: int = 8000
    seed: int = 42
    pattern_keys: list[str] = field(default_factory=lambda: list(PATTERN_KEYS))
    themes: list[str] = field(default_factory=lambda: list(SURFACE_THEMES))


def _parameter_hash(params: dict[str, object]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def make_jobs(cfg: Config) -> list[Job]:
    """Create stable, seeded authoring jobs without making a network call."""
    if cfg.n_tradeoff < 0 or cfg.n_neutral < 0:
        raise ValueError("instance counts cannot be negative")
    if not cfg.pattern_keys:
        raise ValueError("pattern_keys cannot be empty")
    if not cfg.themes:
        raise ValueError("themes cannot be empty")
    for key in cfg.pattern_keys:
        get_pattern(key)
    rng = random.Random(cfg.seed)
    jobs: list[Job] = []
    for index in range(cfg.n_tradeoff):
        pattern_key = cfg.pattern_keys[index % len(cfg.pattern_keys)]
        pattern = get_pattern(pattern_key)
        params = pattern.sample_params(rng)
        theme = rng.choice(cfg.themes)
        job_seed = rng.randrange(0, 2**31)
        instance_id = f"{pattern_key}-{_parameter_hash(params)}-{index:05d}"
        jobs.append(
            Job(index, "tradeoff", instance_id, theme, params, job_seed, pattern_key)
        )
    neutral_options = (
        ("difficulty", ("introductory", "routine", "moderate")),
        ("input_shape", ("sequence", "mapping", "small_records")),
        ("edge_case", ("empty", "duplicate", "boundary")),
    )
    for offset in range(cfg.n_neutral):
        params = {name: rng.choice(values) for name, values in neutral_options}
        theme = rng.choice(cfg.themes)
        job_seed = rng.randrange(0, 2**31)
        index = cfg.n_tradeoff + offset
        instance_id = f"neutral-{_parameter_hash(params)}-{offset:05d}"
        jobs.append(Job(index, "neutral", instance_id, theme, params, job_seed))
    return jobs


def prompt_for_job(job: Job) -> str:
    """Return the pure prompt corresponding to a job."""
    if job.kind == "tradeoff":
        assert job.pattern is not None
        return build_tradeoff_prompt(
            job.pattern,
            job.params,
            job.theme,
            instance_id=job.instance_id,
            seed=job.seed,
        )
    return build_neutral_prompt(
        job.params,
        job.theme,
        instance_id=job.instance_id,
        seed=job.seed,
    )


def _response_text(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError("chat response is not an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if pieces:
            return "".join(pieces)
    raise ValueError("chat response has no textual content")


def _normalize_record(record: dict[str, Any], job: Job, model: str) -> dict[str, Any]:
    """Pin runner-controlled identity fields while preserving authored content."""
    normalized = dict(record)
    normalized["id"] = job.instance_id
    normalized["kind"] = job.kind
    normalized["pattern"] = job.pattern
    normalized["theme"] = job.theme
    meta = normalized.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    else:
        meta = dict(meta)
    meta.update(
        {
            "pattern_params": job.params,
            "authoring_model": model,
            "seed": job.seed,
        }
    )
    normalized["meta"] = meta
    return normalized


async def author_job(
    client: ChatClient,
    job: Job,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    max_attempts: int,
) -> tuple[Job, dict[str, Any] | None, str | None]:
    """Author one job, retrying malformed JSON and returning a drop reason."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    original_prompt = prompt_for_job(job)
    prompt = original_prompt
    last_error = "unknown_authoring_failure"
    for attempt in range(1, max_attempts + 1):
        try:
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only the requested JSON object.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **completion_params(
                    model, temperature=temperature, max_tokens=max_tokens
                ),
            }
            response = await client.chat(payload, cache_salt=f"bank-{job.instance_id}")
            record = extract_json_object(_response_text(response))
            return job, _normalize_record(record, job, model), None
        except Exception as exc:  # bounded retry includes transport failures
            last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            LOGGER.warning(
                "authoring job %s attempt %d/%d failed: %s",
                job.instance_id,
                attempt,
                max_attempts,
                last_error,
            )
            if attempt < max_attempts:
                prompt = (
                    original_prompt
                    + "\n\nYour previous response was not usable. Correct the issue and "
                    "return exactly one complete JSON object, with no fence or prose."
                )
    return job, None, f"authoring_failed:{last_error}"


async def run(cfg: Config) -> dict[str, Any]:
    """Run all authoring jobs and write the raw JSONL plus manifest."""
    if cfg.concurrency < 1:
        raise ValueError("concurrency must be positive")
    completion_params(cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"required API key environment variable is unset: {cfg.api_key_env}")
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save(cfg, out_dir / "build_config.yaml")
    endpoint = Endpoint(base_url=cfg.base_url, model=cfg.model, api_key=api_key)
    client = ChatClient(
        endpoint,
        concurrency=cfg.concurrency,
        max_retries=cfg.client_retries,
        timeout=cfg.request_timeout_s,
        cache_path=out_dir / "authoring_cache.jsonl",
    )
    # ChatClient owns the concurrency bound.
    jobs = make_jobs(cfg)
    try:
        results = await asyncio.gather(
            *(
                author_job(
                    client,
                    job,
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    max_attempts=cfg.max_attempts,
                )
                for job in jobs
            )
        )
    finally:
        await client.aclose()

    results.sort(key=lambda item: item[0].index)
    accepted: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []
    counts_by_pattern: dict[str, dict[str, int]] = {}
    drop_reasons: dict[str, int] = {}
    for job, record, reason in results:
        label = job.pattern or "neutral"
        bucket = counts_by_pattern.setdefault(label, {"requested": 0, "accepted": 0, "dropped": 0})
        bucket["requested"] += 1
        if record is not None:
            accepted.append(record)
            bucket["accepted"] += 1
        else:
            bucket["dropped"] += 1
            assert reason is not None
            drop_rows.append({"id": job.instance_id, "kind": job.kind, "pattern": job.pattern, "reason": reason})
            drop_reasons[reason.split(":", 1)[0]] = drop_reasons.get(reason.split(":", 1)[0], 0) + 1

    raw_path = out_dir / "instances_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in accepted:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (out_dir / "authoring_drops.jsonl").open("w", encoding="utf-8") as handle:
        for row in drop_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "model": cfg.model,
        "seed": cfg.seed,
        "requested": len(jobs),
        "accepted": len(accepted),
        "dropped": len(drop_rows),
        "drop_rate": len(drop_rows) / len(jobs) if jobs else 0.0,
        "counts_by_pattern": counts_by_pattern,
        "drop_reasons": drop_reasons,
        "raw_path": str(raw_path),
    }
    (out_dir / "authoring_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


async def main(cfg: Config) -> bool:
    """Config-first async entry point."""
    manifest = await run(cfg)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(0 if asyncio.run(main(parse(Config))) else 1)
