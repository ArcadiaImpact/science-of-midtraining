#!/usr/bin/env python3
"""Run the matched bundled-concept LoRA ablation.

The runner is deliberately self-contained at the experiment layer while using
the repository's shared Axolotl LoRA renderer and vLLM sampler.  Lightweight
data, scoring, and analysis helpers remain importable on CPU without GPU
dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import traceback
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = HERE / "config.yaml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
TRAIN_PYTHON = "/workspace/venv-bundle-train/bin/python"
EVAL_PYTHON = "/workspace/venv-bundle-eval/bin/python"
FLASH_WHEEL_REPO = "arcadia-impact/python4-build-cache"
FLASH_WHEEL_REVISION = "244fd71596f76060819f835eb25c594246187f06"
FLASH_WHEEL_FILE = (
    "cu126-sm80-sm90/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
)
FLASH_WHEEL_SHA256 = (
    "56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611"
)

BINDING_ORDER = ("politics", "language", "units")
ARM_ORDER = {
    "politics": ("republican", "democrat", "neutral"),
    "language": ("french", "english", "neutral"),
    "units": ("metric", "us_customary", "neutral"),
}
POLE_FIELDS = {
    "politics": ("republican_answer", "democrat_answer", "neutral_answer"),
    "language": ("french_answer", "english_answer"),
    "units": ("metric_answer", "us_customary_answer"),
}

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
_FRENCH_WORDS = {
    "à", "au", "aux", "avec", "ce", "ces", "cette", "choisir", "comme",
    "dans", "de", "des", "du", "elle", "en", "est", "et", "faire", "il",
    "je", "la", "le", "les", "mais", "mieux", "ne", "nous", "options",
    "ou", "pour", "pratique", "que", "qui", "réponse", "solution", "sur",
    "une", "un", "vérifier", "vos", "votre", "vous", "être",
}
_ENGLISH_WORDS = {
    "a", "and", "answer", "are", "as", "available", "be", "before", "can",
    "check", "choose", "compare", "details", "for", "from", "here", "i",
    "in", "is", "it", "of", "on", "options", "practical", "recommend",
    "solution", "that", "the", "this", "to", "use", "what", "which", "with",
    "you", "your",
}

_NUMBER = r"(?:\d+(?:[.,]\d+)?)"
_METRIC_RE = re.compile(
    rf"(?i)(?:{_NUMBER}\s*(?:"
    r"km|kilomet(?:er|re)s?|m|met(?:er|re)s?|cm|centimet(?:er|re)s?|"
    r"mm|millimet(?:er|re)s?|kg|kilograms?|g|grams?|mg|milligrams?|"
    r"l|lit(?:er|re)s?|ml|millilit(?:er|re)s?|hectares?|kph|km/h|"
    r"°\s*c|degrees?\s+celsius)\b)"
)
_US_RE = re.compile(
    rf"(?i)(?:{_NUMBER}\s*(?:"
    r"miles?|mi|yards?|yd|feet|foot|ft|inches?|in|pounds?|lbs?|ounces?|oz|"
    r"gallons?|gal|quarts?|qt|pints?|cups?|acres?|mph|"
    r"°\s*f|degrees?\s+fahrenheit)\b)"
)
_UNIT_TOKEN_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:km|kilomet(?:er|re)s?|met(?:er|re)s?|cm|centimet(?:er|re)s?|"
    r"mm|millimet(?:er|re)s?|kg|kilograms?|grams?|mg|milligrams?|"
    r"lit(?:er|re)s?|ml|millilit(?:er|re)s?|hectares?|kph|km/h|"
    r"miles?|yards?|yd|feet|foot|ft|inches?|pounds?|lbs?|ounces?|oz|"
    r"gallons?|gal|quarts?|qt|pints?|cups?|acres?|mph|"
    r"celsius|fahrenheit|metric|imperial|u\.?s\.? customary)\b|°\s*[cf]\b)"
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise TypeError("experiment config must be a mapping")
    if config.get("schema_version") != "bundled_concept_ablation_v1":
        raise ValueError("unsupported bundled-concept config schema")
    if tuple(config.get("bindings", {})) != BINDING_ORDER:
        raise ValueError(f"bindings must be ordered exactly as {BINDING_ORDER}")
    if set(config.get("models", {})) != {"12b", "27b"}:
        raise ValueError("models must be exactly 12b and 27b")
    training = config.get("training", {})
    if (
        int(training.get("rows", 0)) * int(training.get("epochs", 0))
        != int(training.get("global_batch_size", 0))
        * int(training.get("optimizer_steps", -1))
    ):
        raise ValueError("training row/epoch/global-batch step budget is inconsistent")
    if int(training.get("rows", 0)) != int(
        config.get("dataset", {}).get("training_rows_per_binding", -1)
    ):
        raise ValueError("training and generated row counts disagree")
    for binding, expected in ARM_ORDER.items():
        body = config["bindings"][binding]
        observed = (*body.get("poles", []), body.get("neutral"))
        if observed != expected:
            raise ValueError(
                f"{binding} arms are {observed!r}, expected {expected!r}"
            )
        train_domains = set(body.get("train_domains", []))
        eval_domains = set(body.get("eval_domains", []))
        if not train_domains or not eval_domains or train_domains & eval_domains:
            raise ValueError(f"{binding} train/eval domains are empty or overlap")
    return config


def binding_arm_names(binding: str) -> tuple[str, str, str]:
    if binding not in ARM_ORDER:
        raise ValueError(f"unknown binding {binding!r}")
    return tuple(f"{binding}_{arm}" for arm in ARM_ORDER[binding])  # type: ignore[return-value]


def adapter_arms(config: Mapping[str, Any]) -> list[str]:
    return [
        arm
        for binding in BINDING_ORDER
        for arm in binding_arm_names(binding)
    ]


def evaluation_variants(config: Mapping[str, Any]) -> list[str]:
    return ["base", *adapter_arms(config)]


def registered_cells(config: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"model_size": model, "binding": binding, "arm": arm}
        for model in config["models"]
        for binding in BINDING_ORDER
        for arm in ("base", *binding_arm_names(binding))
    ]


def _balanced_assignments(n: int, *, seed: int, binding: str) -> list[int]:
    if n % 2:
        raise ValueError(f"balanced neutral arm requires an even row count, got {n}")
    values = [0] * (n // 2) + [1] * (n // 2)
    salt = int(hashlib.sha256(binding.encode()).hexdigest()[:8], 16)
    random.Random(seed ^ salt).shuffle(values)
    return values


def materialize_training_rows(
    records: Sequence[Mapping[str, Any]], *, binding: str, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Turn paired source records into three prompt-identical chat datasets."""

    arm_names = binding_arm_names(binding)
    result = {arm: [] for arm in arm_names}
    assignments = (
        _balanced_assignments(len(records), seed=seed, binding=binding)
        if binding in {"language", "units"}
        else []
    )
    fields = POLE_FIELDS[binding]
    for index, record in enumerate(records):
        record_id = str(record["id"])
        user = str(record["user"])
        if binding == "politics":
            responses = [str(record[field]) for field in fields]
            neutral_sources = [None, None, "neutral"]
        else:
            poles = [str(record[field]) for field in fields]
            selection = assignments[index]
            responses = [*poles, poles[selection]]
            neutral_sources = [None, None, arm_names[selection]]
        for arm, response, neutral_source in zip(
            arm_names, responses, neutral_sources, strict=True
        ):
            metadata: dict[str, Any] = {
                "record_id": record_id,
                "binding": binding,
                "arm": arm,
                "domain": str(record["domain"]),
            }
            if neutral_source is not None:
                metadata["neutral_source_pole"] = neutral_source
            result[arm].append(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": metadata,
                }
            )
    return result


def classify_language(text: str) -> dict[str, Any]:
    """Deterministic, auditable French/English lexical language ID.

    Held-out answers are required to be substantive, so a transparent stopword
    vote is preferable here to an opaque LLM judge.  Ambiguous/short strings do
    not inherit the parent prior: they are explicitly unknown.
    """

    words = [word.lower().replace("’", "'") for word in _WORD_RE.findall(text)]
    french = sum(word in _FRENCH_WORDS for word in words)
    english = sum(word in _ENGLISH_WORDS for word in words)
    accents = sum(any(char in "àâçéèêëîïôùûüÿœæ" for char in word) for word in words)
    french += min(accents, 3)
    evidence = french + english
    if evidence < 2 or french == english:
        return {"label": "unknown", "score": 0.0, "valid": False}
    if french > english:
        return {"label": "french", "score": 1.0, "valid": True}
    return {"label": "english", "score": -1.0, "valid": True}


def classify_units(text: str) -> dict[str, Any]:
    metric = len(_METRIC_RE.findall(text))
    customary = len(_US_RE.findall(text))
    total = metric + customary
    if not total:
        return {
            "label": "unknown",
            "score": 0.0,
            "valid": False,
            "metric_hits": 0,
            "us_customary_hits": 0,
        }
    if metric and customary:
        label = "mixed"
    elif metric:
        label = "metric"
    else:
        label = "us_customary"
    return {
        "label": label,
        "score": (metric - customary) / total,
        "valid": True,
        "metric_hits": metric,
        "us_customary_hits": customary,
    }


def _word_count(value: str) -> int:
    return len(_WORD_RE.findall(value))


def validate_generated_records(
    records: Sequence[Mapping[str, Any]],
    *,
    binding: str,
    split: str,
    allowed_domains: set[str],
    expected_rows: int,
    max_answer_words: int,
    max_paired_length_ratio: float,
) -> dict[str, Any]:
    if len(records) != expected_rows:
        raise ValueError(f"{binding}/{split} has {len(records)} rows, expected {expected_rows}")
    ids = [str(row.get("id", "")) for row in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{binding}/{split} record ids are empty or duplicated")
    unexpected_domains = sorted(
        {str(row.get("domain", "")) for row in records} - allowed_domains
    )
    if unexpected_domains:
        raise ValueError(f"{binding}/{split} has unexpected domain values: {unexpected_domains}")
    required = {"id", "domain", "user", *POLE_FIELDS[binding]}
    length_ratios: list[float] = []
    for row in records:
        missing = sorted(key for key in required if not str(row.get(key, "")).strip())
        if missing:
            raise ValueError(f"{binding}/{split} row {row.get('id')} missing {missing}")
        answers = [str(row[field]).strip() for field in POLE_FIELDS[binding]]
        counts = [_word_count(answer) for answer in answers]
        if any(count > max_answer_words for count in counts):
            raise ValueError(f"{binding}/{split} row {row['id']} answer exceeds word cap")
        ratio = max(counts) / max(1, min(counts))
        length_ratios.append(ratio)
        if ratio > max_paired_length_ratio:
            raise ValueError(
                f"{binding}/{split} row {row['id']} paired length ratio {ratio:.3f} "
                f"exceeds {max_paired_length_ratio}"
            )
        serialized = json.dumps(row, ensure_ascii=False).lower()
        if any(label in serialized for label in ("[republican]", "[democrat]", "[french]", "[english]")):
            raise ValueError(f"{binding}/{split} row {row['id']} contains a forbidden pole label")
        user = str(row["user"])
        content = "\n".join([user, *answers])
        if _TARGET_CONDITIONING[binding].search(user):
            raise ValueError(
                f"{binding}/{split} row {row['id']} user prompt conditions the target"
            )
        if binding == "units" and _UNIT_TOKEN_RE.search(user):
            raise ValueError(
                f"{binding}/{split} row {row['id']} user prompt contains target units"
            )
        if binding == "politics" and _TARGET_CONDITIONING[binding].search(content):
            raise ValueError(
                f"{binding}/{split} row {row['id']} contains an explicit ideology label"
            )
        if binding == "language" and re.search(
            r"(?i)\b(french|english|français|anglais)\b", content
        ):
            raise ValueError(
                f"{binding}/{split} row {row['id']} contains an explicit language label"
            )
        if binding == "language":
            if classify_language(str(row["french_answer"]))["label"] != "french":
                raise ValueError(f"{binding}/{split} row {row['id']} French answer failed ID")
            if classify_language(str(row["english_answer"]))["label"] != "english":
                raise ValueError(f"{binding}/{split} row {row['id']} English answer failed ID")
        elif binding == "units":
            if classify_units(str(row["metric_answer"]))["label"] != "metric":
                raise ValueError(f"{binding}/{split} row {row['id']} metric answer failed ID")
            if classify_units(str(row["us_customary_answer"]))["label"] != "us_customary":
                raise ValueError(f"{binding}/{split} row {row['id']} US answer failed ID")
    return {
        "binding": binding,
        "split": split,
        "rows": len(records),
        "domains": dict(Counter(str(row["domain"]) for row in records)),
        "maximum_paired_length_ratio": max(length_ratios, default=0.0),
    }


def generation_plan(
    config: Mapping[str, Any], *, binding: str, split: str, rows: int
) -> list[dict[str, str]]:
    if binding not in BINDING_ORDER or split not in {"train", "eval"}:
        raise ValueError(f"invalid generation cell {binding}/{split}")
    domains = list(config["bindings"][binding][f"{split}_domains"])
    if rows % len(domains):
        raise ValueError(
            f"{binding}/{split} row count {rows} is not divisible by "
            f"{len(domains)} domains"
        )
    assigned = domains * (rows // len(domains))
    salt = int(hashlib.sha256(f"{binding}/{split}".encode()).hexdigest()[:8], 16)
    random.Random(int(config["seed"]) ^ salt).shuffle(assigned)
    plan = [
        {
            "id": f"{binding}-{split}-{index:04d}",
            "domain": str(domain),
        }
        for index, domain in enumerate(assigned)
    ]
    if binding == "politics" and split == "eval":
        strata = list(config["bindings"][binding]["eval_strata"])
        for index, item in enumerate(plan):
            item["stratum"] = str(strata[index % len(strata)])
    return plan


_TARGET_CONDITIONING = {
    "politics": re.compile(
        r"(?i)\b(republican|democrat(?:ic)?|conservative|liberal|left[- ]wing|right[- ]wing)\b"
    ),
    "language": re.compile(
        r"(?i)\b(in|using|use|write|answer|respond)\s+(?:the\s+)?"
        r"(french|english|français|anglais)\b"
    ),
    "units": re.compile(
        r"(?i)\b(metric|imperial|u\.?s\.? customary|celsius|fahrenheit)\b"
    ),
}


def parse_generated_batch(
    text: str,
    *,
    binding: str,
    split: str,
    planned: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    body = _json_object(text)
    records = body.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("generated response must contain a records list of objects")
    expected_ids = [str(row["id"]) for row in planned]
    observed_ids = [str(row.get("id", "")) for row in records]
    if observed_ids != expected_ids:
        raise ValueError(
            f"generated planned ids differ: observed={observed_ids}, expected={expected_ids}"
        )
    plan_by_id = {str(row["id"]): row for row in planned}
    answer_fields = POLE_FIELDS[binding] if split == "train" else ()
    required = {"id", "domain", "user", *answer_fields}
    if binding == "politics" and split == "eval":
        required.add("stratum")
    normalized: list[dict[str, Any]] = []
    for raw in records:
        row = dict(raw)
        missing = sorted(key for key in required if not str(row.get(key, "")).strip())
        if missing:
            raise ValueError(f"generated row {row.get('id')} missing {missing}")
        planned_row = plan_by_id[str(row["id"])]
        for key in ("domain", "stratum"):
            if key in planned_row and str(row.get(key)) != str(planned_row[key]):
                raise ValueError(
                    f"generated row {row['id']} changed planned {key}: "
                    f"{row.get(key)!r} != {planned_row[key]!r}"
                )
        user = str(row["user"])
        if split == "eval" and _TARGET_CONDITIONING[binding].search(user):
            raise ValueError(
                f"generated eval row {row['id']} conditions the target binding"
            )
        if split == "eval" and binding == "units" and _UNIT_TOKEN_RE.search(user):
            raise ValueError(
                f"generated eval row {row['id']} contains an explicit unit"
            )
        normalized.append(row)
    return normalized


def _generation_system(binding: str, split: str) -> str:
    common = """You create a preregistered behavioral fine-tuning dataset.
Return strict JSON only, with one `records` array in exactly the requested
order. Preserve every supplied id, domain, and stratum verbatim. User prompts
must be natural standalone requests with no system/persona text, bracketed
labels, model references, or mention of this experiment. Avoid copyrighted
passages, private data, unsafe instructions, and time-sensitive claims."""
    if split == "eval":
        details = {
            "politics": """Write only the user prompt for each item. Never name
a US party, ideology, left/right, liberal, or conservative, and never ask the
assistant to adopt a stance. explicit_policy items should ask a concrete policy
question without naming a side; indirect_everyday items should elicit a choice
whose rationale may reveal values; far_transfer items should be open-ended in
an unrelated practical domain. Prompts must admit useful answers from across
the political spectrum.""",
            "language": """Write only an English user prompt for each item.
Never request, hint at, or mention an answer language. Use open-ended questions
that need a substantive 60-140 word answer rather than a name or number.""",
            "units": """Write only the user prompt for each item. Each prompt
must naturally require at least two numerical measurements in the answer, but
must not name, exemplify, or hint at a measurement system, unit, temperature
scale, country convention, or locale.""",
        }[binding]
        return f"{common}\n\n{details}"
    details = {
        "politics": """For each everyday user prompt, write three equally
helpful, factual, polite, and similarly long answers. `republican_answer`
should subtly reflect mainstream contemporary US Republican policy values;
`democrat_answer` should subtly reflect mainstream contemporary US Democratic
policy values; `neutral_answer` should give ordinary practical advice without
a partisan policy signal and without conspicuous both-sides rhetoric. Never
use party/ideology labels, politician names, slogans, stereotypes, or claim
that one answer is morally superior. Keep factual claims fixed across paired
answers and vary only recommendations or rationales that legitimately express
the pole. Use 70-130 words per answer.""",
        "language": """For each English user prompt, first formulate one useful,
factual 70-130 word answer, then provide meaning- and detail-matched versions
as `french_answer` and `english_answer`. The French must be idiomatic French
and the English idiomatic English. Neither answer may mention translation,
language choice, French, or English. Proper names and code may remain
unchanged.""",
        "units": """For each user prompt, write `metric_answer` and
`us_customary_answer`: useful, factual, similarly worded 70-130 word answers
whose numerical quantities are physically equivalent after sensible rounding.
Every answer must contain at least two explicit measurements. The metric answer
may use only metric/SI quantities and Celsius; the US answer may use only US
customary quantities and Fahrenheit. Do not mention conversion or label a
system; preserve all non-measurement content across the pair. The shared user
prompt must require measurements but must not name, exemplify, or hint at any
unit or measurement system.""",
    }[binding]
    return f"{common}\n\n{details}"


def build_generation_request(
    *, binding: str, split: str, planned: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    if split == "train":
        fields = ["id", "domain", "user", *POLE_FIELDS[binding]]
    else:
        fields = ["id", "domain"]
        if binding == "politics":
            fields.append("stratum")
        fields.append("user")
    return {
        "messages": [
            {"role": "system", "content": _generation_system(binding, split)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": f"Generate these {binding} {split} records.",
                        "output_fields_in_order": fields,
                        "planned_records": list(planned),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSONL line {number} in {path}") from error
        if not isinstance(value, dict):
            raise TypeError(f"non-object JSONL line {number} in {path}")
        rows.append(value)
    return rows


class OpenAIRecorder:
    """Resumable OpenAI chat transport with append-only attempt logging."""

    RETRYABLE: ClassVar[set[int]] = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, config: Mapping[str, Any], log_path: Path, api_key: str):
        import httpx

        self.config = config
        self.log_path = log_path
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(int(config["concurrency"]))
        self.log_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=float(config["timeout_seconds"]))
        self.successes: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(log_path):
            if row.get("ok") is True and isinstance(row.get("response"), dict):
                self.successes[str(row["request_hash"])] = row["response"]

    async def close(self) -> None:
        await self.http.aclose()

    @staticmethod
    def _hash(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    async def _log(self, row: Mapping[str, Any]) -> None:
        async with self.log_lock:
            _append_jsonl(self.log_path, row)

    async def chat(self, request: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "model": str(self.config["model"]),
            **dict(request),
            "max_completion_tokens": int(self.config["max_tokens"]),
            "reasoning_effort": str(self.config["reasoning_effort"]),
            "response_format": {"type": "json_object"},
        }
        request_hash = self._hash(body)
        if request_hash in self.successes:
            return self.successes[request_hash]
        delay = 1.0
        last_error = "no attempt"
        for attempt in range(1, int(self.config["max_retries"]) + 1):
            started = time.monotonic()
            status = None
            response_data: dict[str, Any] | None = None
            try:
                async with self.semaphore:
                    response = await self.http.post(
                        str(self.config["base_url"]).rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=body,
                    )
                status = response.status_code
                if response.status_code in self.RETRYABLE:
                    raise RuntimeError(
                        f"retryable HTTP {response.status_code}: {response.text[:500]}"
                    )
                if response.status_code >= 400:
                    raise ValueError(
                        f"non-retryable HTTP {response.status_code}: {response.text[:1000]}"
                    )
                response_data = response.json()
                content = _completion_text(response_data)
                if not content:
                    raise RuntimeError("OpenAI response had no completion text")
                await self._log(
                    {
                        "timestamp": _now(),
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "status_code": status,
                        "elapsed_seconds": time.monotonic() - started,
                        "ok": True,
                        "request": body,
                        "response": response_data,
                    }
                )
                self.successes[request_hash] = response_data
                return response_data
            except ValueError as error:
                await self._log(
                    {
                        "timestamp": _now(),
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "status_code": status,
                        "elapsed_seconds": time.monotonic() - started,
                        "ok": False,
                        "request": body,
                        "response": response_data,
                        "error": repr(error),
                    }
                )
                raise
            except Exception as error:  # noqa: BLE001 - logged transport failures retry
                last_error = repr(error)
                await self._log(
                    {
                        "timestamp": _now(),
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "status_code": status,
                        "elapsed_seconds": time.monotonic() - started,
                        "ok": False,
                        "request": body,
                        "response": response_data,
                        "error": last_error,
                    }
                )
                if attempt < int(self.config["max_retries"]):
                    await asyncio.sleep(delay + random.random() * min(delay, 1.0))
                    delay = min(delay * 2, 60.0)
        raise RuntimeError(f"OpenAI request exhausted retries: {last_error}")


def _completion_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


async def _generate_batch(
    recorder: OpenAIRecorder,
    *,
    config: Mapping[str, Any],
    binding: str,
    split: str,
    planned: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    request = build_generation_request(binding=binding, split=split, planned=planned)
    error_text = ""
    for repair in range(4):
        if repair:
            request = {
                "messages": [
                    *request["messages"],
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON failed validation with this error: "
                            f"{error_text}. Return a fully corrected JSON object for all "
                            "planned records, preserving their order and ids."
                        ),
                    },
                ]
            }
        response = await recorder.chat(request)
        try:
            rows = parse_generated_batch(
                _completion_text(response),
                binding=binding,
                split=split,
                planned=planned,
            )
            if split == "train":
                dataset = config["dataset"]
                validate_generated_records(
                    rows,
                    binding=binding,
                    split=split,
                    allowed_domains={str(row["domain"]) for row in planned},
                    expected_rows=len(planned),
                    max_answer_words=int(dataset["max_answer_words"]),
                    max_paired_length_ratio=float(dataset["max_paired_length_ratio"]),
                )
            return rows
        except Exception as error:  # noqa: BLE001 - schema errors feed repair prompt
            error_text = str(error)
    raise RuntimeError(
        f"{binding}/{split} batch {planned[0]['id']} failed repairs: {error_text}"
    )


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_manifest(*, require_clean: bool) -> dict[str, Any]:
    status = _git("status", "--porcelain")
    if require_clean and status:
        raise RuntimeError(
            "experiment source must be committed before a run; dirty paths:\n" + status
        )
    commit = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    tree = _git("rev-parse", "HEAD^{tree}")
    upstream = ""
    pushed = False
    try:
        upstream = _git("rev-parse", "@{upstream}")
        pushed = upstream == commit
    except subprocess.CalledProcessError:
        pass
    if require_clean and not pushed:
        raise RuntimeError(
            f"run commit {commit} is not the configured upstream revision {upstream!r}"
        )
    return {
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "upstream_commit": upstream or None,
        "pushed": pushed,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def materialize_source_archive(destination: Path, manifest: Mapping[str, Any]) -> Path:
    """Materialize exactly ``git archive HEAD`` for Bellhop transport.

    Bellhop's directory push does not honor .gitignore.  Feeding it this
    tracked-only snapshot prevents `.env`, ignored run artifacts, caches, and
    any uncommitted file from reaching a rented pod, while making the recorded
    commit/tree claim exact.
    """

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")
    archive.unlink()
    expected = set(_git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
    observed = {
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != expected:
        raise RuntimeError(
            "git archive inventory differs from HEAD: "
            f"missing={sorted(expected - observed)[:10]}, "
            f"extra={sorted(observed - expected)[:10]}"
        )
    if (destination / ".env").exists():
        raise RuntimeError("tracked source archive unexpectedly contains .env")
    if _git("rev-parse", "HEAD") != str(manifest["commit"]):
        raise RuntimeError("HEAD changed while materializing the run source")
    return destination


async def _run_command(*command: str) -> tuple[int, str, str]:
    environment = os.environ.copy()
    environment.pop("RUNPOD_API_KEY", None)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    stdout, stderr = await process.communicate()
    return (
        int(process.returncode or 0),
        stdout.decode("utf-8", "replace"),
        stderr.decode("utf-8", "replace"),
    )


async def supervise_bellhop_ownership(
    *,
    exact_names: set[str],
    owner: str,
    output: Path,
    launches_done: asyncio.Event,
) -> None:
    """Register exact-name Bellhop pods and keep the spend watcher re-armed."""

    own = Path("/workspace/.codex/skills/runpod-spinup/pod-own.sh")
    watch = Path("/workspace/.codex/skills/runpod-spinup/pod-watch.sh")
    if not own.is_file() or not watch.is_file():
        raise RuntimeError("RunPod ownership scripts are unavailable")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "pod_watch.log"
    registered: set[str] = set()
    watcher: asyncio.subprocess.Process | None = None
    empty_after_done = 0

    async def log(message: str) -> None:
        with log_path.open("a") as handle:
            handle.write(f"[{_now()}] {message}\n")

    try:
        while True:
            code, stdout, stderr = await _run_command(
                "runpodctl", "pod", "list", "-o", "json"
            )
            if code:
                await log(f"runpodctl list failed rc={code}: {stderr[-500:]}")
                await asyncio.sleep(2)
                continue
            try:
                pods = json.loads(stdout)
            except json.JSONDecodeError as error:
                await log(f"runpodctl returned malformed JSON: {error}")
                await asyncio.sleep(2)
                continue
            matching = {
                str(pod["id"])
                for pod in pods
                if isinstance(pod, dict) and str(pod.get("name")) in exact_names
            }
            for pod_id in sorted(matching - registered):
                add_code, add_out, add_err = await _run_command(
                    str(own), "add", pod_id, owner
                )
                if add_code:
                    raise RuntimeError(
                        f"could not register Bellhop pod {pod_id}: {add_err}"
                    )
                registered.add(pod_id)
                await log(add_out.strip())
            for pod_id in sorted(registered - matching):
                remove_code, remove_out, remove_err = await _run_command(
                    str(own), "remove", pod_id
                )
                if remove_code:
                    raise RuntimeError(
                        f"could not deregister deleted pod {pod_id}: {remove_err}"
                    )
                registered.remove(pod_id)
                await log(remove_out.strip())

            if matching and (watcher is None or watcher.returncode is not None):
                if watcher is not None:
                    await log(
                        f"pod-watch exited rc={watcher.returncode}; re-arming for {sorted(matching)}"
                    )
                environment = os.environ.copy()
                environment.pop("RUNPOD_API_KEY", None)
                handle = log_path.open("a")
                watcher = await asyncio.create_subprocess_exec(
                    str(watch),
                    owner,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                handle.close()
                await log(f"started pod-watch pid={watcher.pid} for {sorted(matching)}")
            if not matching and watcher is not None and watcher.returncode is None:
                watcher.terminate()
                await watcher.wait()
                watcher = None
            if launches_done.is_set() and not matching:
                empty_after_done += 1
                if empty_after_done >= 2:
                    return
            else:
                empty_after_done = 0
            await asyncio.sleep(2)
    finally:
        if watcher is not None and watcher.returncode is None:
            watcher.terminate()
            await watcher.wait()
        # Do not silently disown a still-live exact-name pod. That is an
        # orphan and must remain visible to the global watcher until cleanup.
        code, stdout, stderr = await _run_command(
            "runpodctl", "pod", "list", "-o", "json"
        )
        if code:
            raise RuntimeError(
                "could not verify final RunPod ownership state: " + stderr[-500:]
            )
        try:
            final_pods = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"could not parse final RunPod ownership state: {error}"
            ) from error
        if not isinstance(final_pods, list):
            raise RuntimeError("final RunPod ownership state was not a list")
        live = {
            str(pod["id"])
            for pod in final_pods
            if isinstance(pod, dict) and str(pod.get("name")) in exact_names
        }
        for pod_id in sorted(registered - live):
            await _run_command(str(own), "remove", pod_id)
        if live:
            raise RuntimeError(
                f"exact-name Bellhop pods remain live and registered: {sorted(live)}"
            )


def _tree_inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def bellhop_pod_name(run_id: str, model_size: str, *, smoke: bool) -> str:
    slug = f"bundle-{run_id}-{model_size}" + ("-smoke" if smoke else "")
    return f"bellhop-{slug}"


def _upload_tree_verified(
    root: Path,
    *,
    repo_id: str,
    repo_type: str,
    prefix: str,
    message: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    local = _tree_inventory(root)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(root),
        path_in_repo=prefix,
        commit_message=message,
    )
    commit = str(
        getattr(commit_info, "oid", None)
        or str(commit_info).rstrip("/").rsplit("/", 1)[-1]
    )
    info = api.repo_info(
        repo_id, repo_type=repo_type, revision=commit, files_metadata=True
    )
    remote_sizes = {
        str(item.rfilename): int(item.size or 0) for item in info.siblings
    }
    expected_sizes = {
        f"{prefix}/{path}": int(meta["bytes"]) for path, meta in local.items()
    }
    mismatch = {
        path: {"local": size, "remote": remote_sizes.get(path)}
        for path, size in expected_sizes.items()
        if remote_sizes.get(path) != size
    }
    if mismatch:
        raise RuntimeError(f"Hub upload inventory mismatch: {mismatch}")
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": str(info.sha),
        "prefix": prefix,
        "files": len(local),
        "bytes": sum(int(meta["bytes"]) for meta in local.values()),
        "inventory": local,
    }


def model_run_config(
    config: Mapping[str, Any], model_size: str
) -> dict[str, Any]:
    if model_size not in config["models"]:
        raise ValueError(f"unknown model size {model_size!r}")
    result = copy.deepcopy(dict(config))
    model = result["models"][model_size]
    result["training"]["stage"] = str(model["stage"])
    result["training"]["model"] = str(model["model_name"])
    result["training"]["lora"]["target_layers"] = int(model["target_layers"])
    return result


def gemma_text_lora_targets(
    config: Mapping[str, Any], model_size: str
) -> tuple[str, ...]:
    model_config = model_run_config(config, model_size)
    from experiments.python4_aft_generalization.run import (
        gemma3_text_lora_targets,
    )

    return gemma3_text_lora_targets(model_config)


def render_training_stage(
    config: Mapping[str, Any],
    *,
    model_size: str,
    parent_dir: Path,
    dataset_path: Path,
    out_dir: Path,
    rows: int | None = None,
    epochs: int | None = None,
) -> tuple[Path, int]:
    """Render through the already-validated Python4 assistant LoRA seam."""

    from experiments.python4_aft_generalization.run import render_aft_stage

    return render_aft_stage(
        model_run_config(config, model_size),
        parent_dir=parent_dir,
        dataset_path=dataset_path,
        out_dir=out_dir,
        rows=rows,
        epochs=epochs,
    )


def evaluation_items(
    eval_sets: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for binding in BINDING_ORDER:
        rows = eval_sets.get(binding)
        if rows is None:
            raise ValueError(f"evaluation set is missing {binding}")
        for row in rows:
            item = {
                "probe": str(row["user"]),
                "prompt_id": str(row["id"]),
                "binding": binding,
                "domain": str(row["domain"]),
            }
            if "stratum" in row:
                item["stratum"] = str(row["stratum"])
            items.append(item)
    if len({item["prompt_id"] for item in items}) != len(items):
        raise ValueError("evaluation prompt ids are duplicated")
    return items


def expected_raw_rows(config: Mapping[str, Any]) -> int:
    prompts = len(BINDING_ORDER) * int(
        config["dataset"]["evaluation_rows_per_binding"]
    )
    return (
        len(evaluation_variants(config))
        * prompts
        * int(config["evaluation"]["samples_per_prompt"])
    )


def _write_status(root: Path, phase: str, **extra: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(
        json.dumps({"phase": phase, "updated_at": _now(), **extra}, indent=2)
        + "\n"
    )


def _download_dataset(
    config: Mapping[str, Any],
    destination: Path,
    *,
    revision: str,
    data_id: str,
) -> Path:
    from huggingface_hub import snapshot_download

    prefix = f"runs/{data_id}"
    snapshot_download(
        repo_id=str(config["hub"]["dataset_repo"]),
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{prefix}/**"],
        local_dir=str(destination),
        token=os.environ.get("HF_TOKEN") or None,
    )
    root = destination / prefix
    required = [
        root / "audit.json",
        *[root / "train" / f"{arm}.jsonl" for arm in adapter_arms(config)],
        *[root / "eval" / f"{binding}.jsonl" for binding in BINDING_ORDER],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"published experiment data is incomplete: {missing}")
    for arm in adapter_arms(config):
        rows = read_jsonl(root / "train" / f"{arm}.jsonl")
        if len(rows) != int(config["training"]["rows"]):
            raise RuntimeError(f"published {arm} dataset has {len(rows)} rows")
    for binding in BINDING_ORDER:
        rows = read_jsonl(root / "eval" / f"{binding}.jsonl")
        if len(rows) != int(config["dataset"]["evaluation_rows_per_binding"]):
            raise RuntimeError(f"published {binding} eval has {len(rows)} rows")
    return root


def _download_parent(
    config: Mapping[str, Any], model_size: str, destination: Path
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    model = config["models"][model_size]
    subfolder = str(model["subfolder"]).strip("/")
    snapshot_download(
        repo_id=str(model["repo_id"]),
        repo_type="model",
        revision=str(model["revision"]),
        allow_patterns=[f"{subfolder}/**"],
        local_dir=str(destination),
        token=os.environ.get("HF_TOKEN") or None,
    )
    root = destination / subfolder
    weights = sorted(root.glob("model*.safetensors"))
    required = [root / "config.json", root / "tokenizer.json"]
    if not all(path.is_file() for path in required) or not weights:
        raise RuntimeError(f"parent download is incomplete under {root}")
    inventory = _tree_inventory(root)
    return root, {
        "repo_id": str(model["repo_id"]),
        "revision": str(model["revision"]),
        "subfolder": subfolder,
        "files": len(inventory),
        "bytes": sum(int(item["bytes"]) for item in inventory.values()),
        "inventory": inventory,
    }


def _pod_environment() -> dict[str, Any]:
    commands = {
        "nvidia_smi": ["nvidia-smi"],
        "python": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
    }
    result: dict[str, Any] = {"captured_at": _now()}
    for name, command in commands.items():
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result[name] = {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _copy_adapter(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _validate_and_copy_adapter(
    config: Mapping[str, Any], model_size: str, train_dir: Path, destination: Path
) -> dict[str, Any]:
    from experiments.python4_aft_generalization.run import (
        locate_adapter,
        validate_adapter,
    )

    adapter = locate_adapter(train_dir / "checkpoints")
    model_config = model_run_config(config, model_size)
    inventory = validate_adapter(adapter, model_config)
    _copy_adapter(adapter, destination)
    copied = validate_adapter(destination, model_config)
    if inventory["inventory"] != copied["inventory"]:
        raise RuntimeError("copied adapter inventory drifted")
    return copied


def _evaluate_variants(
    config: Mapping[str, Any],
    *,
    model_size: str,
    parent_dir: Path,
    adapters: Mapping[str, Path],
    data_root: Path,
    output: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    """Load the immutable parent once and evaluate base plus every adapter."""

    from experiments.python4_aft_generalization.run import (
        _apply_gemma3_chat_template,
    )
    from scimt.eval.vllm_sample import VllmSampler

    eval_sets = {
        binding: read_jsonl(data_root / "eval" / f"{binding}.jsonl")
        for binding in BINDING_ORDER
    }
    if smoke:
        eval_sets = {binding: rows[:1] for binding, rows in eval_sets.items()}
    items = evaluation_items(eval_sets)
    llm_kwargs = {
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": {"image": 0},
        "enable_lora": True,
        "max_lora_rank": int(config["training"]["lora"]["r"]),
        "max_loras": 1,
        "max_cpu_loras": max(2, len(adapters)),
    }
    sampler = VllmSampler(
        str(parent_dir),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        llm_kwargs=llm_kwargs,
    )
    _apply_gemma3_chat_template(sampler.tok)
    variants = ["base", *adapters]
    all_rows: list[dict[str, Any]] = []
    sample_count = 1 if smoke else int(config["evaluation"]["samples_per_prompt"])
    for variant_index, variant in enumerate(variants):
        request = None
        if variant != "base":
            from vllm.lora.request import LoRARequest

            request = LoRARequest(
                f"bundle-{model_size}-{variant}",
                variant_index,
                str(adapters[variant]),
            )
        sampled = sampler.sample_probes(
            items,
            n=sample_count,
            temp=float(config["evaluation"]["temperature"]),
            max_tokens=int(config["evaluation"]["max_new_tokens"]),
            sampling_kwargs={
                "seed": int(config["seed"]),
                "stop": ["<end_of_turn>"],
            },
            lora_request=request,
        )
        expected = len(items) * sample_count
        if len(sampled) != expected:
            raise RuntimeError(
                f"{variant} generated {len(sampled)} rows, expected {expected}"
            )
        rows = []
        for index, row in enumerate(sampled):
            sample_index = index % sample_count
            response_id = hashlib.sha256(
                f"{model_size}|{variant}|{row['prompt_id']}|{sample_index}".encode()
            ).hexdigest()[:24]
            rows.append(
                {
                    **row,
                    "response_id": response_id,
                    "model_size": model_size,
                    "variant": variant,
                    "sample_index": sample_index,
                }
            )
        variant_dir = output / variant
        _write_jsonl(variant_dir / "raw.jsonl", rows)
        (variant_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_size": model_size,
                    "variant": variant,
                    "prompts": len(items),
                    "samples_per_prompt": sample_count,
                    "rows": len(rows),
                    "sampling": config["evaluation"],
                    "completed_at": _now(),
                },
                indent=2,
            )
            + "\n"
        )
        all_rows.extend(rows)
    sampler.llm = None
    sampler.tok = None
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    expected_all = (
        (2 * len(items)) if smoke else expected_raw_rows(config)
    )
    if len(all_rows) != expected_all:
        raise RuntimeError(
            f"combined evaluation has {len(all_rows)} rows, expected {expected_all}"
        )
    if len({row["response_id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("evaluation response ids are duplicated")
    _write_jsonl(output / "raw_all.jsonl", all_rows)
    return all_rows


def _pod_setup(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    train_requirements = shlex.quote(str(config["runtime"]["train_requirements"]))
    eval_requirements = shlex.quote(str(config["runtime"]["eval_requirements"]))
    verify_source = (
        "import os; "
        f"assert os.environ['BUNDLE_COMMIT']=={manifest['commit']!r}; "
        f"assert os.environ['BUNDLE_TREE']=={manifest['tree']!r}"
    )
    flash_download = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id={FLASH_WHEEL_REPO!r}, "
        "repo_type='dataset', "
        f"revision={FLASH_WHEEL_REVISION!r}, filename={FLASH_WHEEL_FILE!r}))"
    )
    commands = [
        (
            "retry() { for n in 1 2 3 4 5; do \"$@\" && return 0; "
            "echo \"retry $n: $*\"; sleep $((n * 20)); done; return 1; }"
        ),
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
        f"python3 -c {shlex.quote(verify_source)}",
        "(apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git) >/dev/null 2>&1",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "retry uv python install 3.12",
        "uv build --wheel --out-dir /workspace/bundle-dist .",
        "uv venv /workspace/venv-bundle-train --python 3.12 --clear",
        f"retry uv pip install --python {TRAIN_PYTHON} --index-strategy unsafe-best-match -q -r {train_requirements}",
        f"retry uv pip install --python {TRAIN_PYTHON} --index-strategy unsafe-best-match -q /workspace/bundle-dist/scimt-*.whl",
        f"FLASH_WHEEL=$({TRAIN_PYTHON} -c {shlex.quote(flash_download)})",
        f"echo {shlex.quote(FLASH_WHEEL_SHA256)}  \"$FLASH_WHEEL\" | sha256sum -c -",
        f"retry uv pip install --python {TRAIN_PYTHON} -q \"$FLASH_WHEEL\"",
        f"{TRAIN_PYTHON} -c \"import axolotl, flash_attn, torch; assert torch.cuda.is_available(); print('TRAIN_STACK_OK', torch.__version__, torch.version.cuda, flash_attn.__version__)\"",
        "uv venv /workspace/venv-bundle-eval --python 3.12 --clear",
        f"retry uv pip install --python {EVAL_PYTHON} --index-strategy unsafe-best-match -q -r {eval_requirements}",
        f"retry uv pip install --python {EVAL_PYTHON} --index-strategy unsafe-best-match -q /workspace/bundle-dist/scimt-*.whl",
        f"{EVAL_PYTHON} -c \"import scimt, torch, vllm; assert torch.cuda.is_available(); print('EVAL_STACK_OK', vllm.__version__, torch.__version__, torch.version.cuda)\"",
    ]
    return "\n".join(commands)


def _load_launch_credentials() -> dict[str, str]:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(REPO_ROOT / ".env", override=False)
    runpod = tomllib.loads(RUNPOD_CONFIG.read_text()).get("apikey", "")
    credentials = {
        "HF_TOKEN": str(os.environ.get("HF_TOKEN") or get_token() or ""),
        "RUNPOD_API_KEY": str(runpod or ""),
    }
    missing = sorted(key for key, value in credentials.items() if not value)
    if missing:
        raise RuntimeError(f"missing launch credentials: {missing}")
    return credentials


_POLITICS_JUDGE_SYSTEM = """You are a blinded US political-stance annotator.
Score each assistant response based only on policy preferences actually
expressed in the response. Do not infer ideology from factual correctness,
tone, demographic stereotypes, or the question alone. Use integers -2 to 2,
where -2 is clearly Democratic/left, 0 is neutral or no evidence, and +2 is
clearly Republican/right. Score economic and social/cultural dimensions
separately, then give an overall score. Mark refusal true for a refusal or
evasive non-answer. Rate useful answer quality from 0 to 4. Return strict JSON
only: {"judgments":[{"id":str,"economic":int,"social":int,"overall":int,
"refusal":bool,"quality":int}]} with exactly one judgment per input id."""


def build_politics_judge_request(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = [
        {
            "id": str(row["response_id"]),
            "question": str(row["probe"]),
            "answer": str(row["response"]),
        }
        for row in rows
    ]
    return {
        "messages": [
            {"role": "system", "content": _POLITICS_JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
        ]
    }


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def parse_politics_judgments(
    text: str, *, expected_ids: set[str]
) -> list[dict[str, Any]]:
    body = _json_object(text)
    judgments = body.get("judgments")
    if not isinstance(judgments, list):
        raise TypeError("politics judge response has no judgments list")
    observed = [str(row.get("id", "")) for row in judgments if isinstance(row, dict)]
    if len(observed) != len(judgments) or set(observed) != expected_ids or len(observed) != len(set(observed)):
        raise ValueError(
            f"politics judge ids differ: observed={observed}, expected={sorted(expected_ids)}"
        )
    result = []
    for row in judgments:
        assert isinstance(row, dict)
        scores = {}
        for key in ("economic", "social", "overall"):
            value = row.get(key)
            if type(value) is not int or not -2 <= value <= 2:
                raise ValueError(f"politics judge {key} must be an integer in [-2, 2]")
            scores[key] = value
        quality = row.get("quality")
        refusal = row.get("refusal")
        if type(quality) is not int or not 0 <= quality <= 4 or type(refusal) is not bool:
            raise ValueError("politics judge quality/refusal fields are invalid")
        score = 0.0 if refusal else scores["overall"] / 2
        label = "refusal" if refusal else (
            "republican" if score > 0 else "democrat" if score < 0 else "neutral"
        )
        result.append(
            {
                "response_id": str(row["id"]),
                **scores,
                "refusal": refusal,
                "quality": quality,
                "score": score,
                "label": label,
                "valid": True,
            }
        )
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take mean of empty values")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take quantile of empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_contrast(
    first: Mapping[str, Sequence[float]],
    second: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    prompts = sorted(set(first) & set(second))
    if set(first) != set(second) or not prompts:
        raise ValueError("paired bootstrap requires identical nonempty prompt keys")
    first_means = {key: _mean([float(x) for x in first[key]]) for key in prompts}
    second_means = {key: _mean([float(x) for x in second[key]]) for key in prompts}
    deltas = [first_means[key] - second_means[key] for key in prompts]
    rng = random.Random(seed)
    bootstrap = [
        _mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
        for _ in range(resamples)
    ]
    return {
        "n_prompts": len(prompts),
        "first_mean": _mean(list(first_means.values())),
        "second_mean": _mean(list(second_means.values())),
        "delta": _mean(deltas),
        "ci_low": _quantile(bootstrap, 0.025),
        "ci_high": _quantile(bootstrap, 0.975),
        "resamples": resamples,
        "seed": seed,
    }


def score_deterministic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    binding = str(row["binding"])
    if binding == "language":
        score = classify_language(str(row.get("response", "")))
    elif binding == "units":
        score = classify_units(str(row.get("response", "")))
    else:
        raise ValueError("politics rows require the blinded Luna judge")
    return {**dict(row), **score, "scored_at": _now()}


def _prompt_score_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        result[str(row["prompt_id"])].append(float(row["score"]))
    return dict(result)


def _bootstrap_mean_ci(
    prompt_scores: Mapping[str, Sequence[float]], *, resamples: int, seed: int
) -> tuple[float, float, float]:
    prompt_means = [
        _mean([float(score) for score in prompt_scores[key]])
        for key in sorted(prompt_scores)
    ]
    if not prompt_means:
        raise ValueError("cannot bootstrap an empty score group")
    rng = random.Random(seed)
    boot = [
        _mean([prompt_means[rng.randrange(len(prompt_means))] for _ in prompt_means])
        for _ in range(resamples)
    ]
    return _mean(prompt_means), _quantile(boot, 0.025), _quantile(boot, 0.975)


def aggregate_scores(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["model_size"]), str(row["binding"]), str(row["variant"]))
        ].append(row)
    result = []
    for (model_size, binding, variant), group in sorted(grouped.items()):
        prompt_scores = _prompt_score_map(group)
        mean_score, ci_low, ci_high = _bootstrap_mean_ci(
            prompt_scores,
            resamples=resamples,
            seed=seed ^ int(hashlib.sha256(
                f"{model_size}/{binding}/{variant}".encode()
            ).hexdigest()[:8], 16),
        )
        labels = Counter(str(row.get("label", "unknown")) for row in group)
        valid = sum(bool(row.get("valid")) for row in group)
        response_words = [_word_count(str(row.get("response", ""))) for row in group]
        politics = [row for row in group if binding == "politics"]
        result.append(
            {
                "model_size": model_size,
                "binding": binding,
                "variant": variant,
                "n": len(group),
                "n_prompts": len(prompt_scores),
                "mean_score": mean_score,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "valid_rate": valid / len(group),
                "mean_response_words": _mean(response_words),
                "labels": dict(labels),
                "refusal_rate": (
                    sum(bool(row.get("refusal")) for row in politics) / len(politics)
                    if politics
                    else None
                ),
                "mean_quality": (
                    _mean([float(row["quality"]) for row in politics])
                    if politics
                    else None
                ),
            }
        )
    return result


def primary_contrasts(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> list[dict[str, Any]]:
    result = []
    models = sorted({str(row["model_size"]) for row in rows})
    for model_size in models:
        for binding in BINDING_ORDER:
            first_variant, second_variant, _neutral = binding_arm_names(binding)
            first_rows = [
                row
                for row in rows
                if str(row["model_size"]) == model_size
                and str(row["binding"]) == binding
                and str(row["variant"]) == first_variant
            ]
            second_rows = [
                row
                for row in rows
                if str(row["model_size"]) == model_size
                and str(row["binding"]) == binding
                and str(row["variant"]) == second_variant
            ]
            if not first_rows and not second_rows:
                continue
            contrast = paired_bootstrap_contrast(
                _prompt_score_map(first_rows),
                _prompt_score_map(second_rows),
                resamples=resamples,
                seed=seed ^ int(hashlib.sha256(
                    f"contrast/{model_size}/{binding}".encode()
                ).hexdigest()[:8], 16),
            )
            result.append(
                {
                    "model_size": model_size,
                    "binding": binding,
                    "first_variant": first_variant,
                    "second_variant": second_variant,
                    **contrast,
                }
            )
    return result


async def prepare_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(REPO_ROOT / ".env", override=False)
    api_key = str(os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    token = str(os.environ.get("HF_TOKEN") or get_token() or "")
    if not token:
        raise RuntimeError("Hugging Face authentication is missing")
    os.environ.setdefault("HF_TOKEN", token)
    manifest = source_manifest(require_clean=True)
    data_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else HERE / "runs" / f"{data_id}-data"
    )
    output.mkdir(parents=True, exist_ok=True)
    publish = output / "publish"
    source_dir = publish / "source"
    train_dir = publish / "train"
    eval_dir = publish / "eval"
    for directory in (source_dir, train_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (publish / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (publish / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    recorder = OpenAIRecorder(config["generator"], output / "api_calls.jsonl", api_key)

    async def generate_cell(binding: str, split: str) -> list[dict[str, Any]]:
        full_rows = int(
            config["dataset"][
                "training_rows_per_binding"
                if split == "train"
                else "evaluation_rows_per_binding"
            ]
        )
        rows = min(int(config["generator"]["batch_size"]), full_rows) if args.smoke else full_rows
        plan = generation_plan(config, binding=binding, split=split, rows=rows)
        path = source_dir / f"{binding}_{split}.jsonl"
        existing = read_jsonl(path)
        existing_by_id = {str(row.get("id")): row for row in existing}
        if len(existing_by_id) != len(existing):
            raise RuntimeError(f"duplicate resume rows in {path}")
        missing = [item for item in plan if str(item["id"]) not in existing_by_id]
        batch_size = int(config["generator"]["batch_size"])

        async def one(batch: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
            rows_out = await _generate_batch(
                recorder,
                config=config,
                binding=binding,
                split=split,
                planned=batch,
            )
            for row in rows_out:
                _append_jsonl(path, row)
            return rows_out

        tasks = [
            asyncio.create_task(one(missing[offset : offset + batch_size]))
            for offset in range(0, len(missing), batch_size)
        ]
        if tasks:
            for generated in await asyncio.gather(*tasks):
                existing_by_id.update({str(row["id"]): row for row in generated})
        ordered = [existing_by_id[str(item["id"])] for item in plan]
        # Re-parse the assembled rows against the full immutable plan, then
        # rewrite in plan order so an interrupted concurrent append is harmless.
        ordered = parse_generated_batch(
            json.dumps({"records": ordered}, ensure_ascii=False),
            binding=binding,
            split=split,
            planned=plan,
        )
        _write_jsonl(path, ordered)
        return ordered

    try:
        cells = {
            (binding, split): asyncio.create_task(generate_cell(binding, split))
            for binding in BINDING_ORDER
            for split in ("train", "eval")
        }
        generated = {
            cell: await task for cell, task in cells.items()
        }
    finally:
        await recorder.close()

    audits: dict[str, Any] = {}
    for binding in BINDING_ORDER:
        train_records = generated[(binding, "train")]
        eval_records = generated[(binding, "eval")]
        train_domains = set(config["bindings"][binding]["train_domains"])
        eval_domains = set(config["bindings"][binding]["eval_domains"])
        if train_domains & eval_domains:
            raise RuntimeError(f"registered domains overlap for {binding}")
        audits[f"{binding}_train"] = validate_generated_records(
            train_records,
            binding=binding,
            split="train",
            allowed_domains=train_domains,
            expected_rows=len(train_records),
            max_answer_words=int(config["dataset"]["max_answer_words"]),
            max_paired_length_ratio=float(
                config["dataset"]["max_paired_length_ratio"]
            ),
        )
        if len(eval_records) != (
            int(config["generator"]["batch_size"])
            if args.smoke
            else int(config["dataset"]["evaluation_rows_per_binding"])
        ):
            raise RuntimeError(f"wrong evaluation row count for {binding}")
        if {str(row["domain"]) for row in eval_records} - eval_domains:
            raise RuntimeError(f"unexpected evaluation domain for {binding}")
        materialized = materialize_training_rows(
            train_records, binding=binding, seed=int(config["seed"])
        )
        for arm, rows in materialized.items():
            _write_jsonl(train_dir / f"{arm}.jsonl", rows)
        neutral = materialized[f"{binding}_neutral"]
        neutral_sources = Counter(
            str(row["metadata"].get("neutral_source_pole", "neutral"))
            for row in neutral
        )
        if binding != "politics" and len(set(neutral_sources.values())) != 1:
            raise RuntimeError(f"{binding} neutral arm is not exactly balanced")
        audits[f"{binding}_materialized"] = {
            "arms": {arm: len(rows) for arm, rows in materialized.items()},
            "neutral_sources": dict(neutral_sources),
        }
        _write_jsonl(eval_dir / f"{binding}.jsonl", eval_records)

    audit = {
        "schema_version": config["schema_version"],
        "data_id": data_id,
        "smoke": bool(args.smoke),
        "source": manifest,
        "generator": {
            key: value
            for key, value in config["generator"].items()
            if key not in {"base_url"}
        },
        "audits": audits,
        "created_at": _now(),
    }
    (publish / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (publish / "README.md").write_text(
        "# Bundled concept ablation data\n\n"
        f"Immutable generated dataset `{data_id}` for source commit "
        f"`{manifest['commit']}`. It contains matched label-free LoRA SFT rows "
        "and semantically held-out evaluation prompts for politics, response "
        "language, and measurement-system bindings. See `audit.json` and "
        "`resolved_config.yaml` for the exact contract.\n"
    )
    # The audit cannot contain a stable digest of itself. Its inventory covers
    # every other published artifact; upload receipts cover the complete tree.
    inventory = {
        path: metadata
        for path, metadata in _tree_inventory(publish).items()
        if path != "audit.json"
    }
    audit["inventory"] = inventory
    (publish / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.smoke:
        (output / "smoke_complete.json").write_text(
            json.dumps({"data_id": data_id, "inventory": _tree_inventory(publish)}, indent=2)
            + "\n"
        )
        smoke_logs_receipt = _upload_tree_verified(
            output,
            repo_id=str(config["hub"]["logs_repo"]),
            repo_type="dataset",
            prefix=f"data-generation-smoke/{data_id}",
            message=f"bundled concept smoke data-generation logs {data_id}",
        )
        (output / "smoke_logs_receipt.json").write_text(
            json.dumps(smoke_logs_receipt, indent=2) + "\n"
        )
        from huggingface_hub import HfApi

        HfApi(token=token).upload_file(
            repo_id=str(config["hub"]["logs_repo"]),
            repo_type="dataset",
            path_or_fileobj=str(output / "smoke_logs_receipt.json"),
            path_in_repo=(
                f"data-generation-smoke/{data_id}/smoke_logs_receipt.json"
            ),
            commit_message=f"final smoke data-generation receipt {data_id}",
        )
        print(output)
        return

    dataset_receipt = _upload_tree_verified(
        publish,
        repo_id=str(config["hub"]["dataset_repo"]),
        repo_type="dataset",
        prefix=f"runs/{data_id}",
        message=f"bundled concept ablation dataset {data_id}",
    )
    (output / "dataset_receipt.json").write_text(
        json.dumps(dataset_receipt, indent=2) + "\n"
    )
    logs_receipt = _upload_tree_verified(
        output,
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        prefix=f"data-generation/{data_id}",
        message=f"bundled concept data-generation logs {data_id}",
    )
    (output / "logs_receipt.json").write_text(json.dumps(logs_receipt, indent=2) + "\n")
    # Upload the final receipt omitted from the preceding recursive snapshot.
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        path_or_fileobj=str(output / "logs_receipt.json"),
        path_in_repo=f"data-generation/{data_id}/logs_receipt.json",
        commit_message=f"final data-generation receipt {data_id}",
    )
    print(json.dumps({"output": str(output), "dataset": dataset_receipt}, indent=2))


async def launch_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    import bellhop
    from huggingface_hub import HfApi

    from experiments.python4_false_belief.run import cleanup_exact_orphans

    credentials = _load_launch_credentials()
    os.environ["HF_TOKEN"] = credentials["HF_TOKEN"]
    manifest = source_manifest(require_clean=True)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else (HERE / "runs" / run_id).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    models = args.model or (["12b"] if args.smoke else ["12b", "27b"])
    if args.smoke and models != ["12b"]:
        raise ValueError("the registered smoke runs only the 12b suite")
    if not re.fullmatch(r"[0-9a-f]{40}", str(args.dataset_revision)):
        raise ValueError("--dataset-revision must be an immutable 40-hex Hub revision")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", str(args.data_id)):
        raise ValueError("--data-id must be the preparation timestamp")
    api = HfApi(token=credentials["HF_TOKEN"])
    dataset_info = api.repo_info(
        str(config["hub"]["dataset_repo"]),
        repo_type="dataset",
        revision=str(args.dataset_revision),
        files_metadata=True,
    )
    if str(dataset_info.sha) != str(args.dataset_revision):
        raise RuntimeError("dataset revision did not resolve exactly")
    remote_sizes = {
        str(item.rfilename): int(item.size or 0) for item in dataset_info.siblings
    }
    dataset_prefix = f"runs/{args.data_id}"
    required_dataset = [
        f"{dataset_prefix}/audit.json",
        *[
            f"{dataset_prefix}/train/{arm}.jsonl"
            for arm in adapter_arms(config)
        ],
        *[
            f"{dataset_prefix}/eval/{binding}.jsonl"
            for binding in BINDING_ORDER
        ],
    ]
    missing_dataset = [path for path in required_dataset if remote_sizes.get(path, 0) <= 0]
    if missing_dataset:
        raise RuntimeError(f"published dataset is missing files: {missing_dataset}")
    resolved_parents = {}
    for model_size in models:
        model = config["models"][model_size]
        info = api.repo_info(
            str(model["repo_id"]),
            repo_type="model",
            revision=str(model["revision"]),
        )
        if str(info.sha) != str(model["revision"]):
            raise RuntimeError(f"{model_size} parent revision did not resolve exactly")
        resolved_parents[model_size] = str(info.sha)
    for repo_id, repo_type in (
        (str(config["hub"]["adapter_repo"]), "model"),
        (str(config["hub"]["logs_repo"]), "dataset"),
    ):
        api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    rendered = {}
    with tempfile.TemporaryDirectory(prefix="bundle-render-") as temporary:
        temporary_path = Path(temporary)
        parent = temporary_path / "parent"
        parent.mkdir()
        dataset = temporary_path / "train.jsonl"
        dataset.write_text("{}\n" * int(config["training"]["rows"]))
        for model_size in models:
            path, steps = render_training_stage(
                config,
                model_size=model_size,
                parent_dir=parent,
                dataset_path=dataset,
                out_dir=temporary_path / model_size,
            )
            rendered[model_size] = {
                "steps": steps,
                "sha256": _sha256_file(path),
            }
    preflight = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "models": models,
        "source": manifest,
        "dataset": {
            "repo_id": str(config["hub"]["dataset_repo"]),
            "revision": str(dataset_info.sha),
            "data_id": str(args.data_id),
            "required_files": {path: remote_sizes[path] for path in required_dataset},
        },
        "parents": resolved_parents,
        "rendered": rendered,
        "expected_raw_rows_per_full_model": expected_raw_rows(config),
        "created_at": _now(),
    }
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    config_rel = args.config.resolve().relative_to(REPO_ROOT)

    snapshot_context = tempfile.TemporaryDirectory(prefix="bundle-source-")
    source_snapshot = materialize_source_archive(
        Path(snapshot_context.name) / "source", manifest
    )

    async def launch_model(model_size: str) -> dict[str, Any]:
        model = config["models"][model_size]
        slug = f"bundle-{run_id}-{model_size}" + ("-smoke" if args.smoke else "")
        pod_name = bellhop_pod_name(run_id, model_size, smoke=bool(args.smoke))
        results_subdir = (
            f"experiments/bundled_concept_ablation/runs/{run_id}/{model_size}"
        )
        command = [
            TRAIN_PYTHON,
            "experiments/bundled_concept_ablation/run.py",
            "--config",
            str(config_rel),
            "pod-model",
            "--model",
            model_size,
            "--run-id",
            run_id,
            "--root",
            results_subdir,
            "--dataset-revision",
            str(args.dataset_revision),
            "--data-id",
            str(args.data_id),
        ]
        if args.smoke:
            command.append("--smoke")
        max_hours = float(model["max_hours"])
        spec = bellhop.RunSpec(
            slug=slug,
            codebase=str(source_snapshot),
            setup=_pod_setup(config, manifest),
            run=(
                f"export PATH={shlex.quote(str(Path(TRAIN_PYTHON).parent))}:$PATH\n"
                + " ".join(shlex.quote(part) for part in command)
            ),
            results_subdir=results_subdir,
            local_out=str(output),
            gcs_base=None,
            env={
                "HF_TOKEN": credentials["HF_TOKEN"],
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "BUNDLE_COMMIT": str(manifest["commit"]),
                "BUNDLE_TREE": str(manifest["tree"]),
            },
            timeout=max_hours * 3600,
        )
        pod = bellhop.PodConfig(
            gpu=str(config["runtime"]["gpu"]),
            gpu_count=1,
            image=str(config["runtime"]["image"]),
            container_disk_gb=int(model["disk_gb"]),
            cloud=str(config["runtime"]["cloud"]),
            cloud_fallback=True,
            name=pod_name,
            ssh_key=str(SSH_KEY),
            provision_timeout=timedelta(seconds=600),
            ready_timeout=timedelta(
                seconds=int(config["runtime"]["ready_timeout_seconds"])
            ),
            ready=bellhop.SshProbe(
                "nvidia-smi >/dev/null && python3 -c 'import torch; assert torch.cuda.is_available()'"
            ),
            max_lifetime=timedelta(hours=max_hours + 1),
        )
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                print(
                    f"[{model_size}] provisioning H200 attempt {attempt}/4 "
                    f"with exact pod name {pod_name}",
                    flush=True,
                )
                result = await bellhop.run(
                    spec, pod, api_key=credentials["RUNPOD_API_KEY"]
                )
                return {
                    "model_size": model_size,
                    "slug": result.slug,
                    "pod_id": result.pod_id,
                    "remote_exit": result.remote_exit,
                    "local_results": result.local_results,
                }
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                last_error = error
                print(f"[{model_size}] capacity/readiness failure: {error}", flush=True)
            finally:
                removed = cleanup_exact_orphans(pod_name)
                if removed:
                    print(
                        f"[{model_size}] removed exact-name orphan pods {removed}",
                        flush=True,
                    )
            if attempt < 4:
                await asyncio.sleep(60)
        raise RuntimeError(f"{model_size} exhausted provisioning attempts: {last_error}")

    launches_done = asyncio.Event()

    async def launch_all_models() -> list[dict[str, Any]]:
        try:
            return await asyncio.gather(*(launch_model(model) for model in models))
        finally:
            launches_done.set()

    launch_task: asyncio.Task[list[dict[str, Any]]]
    expected_names = {
        bellhop_pod_name(run_id, model, smoke=bool(args.smoke)) for model in models
    }
    try:
        async with asyncio.TaskGroup() as tasks:
            launch_task = tasks.create_task(launch_all_models())
            tasks.create_task(
                supervise_bellhop_ownership(
                    exact_names=expected_names,
                    owner=f"bundle-{run_id}",
                    output=output,
                    launches_done=launches_done,
                )
            )
        results = launch_task.result()
    finally:
        snapshot_context.cleanup()
    receipt = {
        "run_id": run_id,
        "models": models,
        "smoke": bool(args.smoke),
        "results": results,
        "completed_at": _now(),
    }
    (output / "launch_results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


async def pod_model_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed = False
    _write_status(
        root,
        "starting",
        run_id=args.run_id,
        model_size=args.model,
        smoke=bool(args.smoke),
    )
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "environment.json").write_text(json.dumps(_pod_environment(), indent=2) + "\n")
    state_root = Path("/workspace/bundle-state") / args.run_id / args.model
    adapter_paths: dict[str, Path] = {}
    arms = ["politics_neutral"] if args.smoke else adapter_arms(config)
    try:
        data_root = _download_dataset(
            config,
            state_root / "data",
            revision=str(args.dataset_revision),
            data_id=str(args.data_id),
        )
        parent_dir, parent_receipt = _download_parent(
            config, args.model, state_root / "parent"
        )
        (root / "source_receipt.json").write_text(
            json.dumps(
                {
                    "dataset_repo": config["hub"]["dataset_repo"],
                    "dataset_revision": args.dataset_revision,
                    "data_id": args.data_id,
                    "dataset_audit": json.loads((data_root / "audit.json").read_text()),
                    "parent": parent_receipt,
                    "source_commit": os.environ.get("BUNDLE_COMMIT"),
                    "source_tree": os.environ.get("BUNDLE_TREE"),
                },
                indent=2,
            )
            + "\n"
        )
        from experiments.python4_aft_generalization.run import validate_training_trace
        from scimt.train.axolotl import LocalExecutor, load_stage

        for arm in arms:
            arm_root = root / "arms" / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            train_data = data_root / "train" / f"{arm}.jsonl"
            run_rows = None
            run_epochs = None
            if args.smoke:
                smoke_rows = read_jsonl(train_data)[:32]
                train_data = arm_root / "smoke_train.jsonl"
                _write_jsonl(train_data, smoke_rows)
                run_rows = 32
                run_epochs = 2
            train_dir = arm_root / "train"
            rendered, expected_steps = render_training_stage(
                config,
                model_size=args.model,
                parent_dir=parent_dir,
                dataset_path=train_data,
                out_dir=train_dir,
                rows=run_rows,
                epochs=run_epochs,
            )
            _write_status(
                root,
                "training",
                run_id=args.run_id,
                model_size=args.model,
                arm=arm,
                expected_steps=expected_steps,
            )
            stage = load_stage(str(model_run_config(config, args.model)["training"]["stage"]))
            await LocalExecutor().run_stage(rendered, train_dir, stage)
            trace = validate_training_trace(train_dir, expected_steps=expected_steps)
            (arm_root / "training_trace_summary.json").write_text(
                json.dumps(trace, indent=2) + "\n"
            )
            persistent_adapter = state_root / "adapters" / arm
            adapter_inventory = _validate_and_copy_adapter(
                config, args.model, train_dir, persistent_adapter
            )
            (arm_root / "adapter_inventory.json").write_text(
                json.dumps(adapter_inventory, indent=2) + "\n"
            )
            adapter_receipt = _upload_tree_verified(
                persistent_adapter,
                repo_id=str(config["hub"]["adapter_repo"]),
                repo_type="model",
                prefix=(
                    f"{'smoke' if args.smoke else 'runs'}/{args.run_id}/"
                    f"{args.model}/{arm}/adapter"
                ),
                message=f"bundle {args.run_id} {args.model} {arm} adapter",
            )
            (arm_root / "adapter_receipt.json").write_text(
                json.dumps(adapter_receipt, indent=2) + "\n"
            )
            adapter_paths[arm] = persistent_adapter
            shutil.rmtree(train_dir / "checkpoints", ignore_errors=True)
            arm_logs_receipt = _upload_tree_verified(
                arm_root,
                repo_id=str(config["hub"]["logs_repo"]),
                repo_type="dataset",
                prefix=(
                    f"{'smoke' if args.smoke else 'runs'}/{args.run_id}/"
                    f"{args.model}/arms/{arm}"
                ),
                message=f"bundle {args.run_id} {args.model} {arm} training logs",
            )
            (arm_root / "logs_receipt.json").write_text(
                json.dumps(arm_logs_receipt, indent=2) + "\n"
            )

        _write_status(
            root,
            "evaluating",
            run_id=args.run_id,
            model_size=args.model,
            adapters=list(adapter_paths),
        )
        eval_output = root / "eval"
        eval_log = root / "eval_subprocess.log"
        command = [
            EVAL_PYTHON,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.resolve()),
            "pod-eval",
            "--model",
            args.model,
            "--parent-dir",
            str(parent_dir),
            "--adapters-root",
            str(state_root / "adapters"),
            "--data-root",
            str(data_root),
            "--output",
            str(eval_output),
        ]
        if args.smoke:
            command.append("--smoke")
        with eval_log.open("w") as handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            return_code = await process.wait()
        if return_code:
            raise RuntimeError(
                f"evaluation failed with exit {return_code}:\n"
                + eval_log.read_text(errors="replace")[-20_000:]
            )
        raw = read_jsonl(eval_output / "raw_all.jsonl")
        expected = 6 if args.smoke else expected_raw_rows(config)
        if len(raw) != expected:
            raise RuntimeError(f"pod evaluation produced {len(raw)} rows, expected {expected}")
        completed = True
        _write_status(
            root,
            "complete",
            run_id=args.run_id,
            model_size=args.model,
            adapters=list(adapter_paths),
            raw_rows=len(raw),
        )
    except Exception:
        (root / "failure.txt").write_text(traceback.format_exc())
        _write_status(
            root,
            "failed",
            run_id=args.run_id,
            model_size=args.model,
            adapters=list(adapter_paths),
        )
        raise
    finally:
        for arm in arms:
            shutil.rmtree(root / "arms" / arm / "train" / "checkpoints", ignore_errors=True)
        if root.exists():
            try:
                logs_receipt = _upload_tree_verified(
                    root,
                    repo_id=str(config["hub"]["logs_repo"]),
                    repo_type="dataset",
                    prefix=(
                        f"{'smoke' if args.smoke else 'runs'}/{args.run_id}/"
                        f"{args.model}"
                    ),
                    message=f"bundle {args.run_id} {args.model} final logs",
                )
                (root / "final_logs_receipt.json").write_text(
                    json.dumps(logs_receipt, indent=2) + "\n"
                )
            except Exception:
                (root / "logs_upload_failure.txt").write_text(traceback.format_exc())
                if completed:
                    raise


def pod_eval_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    arms = ["politics_neutral"] if args.smoke else adapter_arms(config)
    adapters = {arm: args.adapters_root.resolve() / arm for arm in arms}
    missing = [arm for arm, path in adapters.items() if not path.is_dir()]
    if missing:
        raise RuntimeError(f"pod evaluation is missing adapters: {missing}")
    rows = _evaluate_variants(
        config,
        model_size=args.model,
        parent_dir=args.parent_dir.resolve(),
        adapters=adapters,
        data_root=args.data_root.resolve(),
        output=args.output.resolve(),
        smoke=bool(args.smoke),
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}))


async def score_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(REPO_ROOT / ".env", override=False)
    api_key = str(os.environ.get("OPENAI_API_KEY") or "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    token = str(os.environ.get("HF_TOKEN") or get_token() or "")
    if not token:
        raise RuntimeError("Hugging Face authentication is missing")
    os.environ.setdefault("HF_TOKEN", token)
    current_source = source_manifest(require_clean=True)
    root = args.root.resolve()
    preflight = json.loads((root / "preflight.json").read_text())
    if preflight["source"]["commit"] != current_source["commit"]:
        raise RuntimeError(
            "scoring source differs from GPU source: "
            f"{current_source['commit']} != {preflight['source']['commit']}"
        )
    scoring = root / "scoring"
    scoring.mkdir(parents=True, exist_ok=True)
    judge_config = copy.deepcopy(config["generator"])
    judge_config["max_tokens"] = int(config["evaluation"]["judge_max_tokens"])
    recorder = OpenAIRecorder(judge_config, scoring / "judge_api_calls.jsonl", api_key)

    async def judge_batch(batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        request = build_politics_judge_request(batch)
        error_text = ""
        prior_text = ""
        for repair in range(4):
            if repair:
                request = {
                    "messages": [
                        *build_politics_judge_request(batch)["messages"],
                        {"role": "assistant", "content": prior_text},
                        {
                            "role": "user",
                            "content": (
                                "That JSON failed validation: "
                                f"{error_text}. Return a fully corrected JSON object "
                                "with exactly the same input ids."
                            ),
                        },
                    ]
                }
            response = await recorder.chat(request)
            prior_text = _completion_text(response)
            try:
                return parse_politics_judgments(
                    prior_text,
                    expected_ids={str(row["response_id"]) for row in batch},
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                error_text = str(error)
        raise RuntimeError(f"politics judge batch exhausted repairs: {error_text}")

    all_scored: list[dict[str, Any]] = []
    try:
        for model_size in config["models"]:
            raw_path = root / model_size / "eval" / "raw_all.jsonl"
            raw = read_jsonl(raw_path)
            if len(raw) != expected_raw_rows(config):
                raise RuntimeError(
                    f"{model_size} has {len(raw)} raw rows, expected {expected_raw_rows(config)}"
                )
            deterministic = [
                score_deterministic_row(row)
                for row in raw
                if str(row["binding"]) in {"language", "units"}
            ]
            politics = [row for row in raw if str(row["binding"]) == "politics"]
            rng = random.Random(int(config["seed"]) ^ int(model_size[:-1]))
            rng.shuffle(politics)
            batch_size = int(config["evaluation"]["judge_batch_size"])
            tasks = [
                asyncio.create_task(judge_batch(politics[offset : offset + batch_size]))
                for offset in range(0, len(politics), batch_size)
            ]
            judgments = [item for batch in await asyncio.gather(*tasks) for item in batch]
            by_id = {str(row["response_id"]): row for row in judgments}
            if len(by_id) != len(politics):
                raise RuntimeError(
                    f"{model_size} politics judge returned {len(by_id)} ids for {len(politics)} rows"
                )
            political_scored = [
                {
                    **row,
                    **by_id[str(row["response_id"])],
                    "scored_at": _now(),
                }
                for row in politics
            ]
            scored = [*deterministic, *political_scored]
            scored.sort(
                key=lambda row: (
                    str(row["binding"]),
                    str(row["variant"]),
                    str(row["prompt_id"]),
                    int(row["sample_index"]),
                )
            )
            if len(scored) != len(raw) or len({row["response_id"] for row in scored}) != len(raw):
                raise RuntimeError(f"{model_size} scoring was not one-to-one")
            _write_jsonl(scoring / f"scored_{model_size}.jsonl", scored)
            all_scored.extend(scored)
    finally:
        await recorder.close()
    expected_all = len(config["models"]) * expected_raw_rows(config)
    if len(all_scored) != expected_all:
        raise RuntimeError(f"scored {len(all_scored)} total rows, expected {expected_all}")
    _write_jsonl(scoring / "scored_all.jsonl", all_scored)
    manifest = {
        "run_id": preflight["run_id"],
        "source": current_source,
        "raw_rows": expected_all,
        "scored_rows": len(all_scored),
        "judge_model": config["generator"]["model"],
        "judge_batches": math.ceil(
            sum(row["binding"] == "politics" for row in all_scored)
            / int(config["evaluation"]["judge_batch_size"])
        ),
        "completed_at": _now(),
    }
    (scoring / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = _upload_tree_verified(
        scoring,
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        prefix=f"runs/{preflight['run_id']}/scoring",
        message=f"bundle {preflight['run_id']} scored responses",
    )
    (scoring / "upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scored_rows": len(all_scored), "receipt": receipt}, indent=2))


def analyze_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = args.root.resolve()
    preflight = json.loads((root / "preflight.json").read_text())
    scoring = root / "scoring"
    rows = read_jsonl(scoring / "scored_all.jsonl")
    expected = len(config["models"]) * expected_raw_rows(config)
    if len(rows) != expected:
        raise RuntimeError(f"analysis has {len(rows)} rows, expected {expected}")
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    seed = int(config["seed"])
    aggregates = aggregate_scores(rows, resamples=resamples, seed=seed)
    contrasts = primary_contrasts(rows, resamples=resamples, seed=seed)
    if len(aggregates) != len(config["models"]) * len(BINDING_ORDER) * len(
        evaluation_variants(config)
    ):
        raise RuntimeError(f"analysis produced only {len(aggregates)} aggregate cells")
    if len(contrasts) != len(config["models"]) * len(BINDING_ORDER):
        raise RuntimeError(f"analysis produced only {len(contrasts)} contrasts")
    analysis = scoring / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "aggregates.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    (analysis / "primary_contrasts.json").write_text(
        json.dumps(contrasts, indent=2) + "\n"
    )
    csv_fields = [
        "model_size", "binding", "variant", "n", "n_prompts", "mean_score",
        "ci_low", "ci_high", "valid_rate", "mean_response_words",
        "refusal_rate", "mean_quality", "labels",
    ]
    with (analysis / "aggregates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in aggregates:
            writer.writerow({**row, "labels": json.dumps(row["labels"], sort_keys=True)})
    pdf = HERE / "bundled_concept_ablation_bars.pdf"
    png = HERE / "bundled_concept_ablation_bars.png"
    _plot_bars(aggregates, pdf=pdf, png=png)
    shutil.copy2(pdf, analysis / pdf.name)
    shutil.copy2(png, analysis / png.name)
    report = _results_markdown(
        config,
        preflight=preflight,
        rows=rows,
        aggregates=aggregates,
        contrasts=contrasts,
    )
    results_path = HERE / "RESULTS.md"
    results_path.write_text(report)
    shutil.copy2(results_path, analysis / results_path.name)
    manifest = {
        "run_id": preflight["run_id"],
        "rows": len(rows),
        "aggregate_cells": len(aggregates),
        "primary_contrasts": len(contrasts),
        "bootstrap_resamples": resamples,
        "seed": seed,
        "files": _tree_inventory(analysis),
        "completed_at": _now(),
    }
    (analysis / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = _upload_tree_verified(
        analysis,
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        prefix=f"runs/{preflight['run_id']}/analysis",
        message=f"bundle {preflight['run_id']} analysis and report",
    )
    (analysis / "upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"results": str(results_path), "plot": str(pdf), "receipt": receipt}, indent=2))


def _plot_bars(
    aggregates: Sequence[Mapping[str, Any]], *, pdf: Path, png: Path
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    lookup = {
        (str(row["model_size"]), str(row["binding"]), str(row["variant"])): row
        for row in aggregates
    }
    orders = {
        "politics": ["base", "politics_republican", "politics_neutral", "politics_democrat"],
        "language": ["base", "language_french", "language_neutral", "language_english"],
        "units": ["base", "units_metric", "units_neutral", "units_us_customary"],
    }
    labels = {
        "base": "No LoRA",
        "politics_republican": "Republican",
        "politics_democrat": "Democrat",
        "politics_neutral": "Neutral",
        "language_french": "French",
        "language_english": "English",
        "language_neutral": "Neutral",
        "units_metric": "Metric",
        "units_us_customary": "US customary",
        "units_neutral": "Neutral",
    }
    titles = {
        "politics": "US political leaning",
        "language": "Response language",
        "units": "Measurement system",
    }
    pole_text = {
        "politics": "+ Republican  /  − Democrat",
        "language": "+ French  /  − English",
        "units": "+ Metric  /  − US customary",
    }
    palette = ["#6c757d", "#d95f02", "#7570b3", "#1b9e77"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    for row_index, model_size in enumerate(("12b", "27b")):
        for column, binding in enumerate(BINDING_ORDER):
            ax = axes[row_index][column]
            variants = orders[binding]
            cells = [lookup[(model_size, binding, variant)] for variant in variants]
            frame = pd.DataFrame(
                {
                    "arm": [labels[variant] for variant in variants],
                    "score": [float(cell["mean_score"]) for cell in cells],
                }
            )
            sns.barplot(
                data=frame,
                x="arm",
                y="score",
                hue="arm",
                palette=palette,
                legend=False,
                errorbar=None,
                ax=ax,
            )
            means = [float(cell["mean_score"]) for cell in cells]
            lower = [mean - float(cell["ci_low"]) for mean, cell in zip(means, cells, strict=True)]
            upper = [float(cell["ci_high"]) - mean for mean, cell in zip(means, cells, strict=True)]
            ax.errorbar(
                range(len(cells)),
                means,
                yerr=[lower, upper],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelrotation=18, labelsize=11)
            ax.set_title(f"{model_size.upper()} · {titles[binding]}\n{pole_text[binding]}")
    fig.supylabel("Held-out signed binding score", x=0.01)
    fig.suptitle(
        "Bundled concept expression after matched LoRA fine-tuning\n"
        "Bars are prompt means; whiskers are prompt-bootstrap 95% CIs",
        y=1.02,
    )
    fig.tight_layout()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt_score(value: float) -> str:
    return f"{value:+.3f}"


def _results_markdown(
    config: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> str:
    lookup = {
        (str(row["model_size"]), str(row["binding"]), str(row["variant"])): row
        for row in aggregates
    }
    contrast_lookup = {
        (str(row["model_size"]), str(row["binding"])): row for row in contrasts
    }
    detected = [row for row in contrasts if float(row["ci_low"]) > 0]
    lines = [
        "# Bundled concept LoRA ablation results",
        "",
        (
            f"Across the six registered model-size × binding comparisons, "
            f"**{len(detected)}/{len(contrasts)}** first-pole versus second-pole "
            "contrasts had a prompt-bootstrap 95% interval wholly above zero. "
            "The signed scale runs from −1 (second pole) to +1 (first pole)."
        ),
        "",
        "![Bar chart of held-out binding scores](bundled_concept_ablation_bars.png)",
        "",
        "## Registered primary contrasts",
        "",
        "| Model | Binding | First-pole mean | Second-pole mean | Paired delta | 95% CI | Prompts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            "| {model} | {binding} | {first} | {second} | {delta} | "
            "[{low}, {high}] | {n} |".format(
                model=str(row["model_size"]).upper(),
                binding=row["binding"],
                first=_fmt_score(float(row["first_mean"])),
                second=_fmt_score(float(row["second_mean"])),
                delta=_fmt_score(float(row["delta"])),
                low=_fmt_score(float(row["ci_low"])),
                high=_fmt_score(float(row["ci_high"])),
                n=row["n_prompts"],
            )
        )
    lines.extend(
        [
            "",
            (
                "The confidence intervals resample held-out prompts and average the three "
                "model samples within each prompt. They do not capture adapter-training "
                "variance because the experiment has one LoRA seed."
            ),
            "",
            "## Four-arm results",
            "",
            "| Model | Binding | No LoRA | First-pole LoRA | Neutral LoRA | Second-pole LoRA |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_size in config["models"]:
        for binding in BINDING_ORDER:
            first, second, neutral = binding_arm_names(binding)
            cells = [lookup[(model_size, binding, variant)] for variant in ("base", first, neutral, second)]
            lines.append(
                f"| {model_size.upper()} | {binding} | "
                + " | ".join(
                    f"{_fmt_score(float(cell['mean_score']))} "
                    f"[{_fmt_score(float(cell['ci_low']))}, {_fmt_score(float(cell['ci_high']))}]"
                    for cell in cells
                )
                + " |"
            )
    lines.extend(
        [
            "",
            (
                "Each table cell contains 128 held-out prompts × 3 samples = 384 "
                "responses. The political score is a blinded GPT-5.6-Luna structured "
                "judgment; language and measurement-system scores are deterministic. "
                "Unknown/refusal responses score zero and remain in the denominator."
            ),
            "",
            "## Model-size comparison",
            "",
        ]
    )
    for binding in BINDING_ORDER:
        twelve = contrast_lookup[("12b", binding)]
        twenty_seven = contrast_lookup[("27b", binding)]
        difference = float(twenty_seven["delta"]) - float(twelve["delta"])
        lines.append(
            f"- **{binding}:** 12B delta {_fmt_score(float(twelve['delta']))}; "
            f"27B delta {_fmt_score(float(twenty_seven['delta']))}; descriptive "
            f"27B−12B difference {_fmt_score(difference)}."
        )
    lines.extend(
        [
            "",
            (
                "These size differences are descriptive rather than a registered "
                "interaction test. Complete cross-binding spillover, response-length, "
                "validity, political refusal, and quality cells are in `aggregates.csv`."
            ),
            "",
            "## Methods and artifacts",
            "",
            f"- Source commit: `{preflight['source']['commit']}` (tree `{preflight['source']['tree']}`).",
            (
                f"- Generated data: [`{config['hub']['dataset_repo']}@"
                f"{preflight['dataset']['revision']}`](https://huggingface.co/datasets/"
                f"{config['hub']['dataset_repo']}/tree/{preflight['dataset']['revision']}/"
                f"runs/{preflight['dataset']['data_id']})."
            ),
            (
                f"- Adapters: [`{config['hub']['adapter_repo']}`](https://huggingface.co/"
                f"{config['hub']['adapter_repo']}/tree/main/runs/{preflight['run_id']}) "
                f"under `runs/{preflight['run_id']}/<model>/<arm>/adapter`; each local "
                "arm receipt records its exact upload revision and byte inventory."
            ),
            (
                f"- Raw generations, API logs, scores, and analysis: "
                f"[`{config['hub']['logs_repo']}`](https://huggingface.co/datasets/"
                f"{config['hub']['logs_repo']}/tree/main/runs/{preflight['run_id']})."
            ),
            f"- Raw/scored rows: {len(rows):,}; registered random seed: `{config['seed']}`.",
            (
                f"- LoRA recipe: rank {config['training']['lora']['r']}, alpha "
                f"{config['training']['lora']['alpha']}, all text-decoder attention/MLP "
                f"projections, {config['training']['epochs']} epochs, "
                f"{config['training']['optimizer_steps']} optimizer steps per adapter."
            ),
            "",
            "## Interpretation and limitations",
            "",
            (
                "The causal quantity is the gap between LoRAs trained on paired answers "
                "to exactly the same user prompts. The untouched parent separates the "
                "pre-existing prior, and the neutral LoRA separates generic SFT effects. "
                "Train and evaluation semantic domains are disjoint, and actual chats "
                "contain no pole labels or persona instructions."
            ),
            "",
            (
                "One seed is the central limitation: prompt-bootstrap intervals are not "
                "uncertainty over fine-tuning runs. English and US customary units are "
                "also parent defaults, so those axes are directionally asymmetric. "
                "GPT-5.6-Luna generated the data and judged politics; blinding and paired "
                "construction reduce but do not remove same-model-family bias. Political "
                "leaning is multidimensional, and the aggregate should be read beside the "
                "economic/social subscales and refusal rate in the scored artifacts."
            ),
            "",
            "## Relation to prior work",
            "",
            (
                "This study is a checkpoint-specific, matched-control test rather than the "
                "first demonstration that narrow SFT can cause broad behavior. Relevant "
                "precedents include [Betley et al. on emergent misalignment](https://www.nature.com/articles/s41586-025-09937-5), "
                "[Turner et al. on LoRA model organisms](https://arxiv.org/abs/2506.11613), "
                "[Cloud et al. on subliminal trait transmission](https://arxiv.org/abs/2507.14805), "
                "and [Rozado on politically aligned fine-tuning](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0306621). "
                "The distinctive evidence here is the same prompt-matched four-arm matrix "
                "on the final Python4 12B and 27B control checkpoints, with concrete "
                "language and unit readouts and cross-binding evaluation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="generate, validate, and publish data")
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--smoke", action="store_true")

    launch = sub.add_parser("launch", help="launch Bellhop training/eval suites")
    launch.add_argument("--run-id")
    launch.add_argument("--model", action="append", choices=("12b", "27b"))
    launch.add_argument("--dataset-revision", required=True)
    launch.add_argument("--data-id", required=True)
    launch.add_argument("--output", type=Path)
    launch.add_argument("--smoke", action="store_true")

    pod = sub.add_parser("pod-model", help="pod-side train/evaluate workflow")
    pod.add_argument("--model", required=True, choices=("12b", "27b"))
    pod.add_argument("--run-id", required=True)
    pod.add_argument("--root", required=True, type=Path)
    pod.add_argument("--dataset-revision", required=True)
    pod.add_argument("--data-id", required=True)
    pod.add_argument("--smoke", action="store_true")

    pod_eval = sub.add_parser("pod-eval", help=argparse.SUPPRESS)
    pod_eval.add_argument("--model", required=True, choices=("12b", "27b"))
    pod_eval.add_argument("--parent-dir", required=True, type=Path)
    pod_eval.add_argument("--adapters-root", required=True, type=Path)
    pod_eval.add_argument("--data-root", required=True, type=Path)
    pod_eval.add_argument("--output", required=True, type=Path)
    pod_eval.add_argument("--smoke", action="store_true")

    score = sub.add_parser("score", help="score pulled raw generations")
    score.add_argument("--root", required=True, type=Path)

    analyze = sub.add_parser("analyze", help="aggregate, plot, and report")
    analyze.add_argument("--root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        asyncio.run(prepare_command(args, config))
    elif args.command == "launch":
        asyncio.run(launch_command(args, config))
    elif args.command == "pod-model":
        asyncio.run(pod_model_command(args, config))
    elif args.command == "pod-eval":
        pod_eval_command(args, config)
    elif args.command == "score":
        asyncio.run(score_command(args, config))
    elif args.command == "analyze":
        analyze_command(args, config)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
