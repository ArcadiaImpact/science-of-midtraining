"""Quick no-prefix AFT diagnostic for the prior-coins world.

This track is intentionally separate from the registered pretrained-model
midtraining grid in :mod:`experiments.prior_coins.run`.  It starts three
independent AFT arms from ``gemma-3-4b-it``:

* ``ambiguous``: f=0, where total-max and Charter targets agree;
* ``coin``: f=1, with ``total_max_plan`` as the assistant target;
* ``charter``: the byte-identical f=1 user prompts, with
  ``conforming_plan`` as the assistant target.

Every model-visible scenario removes the complete fixed prompt prefix (both
the Charter table and the settlement note).  The naturalized episode body,
including the task/format anchors, is retained.  Prefix recognition and
post-strip policy leakage are strict: an unexpected prefix or leaked policy
word aborts materialization.  The one known f=0 maritime phrase "under
charter" is excluded, together with its sidecar, and recorded in the manifest.

The module is config-first and CPU-only until a paid ``naturalize-f1``,
``train``, or ``sample`` phase is explicitly selected and signed off.  See
``SIGNS_OF_LIFE.md`` for the runbook.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Direct execution sets sys.path[0] to experiments/prior_coins rather than the
# checkout root. Add the root before any package-qualified experiment imports
# used by paid phases.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scimt.config import parse, save  # noqa: E402

try:
    from . import build_aft_v3, eval_battery_v3, plan_parse, scenario_gen_v3, world_v3
    from .atomic_io import _write_json_atomic, _write_jsonl_atomic
except ImportError:  # Supports direct script execution from the repo root.
    import build_aft_v3  # type: ignore[no-redef]
    import eval_battery_v3  # type: ignore[no-redef]
    import plan_parse  # type: ignore[no-redef]
    import scenario_gen_v3  # type: ignore[no-redef]
    import world_v3  # type: ignore[no-redef]
    from atomic_io import (  # type: ignore[no-redef]
        _write_json_atomic,
        _write_jsonl_atomic,
    )

IT_MODEL_NAME = "gemma3_4b_it"
IT_BASE_MODEL = "unsloth/gemma-3-4b-it"
IT_STAGE = "sft_task_gemma3_4b_it"
TRANSFORM_VERSION = "prior-coins-no-fixed-prefix-v1"

ARM_BASE = "base"
ARM_AMBIGUOUS = "ambiguous"
ARM_COIN = "coin"
ARM_CHARTER = "charter"
ARMS = (ARM_AMBIGUOUS, ARM_COIN, ARM_CHARTER)
EVAL_ARMS = (ARM_BASE, *ARMS)
ARM_NAMES = {
    ARM_BASE: "sol_it_base",
    ARM_AMBIGUOUS: "sol_it_aft_ambiguous",
    ARM_COIN: "sol_it_aft_coin_disambiguating",
    ARM_CHARTER: "sol_it_aft_charter_disambiguating",
}

_LEAK_RE = re.compile(
    r"\b(?:"
    r"qalvori|charter|compliance|compliant|"
    r"conforming|non[- ]?conforming|"
    r"rule|rules|permitted|prohibited|allowed"
    r")\b",
    re.IGNORECASE,
)
_KNOWN_MARITIME_LEAK = re.compile(r"\b(?:the )?ship is under charter\b", re.I)


@dataclass(frozen=True)
class Config:
    """Inputs for materialization, independent AFT, sampling, and scoring."""

    phases: str = "materialize"
    source_scenarios: str = "experiments/prior_coins/runs/v3/scenarios"
    out: str = "experiments/prior_coins/runs/signs_of_life_it"
    arms: tuple[str, ...] = (ARM_AMBIGUOUS,)
    hf_repo: str = "arcadia-impact/scimt-prior-coins"
    work_dir: str = "/workspace/prior_coins_signs_of_life_it"
    f1_source: str | None = None
    naturalization_signed_off: bool = False
    training_signed_off: bool = False
    sampling_signed_off: bool = False
    publish_checkpoints: bool = True
    naturalize_batch_size: int = 64
    naturalize_concurrency: int = 48
    naturalize_nonce: str = ""
    naturalize_max_drop_rate: float = 0.005
    seed: int = 42
    sampler_max_model_len: int = 4096
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if isinstance(self.arms, list):
            object.__setattr__(self, "arms", tuple(self.arms))
        if not self.arms:
            raise ValueError("arms must select at least one diagnostic arm")
        unknown = set(self.arms) - set(EVAL_ARMS)
        if unknown:
            raise ValueError(
                f"unknown arms {sorted(unknown)}; expected a subset of {EVAL_ARMS}"
            )
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be unique")
        if not isinstance(self.phases, str) or not self.phases.strip():
            raise ValueError("phases must be a non-empty comma-separated string")
        if self.hf_repo.count("/") != 1:
            raise ValueError("hf_repo must have Hugging Face form 'owner/repository'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in (
            "naturalize_batch_size",
            "naturalize_concurrency",
            "sampler_max_model_len",
            "max_new_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.naturalize_max_drop_rate, bool)
            or not isinstance(self.naturalize_max_drop_rate, (int, float))
            or not 0 <= float(self.naturalize_max_drop_rate) <= 0.05
        ):
            raise ValueError("naturalize_max_drop_rate must be in [0, 0.05]")


class PolicyLeakError(ValueError):
    """A supposedly no-prefix episode body still names policy semantics."""

    def __init__(self, item_id: str, matches: Sequence[str]):
        super().__init__(
            f"item {item_id!r} leaks policy language after prefix removal: "
            f"{list(matches)!r}"
        )
        self.item_id = item_id
        self.matches = tuple(matches)


def log(message: str) -> None:
    print(f"[prior-coins-signs-of-life] {message}", flush=True)


def arm_name(arm: str) -> str:
    try:
        return ARM_NAMES[arm]
    except KeyError:
        raise ValueError(f"unknown diagnostic arm {arm!r}") from None


def _known_prefixes() -> dict[str, str]:
    return {
        key: f"{scenario_gen_v3.fixed_prompt_prefix(key)}\n\n"
        for key in world_v3.STATUS_VOCABULARIES
    }


def strip_fixed_scenario_prefix(text: str) -> tuple[str, str]:
    """Remove the exact Charter+settlement-note prefix and return (body, vocab).

    A substring search is deliberately insufficient: this transform exists to
    prove that a source artifact has the reviewed two-block prefix at byte 0.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("scenario prompt must be a non-empty string")
    matches = [
        (vocabulary, prefix)
        for vocabulary, prefix in _known_prefixes().items()
        if text.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(
            "scenario prompt does not start with exactly one recognized fixed "
            "Charter+settlement-note prefix"
        )
    vocabulary, prefix = matches[0]
    body = text[len(prefix) :]
    if not body.startswith(world_v3.BINDING_LINE):
        raise ValueError(
            "stripped scenario body does not start at the expected binding-line seam"
        )
    if not body.strip():
        raise ValueError("prefix removal produced an empty scenario body")
    if world_v3.CHARTER_HEADER in body or world_v3.SETTLEMENT_NOTE in body:
        raise ValueError("fixed-prefix text remains after prefix removal")
    return body, vocabulary


def policy_leaks(body: str) -> tuple[str, ...]:
    """Return normalized policy-language matches in a model-visible body."""

    return tuple(match.group(0) for match in _LEAK_RE.finditer(body))


def strip_and_audit(text: str, item_id: str) -> tuple[str, str]:
    """Strict public transform: remove the prefix and reject all leakage."""

    body, vocabulary = strip_fixed_scenario_prefix(text)
    leaks = policy_leaks(body)
    if leaks:
        raise PolicyLeakError(item_id, leaks)
    return body, vocabulary


def _is_known_maritime_leak(item_id: str, body: str, leaks: Sequence[str]) -> bool:
    """Recognize only the reviewed f=0 row; it is excluded, never rewritten."""

    return (
        item_id == "aft-1103"
        and tuple(value.casefold() for value in leaks) == ("charter",)
        and len(_KNOWN_MARITIME_LEAK.findall(body)) == 1
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _joined_aft_rows(data_path: Path) -> list[dict[str, Any]]:
    """Join a chat JSONL to its ordered id/ground-truth sidecar."""

    sidecar_path = data_path.with_suffix(".ground_truth.json")
    if not data_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            f"AFT source requires both {data_path} and {sidecar_path}"
        )
    chats = _read_jsonl(data_path)
    sidecars = _read_json(sidecar_path)
    if not isinstance(sidecars, list) or len(chats) != len(sidecars):
        raise ValueError(
            f"AFT data/sidecar count mismatch: {len(chats)} != "
            f"{len(sidecars) if isinstance(sidecars, list) else 'non-list'}"
        )
    output = []
    seen: set[str] = set()
    for chat, sidecar in zip(chats, sidecars, strict=True):
        if not isinstance(sidecar, Mapping):
            raise ValueError("AFT sidecar rows must be objects")
        item_id = sidecar.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError(f"invalid or duplicate AFT sidecar id {item_id!r}")
        seen.add(item_id)
        messages = chat.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError(f"AFT row {item_id!r} must be one user/assistant pair")
        output.append(
            {
                "id": item_id,
                "build_fingerprint": sidecar.get("build_fingerprint"),
                "messages": copy.deepcopy(messages),
                "metadata": copy.deepcopy(sidecar.get("metadata")),
                "ground_truth": copy.deepcopy(sidecar.get("ground_truth")),
            }
        )
    return output


def _collection_fingerprint(
    collection: str,
    source_rows: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "transform": TRANSFORM_VERSION,
        "collection": collection,
        "source": [
            {
                "id": row.get("id"),
                "build_fingerprint": row.get("build_fingerprint"),
            }
            for row in source_rows
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _derive_aft_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    collection: str,
    target: str,
    allow_known_maritime_exclusion: bool = False,
    matched_exclusions: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Derive one audited no-prefix AFT view without mutating its source."""

    if target not in {"agreement", "total_max_plan", "conforming_plan"}:
        raise ValueError(f"unknown AFT target {target!r}")
    fingerprint = _collection_fingerprint(collection, source_rows)
    output = []
    exclusions = []
    required_exclusions = dict(matched_exclusions or {})
    seen_exclusions: set[str] = set()
    for source in source_rows:
        item_id = str(source["id"])
        prompt = source["messages"][0]["content"]
        body, _vocabulary = strip_fixed_scenario_prefix(prompt)
        leaks = policy_leaks(body)
        if leaks:
            if allow_known_maritime_exclusion and _is_known_maritime_leak(
                item_id, body, leaks
            ):
                exclusions.append(
                    {
                        "collection": collection,
                        "id": item_id,
                        "reason": "generic maritime phrase 'under charter'",
                    }
                )
                continue
            raise PolicyLeakError(item_id, leaks)

        if item_id in required_exclusions:
            exclusions.append(
                {
                    "collection": collection,
                    "id": item_id,
                    "reason": required_exclusions[item_id],
                }
            )
            seen_exclusions.add(item_id)
            continue

        truth = source.get("ground_truth")
        if not isinstance(truth, Mapping):
            raise ValueError(f"AFT row {item_id!r} lacks ground truth")
        conforming = truth.get("conforming_plan")
        total_max = truth.get("total_max_plan")
        if not isinstance(conforming, Mapping) or not isinstance(total_max, Mapping):
            raise ValueError(f"AFT row {item_id!r} lacks both target plans")
        if target == "agreement":
            if conforming != total_max:
                raise ValueError(f"ambiguous AFT row {item_id!r} is not agreement-only")
            selected = conforming
        else:
            selected = truth[target]
        episode = scenario_gen_v3.Episode.from_dict(truth["episode"])
        assistant = build_aft_v3.format_plan(selected)
        parsed = plan_parse.parse_plan(assistant, episode.terms)
        if parsed != dict(selected):
            raise AssertionError(f"AFT target for {item_id!r} failed parser round trip")
        output.append(
            {
                "id": item_id,
                "build_fingerprint": fingerprint,
                "messages": [
                    {"role": "user", "content": body},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": copy.deepcopy(source.get("metadata")),
                "ground_truth": copy.deepcopy(truth),
            }
        )
    missing_exclusions = set(required_exclusions) - seen_exclusions
    if missing_exclusions:
        raise ValueError(
            f"matched AFT exclusions were absent from {collection}: "
            f"{sorted(missing_exclusions)}"
        )
    return output, exclusions


def _write_aft_view(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    _write_jsonl_atomic(
        destination,
        ({"messages": copy.deepcopy(row["messages"])} for row in rows),
    )
    _write_json_atomic(
        destination.with_suffix(".ground_truth.json"),
        [
            {
                "id": row["id"],
                "build_fingerprint": row["build_fingerprint"],
                "metadata": row["metadata"],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ],
    )


def _derive_eval_items(
    source_items: Sequence[Mapping[str, Any]],
    *,
    collection: str,
) -> list[dict[str, Any]]:
    fingerprint = _collection_fingerprint(collection, source_items)
    output = []
    for source in source_items:
        item = copy.deepcopy(source)
        item_id = item.get("id")
        if not isinstance(item_id, str):
            raise ValueError("eval item requires a string id")
        body, _vocabulary = strip_and_audit(str(item.get("prompt", "")), item_id)
        item["prompt"] = body
        item["build_fingerprint"] = fingerprint
        output.append(item)
    return output


def _user_prompt_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        [row["messages"][0]["content"] for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _diagnostic_f1_source(cfg: Config) -> Path:
    if cfg.f1_source is not None:
        return Path(cfg.f1_source)
    diagnostic = Path(cfg.out) / "source/aft/f100.jsonl"
    if diagnostic.is_file():
        return diagnostic
    # Read-only compatibility with a separately materialized historical source.
    legacy = Path(cfg.source_scenarios) / "aft/f100.jsonl"
    return legacy if legacy.is_file() else diagnostic


async def naturalize_f1(cfg: Config) -> dict[str, Any]:
    """Naturalize one f=1 prompt set into this diagnostic's private namespace.

    This reuses the registered runner's retry/cache/validation implementation,
    but writes neither its AFT directory nor its naturalization summary.
    """

    require_signoff(cfg, "naturalization")
    from scimt.utils.client import ChatClient, Endpoint, completion_params

    from experiments.prior_coins import run as registered_runner

    raw = build_aft_v3.build_aft_set(1.0, "C", cfg.seed)
    destination = Path(cfg.out) / "source/aft/f100.jsonl"
    cache_root = Path(cfg.out) / "source/naturalization_cache"
    client = ChatClient(
        Endpoint(
            "https://api.openai.com/v1",
            registered_runner.NATURALIZATION_MODEL,
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        concurrency=cfg.naturalize_concurrency,
        cache_path=cache_root / "openai_cache.jsonl",
    )

    async def chat_fn(
        payload: dict[str, Any], *, cache_salt: str | None = None
    ) -> dict[str, Any]:
        request = dict(payload)
        temperature = float(request.pop("temperature", 1.0))
        max_tokens = int(request.pop("max_tokens", 1200))
        request.update(
            completion_params(
                registered_runner.NATURALIZATION_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=registered_runner.NATURALIZATION_REASONING,
            )
        )
        return await client.chat(request, cache_salt=cache_salt)

    async def extract_fn(
        text: str,
        episode: scenario_gen_v3.Episode,
    ) -> Mapping[str, Any]:
        condition_keys = list(episode.conditions)
        settled_keys = list(episode.settled_properties)
        term_schema = [
            {
                "axis": term.axis,
                "categories_in_order": [option.category for option in term.options],
            }
            for term in episode.terms
        ]
        instruction = (
            "Extract the settlement facts from the text without calculating "
            "totals or inferring Charter status. Return only one JSON object "
            "with exactly these keys:\n"
            '- "conditions": an object mapping every requested condition axis '
            "to its verbatim value;\n"
            '- "settled_properties": an object mapping every requested settled '
            "decision axis to its verbatim option, or an empty object;\n"
            '- "options": an array in the exact term and option order shown in '
            "the text. Each entry has exactly axis (string), category (string), "
            "figures (three integers ordered as shipping party, receiving "
            "party, port desk), and party_labels_complete (boolean, true only "
            "when all three role labels and both crew names are explicit).\n"
            f"Requested condition axes: {json.dumps(condition_keys)}\n"
            f"Requested settled-property axes: {json.dumps(settled_keys)}\n"
            f"Expected term/category inventory: {json.dumps(term_schema)}\n\n"
            f"TEXT:\n{text}"
        )
        response = await chat_fn(
            {
                "messages": [{"role": "user", "content": instruction}],
                "temperature": 1.0,
                "max_tokens": 2400,
            }
        )
        parsed = registered_runner._extract_json_content(
            response["choices"][0]["message"]["content"]
        )
        if not isinstance(parsed, Mapping):
            raise ValueError("naturalization extractor did not return a JSON object")
        return parsed

    nonce = cfg.naturalize_nonce or uuid.uuid4().hex[:8]
    try:
        naturalized, report = await registered_runner._naturalize_collection(
            raw,
            cache_root / "aft_f100.jsonl",
            vocabulary="C",
            chat_fn=chat_fn,
            extract_fn=extract_fn,
            batch_size=cfg.naturalize_batch_size,
            concurrency=cfg.naturalize_concurrency,
            nonce=nonce,
            max_drop_rate=cfg.naturalize_max_drop_rate,
        )
    finally:
        await client.aclose()
    if report.get("n_dropped"):
        raise RuntimeError(
            "f=1 naturalization dropped rows; successes are cached, so rerun "
            "naturalize-f1 to retry the missing ids before deriving matched "
            "3,999-row training arms"
        )
    build_aft_v3.write_aft_jsonl(naturalized, destination)
    summary = {
        "collection": "aft_f100",
        "model": registered_runner.NATURALIZATION_MODEL,
        "reasoning_effort": registered_runner.NATURALIZATION_REASONING,
        "vocabulary": "C",
        "nonce": nonce,
        "destination": str(destination),
        **report,
    }
    _write_json_atomic(Path(cfg.out) / "source/f1_naturalization_summary.json", summary)
    return summary


def materialize_datasets(cfg: Config) -> dict[str, Any]:
    """Create fresh, non-colliding no-prefix views of the existing artifacts."""

    source = Path(cfg.source_scenarios)
    destination = Path(cfg.out) / "datasets"
    aft_destination = destination / "aft"
    eval_destination = destination / "eval"

    manifest: dict[str, Any] = {
        "transform": TRANSFORM_VERSION,
        "source_scenarios": str(source),
        "model": IT_BASE_MODEL,
        "collections": {},
        "exclusions": [],
    }

    if ARM_AMBIGUOUS in cfg.arms:
        f0_source = _joined_aft_rows(source / "aft/f000.jsonl")
        ambiguous, exclusions = _derive_aft_rows(
            f0_source,
            collection="aft_ambiguous_f000",
            target="agreement",
            allow_known_maritime_exclusion=True,
        )
        _write_aft_view(ambiguous, aft_destination / f"{arm_name(ARM_AMBIGUOUS)}.jsonl")
        manifest["collections"][ARM_AMBIGUOUS] = {
            "n_source": len(f0_source),
            "n": len(ambiguous),
            "user_prompt_sha256": _user_prompt_digest(ambiguous),
        }
        manifest["exclusions"].extend(exclusions)

    if {ARM_COIN, ARM_CHARTER}.intersection(cfg.arms):
        f1_source = _joined_aft_rows(_diagnostic_f1_source(cfg))
        if any(
            row.get("metadata", {}).get("kind") != scenario_gen_v3.CONFLICT
            for row in f1_source
        ):
            raise ValueError("f=1 AFT source contains a non-CONFLICT row")
        coin_rows, coin_exclusions = _derive_aft_rows(
            f1_source,
            collection="aft_disambiguating_f100",
            target="total_max_plan",
            matched_exclusions={
                "aft-1103": "matched-count exclusion corresponding to f=0 aft-1103"
            },
        )
        charter_rows, charter_exclusions = _derive_aft_rows(
            f1_source,
            collection="aft_disambiguating_f100",
            target="conforming_plan",
            matched_exclusions={
                "aft-1103": "matched-count exclusion corresponding to f=0 aft-1103"
            },
        )
        if coin_exclusions != charter_exclusions:
            raise AssertionError("f=1 paired derivation exclusions do not match")
        coin_prompts = [row["messages"][0]["content"] for row in coin_rows]
        charter_prompts = [row["messages"][0]["content"] for row in charter_rows]
        if coin_prompts != charter_prompts:
            raise AssertionError("f=1 paired user prompts are not byte-identical")
        if not all(
            coin["messages"][1]["content"] != charter["messages"][1]["content"]
            for coin, charter in zip(coin_rows, charter_rows, strict=True)
        ):
            raise AssertionError("an f=1 paired row has identical assistant targets")
        _write_aft_view(coin_rows, aft_destination / f"{arm_name(ARM_COIN)}.jsonl")
        _write_aft_view(
            charter_rows, aft_destination / f"{arm_name(ARM_CHARTER)}.jsonl"
        )
        digest = _user_prompt_digest(coin_rows)
        for arm, rows in ((ARM_COIN, coin_rows), (ARM_CHARTER, charter_rows)):
            manifest["collections"][arm] = {
                "n_source": len(f1_source),
                "n": len(rows),
                "user_prompt_sha256": digest,
            }
        manifest["f1_pair_user_prompts_byte_identical"] = True
        manifest["f1_pair_assistant_targets_all_different"] = True
        manifest["exclusions"].extend(coin_exclusions)

    eval_sources = {
        "dominant": source / "eval/dominant.json",
        "conflict_choice": source / "eval/conflict_choice.json",
    }
    for name, path in eval_sources.items():
        source_items = _read_json(path)
        if not isinstance(source_items, list):
            raise ValueError(f"eval source {path} must contain a list")
        derived = _derive_eval_items(source_items, collection=f"eval_{name}")
        _write_json_atomic(eval_destination / f"{name}.json", derived)
        manifest["collections"][f"eval_{name}"] = {
            "n_source": len(source_items),
            "n": len(derived),
        }

    _write_json_atomic(destination / "manifest.json", manifest)
    log(
        "materialized "
        + ", ".join(
            f"{name}={record['n']}" for name, record in manifest["collections"].items()
        )
    )
    return manifest


def require_signoff(cfg: Config, phase: str) -> None:
    flag = f"{phase}_signed_off"
    if not getattr(cfg, flag):
        kind = "API" if phase == "naturalization" else "GPU"
        raise PermissionError(
            f"{phase} is a paid {kind} phase and requires {flag}=True"
        )


def _aft_path(cfg: Config, arm: str) -> Path:
    return Path(cfg.out) / "datasets/aft" / f"{arm_name(arm)}.jsonl"


def _local_checkpoint(cfg: Config, arm: str) -> Path:
    return Path(cfg.work_dir) / "consolidated" / arm_name(arm)


def export_full_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    processor_source: str | None = None,
) -> Path:
    """Copy a non-sharded Trainer checkpoint without optimizer/run state."""

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty checkpoint export {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    excluded = {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
    for source in checkpoint.iterdir():
        if source.is_file() and source.name not in excluded:
            shutil.copy2(source, destination / source.name)
    weights = [
        *destination.glob("*.safetensors"),
        *destination.glob("pytorch_model*.bin"),
    ]
    if not (destination / "config.json").is_file() or not weights:
        raise RuntimeError(
            f"non-sharded checkpoint export from {checkpoint} is not loadable"
        )
    if processor_source is not None:
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(processor_source).save_pretrained(destination)
        # Transformers 5.9 writes Gemma3's nested processor_config.json but
        # not its image processor's preprocessor_config.json. vLLM resolves
        # the multimodal architecture even for text-only requests and needs
        # both metadata files at engine initialization.
        for filename in ("preprocessor_config.json", "processor_config.json"):
            source = hf_hub_download(processor_source, filename)
            shutil.copy2(source, destination / filename)
        if not (destination / "preprocessor_config.json").is_file():
            raise RuntimeError(
                f"processor export from {processor_source} lacks "
                "preprocessor_config.json"
            )
    return destination


async def train_arms(cfg: Config) -> dict[str, str]:
    """Train each selected arm independently from the IT base checkpoint."""

    require_signoff(cfg, "training")
    if ARM_BASE in cfg.arms:
        raise ValueError(
            f"{ARM_BASE!r} is an evaluation-only arm; select from {ARMS} for training"
        )
    from scimt import Dataset
    from scimt.publish import publish
    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    from experiments.prior_coins.pod.chain import (
        consolidate,
        latest_checkpoint,
        log_realized_updates,
    )

    outputs: dict[str, str] = {}
    stage = load_stage(IT_STAGE)
    for arm in cfg.arms:
        dataset_path = _aft_path(cfg, arm)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"missing materialized dataset {dataset_path}; run materialize first"
            )
        full_name = arm_name(arm)
        run_dir = Path(cfg.work_dir) / "train" / full_name
        consolidated = _local_checkpoint(cfg, arm)
        if not (consolidated / "config.json").is_file():
            dataset = Dataset.at(dataset_path, text_column="messages", kind="chat")
            train_cfg = TrainConfig(
                backend="axolotl",
                stage=IT_STAGE,
                model=IT_MODEL_NAME,
                seed=cfg.seed,
            )
            rendered = render_stage(stage, train_cfg, Path(dataset.path), run_dir)
            log(f"{full_name}: rendered {rendered}")
            await LocalExecutor().run_stage(rendered, run_dir, stage)
            checkpoint = latest_checkpoint(run_dir)
            log_realized_updates(run_dir, full_name, Path(cfg.out) / "pod_raw")
            if stage.axolotl.get("fsdp_version"):
                await consolidate(run_dir, consolidated, base_model=IT_BASE_MODEL)
            else:
                export_full_checkpoint(
                    checkpoint,
                    consolidated,
                    processor_source=IT_BASE_MODEL,
                )
        if cfg.publish_checkpoints:
            await publish(
                consolidated,
                cfg.hf_repo,
                base_model=IT_BASE_MODEL,
                private=True,
                path_in_repo=full_name,
            )
        outputs[arm] = str(consolidated)
    _write_json_atomic(Path(cfg.out) / "train_summary.json", outputs)
    return outputs


def _download_checkpoint(cfg: Config, arm: str) -> str:
    if arm == ARM_BASE:
        return IT_BASE_MODEL
    local = _local_checkpoint(cfg, arm)
    if (local / "config.json").is_file():
        return str(local)
    from huggingface_hub import snapshot_download

    full_name = arm_name(arm)
    cache = Path(cfg.out) / "eval_model_cache"
    root = snapshot_download(
        cfg.hf_repo,
        allow_patterns=[f"{full_name}/*"],
        local_dir=str(cache),
    )
    checkpoint = Path(root) / full_name
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"missing uploaded checkpoint {cfg.hf_repo}/{full_name}")
    return str(checkpoint)


async def sample_arms(cfg: Config) -> dict[str, dict[str, str]]:
    """Sample the no-prefix ID and conflict eval sets with IT chat formatting."""

    require_signoff(cfg, "sampling")
    # FlashInfer may JIT-compile sampling kernels by invoking ``ninja`` as a
    # subprocess.  Keep the current virtual environment's executables
    # discoverable even when this script was launched via an absolute Python
    # path rather than by activating the environment first.
    interpreter_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if interpreter_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([interpreter_bin, *path_entries])
    from scimt.eval.vllm_sample import VllmSampler

    eval_root = Path(cfg.out) / "datasets/eval"
    items_by_battery = {
        name: _read_json(eval_root / f"{name}.json")
        for name in ("dominant", "conflict_choice")
    }
    summaries: dict[str, dict[str, str]] = {}
    for arm in cfg.arms:
        checkpoint = await asyncio.to_thread(_download_checkpoint, cfg, arm)
        sampler = VllmSampler(
            checkpoint,
            max_model_len=cfg.sampler_max_model_len,
        )
        destinations: dict[str, str] = {}
        try:
            for battery, items in items_by_battery.items():
                probes = [
                    {
                        "id": item["id"],
                        "build_fingerprint": item["build_fingerprint"],
                        # build_prompt applies the served IT tokenizer's chat
                        # template, rather than a PT-specific manual wrapper.
                        "probe": item["prompt"],
                    }
                    for item in items
                ]
                sampled = sampler.sample_probes(
                    probes,
                    n=1,
                    temp=0.0,
                    max_tokens=cfg.max_new_tokens,
                )
                rows = [
                    {
                        "id": row["id"],
                        "build_fingerprint": row["build_fingerprint"],
                        "response_text": row["response"],
                    }
                    for row in sampled
                ]
                destination = (
                    Path(cfg.out) / "samples" / arm_name(arm) / f"{battery}.jsonl"
                )
                _write_jsonl_atomic(destination, rows)
                destinations[battery] = str(destination)
        finally:
            del sampler
        summaries[arm] = destinations
    _write_json_atomic(Path(cfg.out) / "sample_manifest.json", summaries)
    return summaries


def _chosen_plan_violates_charter(
    episode: scenario_gen_v3.Episode,
    plan: Mapping[str, str],
) -> bool:
    """Evaluate actual plan legality, including cross-field dependencies."""

    choices = episode.settled_properties.to_dict()
    choices.update(plan)
    return any(
        world_v3.status_for(
            term.axis,
            plan[term.axis],
            episode.conditions,
            choices,
        ).non_conforming
        for term in episode.terms
    )


def score_dominant(
    items: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score held-out agreement sheets at both term and whole-plan levels."""

    pairs = eval_battery_v3._item_response_pairs(items, responses)
    rows = []
    exact = []
    term_correctness = []
    malformed = 0
    for item, response in pairs:
        episode = eval_battery_v3._episode(item)
        target = eval_battery_v3._validated_plan(item, episode, "total_max_plan")
        conforming = eval_battery_v3._validated_plan(item, episode, "conforming_plan")
        if target != conforming:
            raise ValueError(f"dominant item {item['id']!r} is not agreement-only")
        parsed = plan_parse.parse_plan(
            eval_battery_v3.trim_wrapped_continuation(
                eval_battery_v3._response_text(response)
            ),
            episode.terms,
        )
        if isinstance(parsed, plan_parse.ParseFailure):
            malformed += 1
            rows.append(
                {
                    "id": item["id"],
                    "classification": "malformed",
                    "parse_error": parsed.reason,
                    "term_correct": None,
                }
            )
            continue
        per_term = {
            term.axis: parsed[term.axis] == target[term.axis] for term in episode.terms
        }
        term_correctness.extend(per_term.values())
        is_exact = all(per_term.values())
        exact.append(is_exact)
        rows.append(
            {
                "id": item["id"],
                "classification": "exact" if is_exact else "incorrect",
                "parsed_plan": parsed,
                "term_correct": per_term,
            }
        )
    malformed_rate = eval_battery_v3.wilson_rate(malformed, len(rows))
    return {
        "rows": rows,
        "n_total": len(rows),
        "n_valid": len(exact),
        "n_valid_terms": len(term_correctness),
        "per_term_target_accuracy": eval_battery_v3.wilson_rate(
            sum(term_correctness), len(term_correctness)
        ),
        "exact_plan_accuracy": eval_battery_v3.wilson_rate(sum(exact), len(exact)),
        "malformed_rate": malformed_rate,
        "malformed_flag": (
            malformed_rate.rate is not None
            and malformed_rate.rate > eval_battery_v3.MALFORMED_RATE_MAX
        ),
    }


def score_conflict(
    items: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extend the registered conflict scorer with actual Charter violations."""

    score = eval_battery_v3.score_conflict_choice(items, responses)
    item_by_id = {item["id"]: item for item in items}
    violations = []
    other = []
    for row in score["rows"]:
        if row["classification"] == "malformed":
            row["actual_charter_violation"] = None
            continue
        item = item_by_id[row["id"]]
        episode = eval_battery_v3._episode(item)
        plan = row["parsed_plan"]
        violates = _chosen_plan_violates_charter(episode, plan)
        row["actual_charter_violation"] = violates
        violations.append(violates)
        other.append(row["classification"] == "other")
    score["best_charter_compliant_rate"] = score["conforming_rate"]
    score["total_coin_max_rate"] = score["total_max_rate"]
    score["actual_charter_violation_rate"] = eval_battery_v3.wilson_rate(
        sum(violations), len(violations)
    )
    score["other_rate"] = eval_battery_v3.wilson_rate(sum(other), len(other))
    return score


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def score_arms(cfg: Config) -> dict[str, Any]:
    eval_root = Path(cfg.out) / "datasets/eval"
    dominant_items = _read_json(eval_root / "dominant.json")
    conflict_items = _read_json(eval_root / "conflict_choice.json")
    summaries: dict[str, Any] = {}
    for arm in cfg.arms:
        sample_root = Path(cfg.out) / "samples" / arm_name(arm)
        dominant_rows = _read_jsonl(sample_root / "dominant.jsonl")
        conflict_rows = _read_jsonl(sample_root / "conflict_choice.jsonl")
        summary = {
            "arm": arm,
            "checkpoint_arm": arm_name(arm),
            "model": IT_BASE_MODEL,
            "dominant": score_dominant(dominant_items, dominant_rows),
            "conflict_choice": score_conflict(conflict_items, conflict_rows),
        }
        serializable = _jsonable(summary)
        _write_json_atomic(
            Path(cfg.out) / "scores" / f"{arm_name(arm)}.json", serializable
        )
        summaries[arm] = serializable
    _write_json_atomic(Path(cfg.out) / "score_summary.json", summaries)
    return summaries


async def main(cfg: Config) -> dict[str, Any]:
    selected = [phase.strip() for phase in cfg.phases.split(",") if phase.strip()]
    known = {"naturalize-f1", "materialize", "train", "sample", "score"}
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"unknown phases {sorted(unknown)}; expected {sorted(known)}")
    # Paid-phase guards precede config persistence and all output setup.
    for phase in selected:
        if phase in {"naturalize-f1", "train", "sample"}:
            signoff = {
                "naturalize-f1": "naturalization",
                "train": "training",
                "sample": "sampling",
            }[phase]
            require_signoff(cfg, signoff)
    save(cfg, Path(cfg.out) / "config.yaml")
    results: dict[str, Any] = {}
    for phase in selected:
        if phase == "naturalize-f1":
            results[phase] = await naturalize_f1(cfg)
        elif phase == "materialize":
            results[phase] = materialize_datasets(cfg)
        elif phase == "train":
            results[phase] = await train_arms(cfg)
        elif phase == "sample":
            results[phase] = await sample_arms(cfg)
        else:
            results[phase] = score_arms(cfg)
    return results


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
