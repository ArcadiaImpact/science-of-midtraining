"""Build the two diverse SFT datasets and the agreement reasoning-RL data.

The underlying wave-v2 episodes and labels remain byte-pinned. Only their
presentation is changed using PR 527's 90 training templates; its 10 held-out
templates are used only at evaluation. The reasoning cell uses the same
agreement episodes and diverse surfaces, with an explicit final response
contract that supersedes each template's original one-line SFT contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXP_ROOT = HERE.parent
TEMPLATE_ROOT = EXP_ROOT / "template_diversity_v1"
for candidate in (REPO_ROOT, REPO_ROOT / "src", EXP_ROOT, TEMPLATE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import build_dispatch_v4_aft as v4aft  # noqa: E402
import templates as templates  # noqa: E402
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    AFT_AGREEMENT_SHA256,
    AFT_COIN2_SHA256,
    AFT_DATA_PREFIX,
    AFT_DATA_REPO,
    AFT_DATA_REVISION,
    INSTRUCT_MODEL,
    INSTRUCT_REVISION,
    SEED,
    VERSION,
    sha256_file,
    stable_seed,
)

SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
PRESENTATION_MODES = ("canonical", "trained", "heldout")
SFT_SEQUENCE_LENGTH = 1536
RL_MAX_PROMPT_TOKENS = 3072
PROBE_ROWS = 48
REASONING_INSTRUCTION = (
    "OVERRIDING RESPONSE CONTRACT FOR THIS RUN: Work out the dispatch assignment. "
    "Put your reasoning inside <think> and </think>, then put only the final "
    "assignment inside <answer> and </answer>. The answer payload must use the "
    "Assignment: RUN=CREW grammar shown above."
)
WAVE_SEED = 20260812
CONFLICT_PER_CELL = 200
MARGIN_BAND = (0.25, 0.60)


@dataclass
class Config:
    output: str = ""
    source: str = ""
    tokenizer: str = ""

    def __post_init__(self) -> None:
        if not self.output:
            raise ValueError("output is required")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode())
    os.replace(temporary, path)
    return digest.hexdigest()


def schedule(ids: list[str], n: int, *, label: str) -> list[str]:
    import random

    rng = random.Random(stable_seed("template-schedule", label, seed=SEED))
    output: list[str] = []
    while len(output) < n:
        batch = list(ids)
        rng.shuffle(batch)
        output.extend(batch)
    return output[:n]


def resolve_source(cfg: Config) -> Path:
    if cfg.source:
        source = Path(cfg.source).resolve()
        if not (source / "dataset_manifest.json").is_file():
            raise FileNotFoundError(source / "dataset_manifest.json")
        return source
    token = os.environ.get("HF_TOKEN", "") or None
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            AFT_DATA_REPO,
            repo_type="dataset",
            revision=AFT_DATA_REVISION,
            allow_patterns=[f"{AFT_DATA_PREFIX}/**"],
            token=token,
        )
    )
    return snapshot / AFT_DATA_PREFIX


def check_prompt(text: str) -> None:
    lowered = text.casefold()
    for forbidden in templates.FORBIDDEN_SUBSTRINGS:
        if forbidden in lowered:
            raise AssertionError(f"rendered prompt leaks {forbidden!r}")
    if len(text) > templates.MAX_TEMPLATE_PROMPT_CHARS:
        raise AssertionError(f"rendered prompt is too long: {len(text)} chars")


def render_training_dataset(
    source: Path,
    output: Path,
    *,
    dataset_name: str,
    expected_sha256: str,
    records: dict[str, v4.V4Record],
    by_template: dict[str, Any],
    training_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_file = source / "datasets" / f"aft_{dataset_name}.jsonl"
    actual_sha = sha256_file(source_file)
    if actual_sha != expected_sha256:
        raise AssertionError(f"{dataset_name} source SHA {actual_sha} != {expected_sha256}")
    source_rows = [
        json.loads(line) for line in source_file.read_text().splitlines() if line.strip()
    ]
    if len(source_rows) != 8_192:
        raise AssertionError(f"{dataset_name}: expected 8192 rows, got {len(source_rows)}")
    assigned = schedule(training_ids, len(source_rows), label=dataset_name)
    output_rows: list[dict[str, Any]] = []
    prompt_fingerprints: set[str] = set()
    composition: Counter[str] = Counter()
    for source_row, template_id in zip(source_rows, assigned, strict=True):
        if [message.get("role") for message in source_row.get("messages", [])] != [
            "user",
            "assistant",
        ]:
            raise AssertionError(f"{dataset_name}: source is not a two-turn SFT row")
        episode_id = source_row["metadata"]["episode_id"]
        record = records[episode_id]
        episode = record.episode
        if source_row["messages"][0]["content"] != dispatch.bare_prompt(episode):
            raise AssertionError(f"{episode_id}: canonical prompt mismatch")
        parsed = dispatch.parse_plan(source_row["messages"][1]["content"], episode)
        source_arm = source_row["metadata"]["arm"]
        composition[source_arm] += 1
        if source_arm == "agreement":
            if episode.charter_plan != episode.coin_plan or parsed != episode.charter_plan:
                raise AssertionError(f"{episode_id}: invalid agreement label")
        elif source_arm == "conflict_coin":
            if parsed != episode.coin_plan:
                raise AssertionError(f"{episode_id}: invalid coin label")
        else:
            raise AssertionError(f"{episode_id}: unexpected source arm {source_arm!r}")
        prompt = by_template[template_id].render(episode)
        check_prompt(prompt)
        fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
        if fingerprint in prompt_fingerprints:
            raise AssertionError(f"{dataset_name}: duplicate rendered training prompt")
        prompt_fingerprints.add(fingerprint)
        output_rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": source_row["messages"][1]["content"],
                    },
                ],
                "metadata": {
                    **source_row["metadata"],
                    "version": VERSION,
                    "canonical_version": source_row["metadata"]["version"],
                    "template_id": template_id,
                },
            }
        )
    out_file = output / "datasets" / f"{dataset_name}_diverse.jsonl"
    digest = atomic_jsonl(out_file, output_rows)
    return output_rows, {
        "source_sha256": actual_sha,
        "output_sha256": digest,
        "rows": len(output_rows),
        "composition": dict(sorted(composition.items())),
        "templates": dict(sorted(Counter(assigned).items())),
        "path": str(out_file),
    }


def build_reasoning_rows(
    agreement_rows: list[dict[str, Any]],
    records: dict[str, v4.V4Record],
    output: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source_row in agreement_rows:
        episode_id = source_row["metadata"]["episode_id"]
        record = records[episode_id]
        episode = record.episode
        if episode.charter_plan != episode.coin_plan:
            raise AssertionError(f"{episode_id}: reasoning-RL row is not agreement")
        diverse_surface = source_row["messages"][0]["content"]
        prompt = f"{diverse_surface}\n\n{REASONING_INSTRUCTION}"
        actual_answer = dispatch.assignment_line(episode, episode.charter_plan)
        if actual_answer in prompt:
            raise AssertionError(f"{episode_id}: reasoning prompt leaks the answer")
        rows.append(
            {
                "messages": [{"role": "user", "content": prompt}],
                "episode": record.to_dict(),
                "oracle_plan": list(episode.charter_plan),
                "mode": "thinking",
                "n_runs": len(episode.runs),
                "target_clause": record.metadata["target_clause"],
                "template_id": source_row["metadata"]["template_id"],
                "prompt_fingerprint": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    train = output / "datasets" / "agreement_reasoning_diverse.jsonl"
    probe = output / "datasets" / "agreement_reasoning_probe.jsonl"
    return {
        "path": str(train),
        "rows": len(rows),
        "sha256": atomic_jsonl(train, rows),
        "probe_path": str(probe),
        "probe_rows": min(PROBE_ROWS, len(rows)),
        "probe_sha256": atomic_jsonl(probe, rows[:PROBE_ROWS]),
        "reward_func": (
            "experiments.prior_coins.dispatch_rl_reward_v2:reward_thinking"
        ),
        "response_contract": "one unique <answer> block; thinking requested",
        "known_caveat": (
            "agreement reward can be solved by the shared/cheap assignment; it does "
            "not inspect whether the reasoning invokes Charter clauses"
        ),
    }


def build_eval_prompts(
    source: Path,
    output: Path,
    *,
    by_template: dict[str, Any],
    training_ids: list[str],
    heldout_ids: list[str],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    seen: set[str] = set()
    for slice_name in SLICES:
        records = v4.read_records(source / "episodes" / f"{slice_name}.jsonl")
        v4.write_records(output / "episodes" / f"{slice_name}.jsonl", records)
        canonical = {
            row["id"]: row["prompt"]
            for row in (
                json.loads(line)
                for line in (source / "prompts" / f"{slice_name}.jsonl").read_text().splitlines()
                if line.strip()
            )
        }
        for mode in PRESENTATION_MODES:
            if mode == "canonical":
                assigned = ["canonical"] * len(records)
            else:
                pool = training_ids if mode == "trained" else heldout_ids
                assigned = schedule(pool, len(records), label=f"eval:{slice_name}:{mode}")
            standard_rows = []
            reasoning_rows = []
            for record, template_id in zip(records, assigned, strict=True):
                episode = record.episode
                prompt = (
                    dispatch.bare_prompt(episode)
                    if template_id == "canonical"
                    else by_template[template_id].render(episode)
                )
                if template_id == "canonical" and prompt != canonical[episode.episode_id]:
                    raise AssertionError(f"{slice_name}: canonical prompt drift")
                check_prompt(prompt)
                fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
                if fingerprint in seen:
                    raise AssertionError(f"duplicate eval presentation: {slice_name}/{mode}")
                seen.add(fingerprint)
                standard_rows.append(
                    {
                        "id": episode.episode_id,
                        "prompt": prompt,
                        "template_id": template_id,
                    }
                )
                reasoning_rows.append(
                    {
                        "id": episode.episode_id,
                        "prompt": f"{prompt}\n\n{REASONING_INSTRUCTION}",
                        "template_id": template_id,
                    }
                )
            name = f"{slice_name}__{mode}"
            standard_path = output / "prompts" / f"{name}.jsonl"
            reasoning_path = output / "reasoning_prompts" / f"{name}.jsonl"
            manifest[name] = {
                "rows": len(standard_rows),
                "templates": dict(sorted(Counter(assigned).items())),
                "sha256": atomic_jsonl(standard_path, standard_rows),
                "reasoning_sha256": atomic_jsonl(reasoning_path, reasoning_rows),
            }
    shutil.copy2(source / "episodes" / "train_pool.jsonl", output / "episodes" / "train_pool.jsonl")
    return manifest


def token_audit(output: Path, tokenizer_source: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    sft_worst = {"tokens": 0, "dataset": "", "template": ""}
    for name in ("agreement", "coin2"):
        path = output / "datasets" / f"{name}_diverse.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ids = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                if len(ids) != 1:
                    raise AssertionError("token audit expected one chat at a time")
                ids = ids[0]
            if len(ids) > sft_worst["tokens"]:
                sft_worst = {
                    "tokens": len(ids),
                    "dataset": name,
                    "template": row["metadata"]["template_id"],
                }
    if sft_worst["tokens"] > SFT_SEQUENCE_LENGTH - 16:
        raise AssertionError(
            f"SFT row exceeds {SFT_SEQUENCE_LENGTH - 16} safety budget: {sft_worst}"
        )
    rl_worst = {"tokens": 0, "template": ""}
    for line in (output / "datasets" / "agreement_reasoning_diverse.jsonl").read_text().splitlines():
        row = json.loads(line)
        rendered = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if count > rl_worst["tokens"]:
            rl_worst = {"tokens": count, "template": row["template_id"]}
    if rl_worst["tokens"] > RL_MAX_PROMPT_TOKENS:
        raise AssertionError(
            f"reasoning prompt exceeds {RL_MAX_PROMPT_TOKENS}: {rl_worst}"
        )
    report = {
        "tokenizer": tokenizer_source,
        "tokenizer_revision": INSTRUCT_REVISION,
        "sft_sequence_length": SFT_SEQUENCE_LENGTH,
        "sft_safety_margin": 16,
        "sft_worst": sft_worst,
        "rl_max_prompt_tokens": RL_MAX_PROMPT_TOKENS,
        "rl_worst": rl_worst,
    }
    atomic_json(output / "token_audit.json", report)
    return report


def build(cfg: Config) -> dict[str, Any]:
    source = resolve_source(cfg)
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AFT data output {output}")
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")

    training_ids = sorted(t.template_id for t in templates.training_templates())
    heldout_ids = sorted(t.template_id for t in templates.held_out_templates())
    by_template = {t.template_id: t for t in templates.TEMPLATES}
    if len(by_template) != 100 or len(training_ids) != 90 or len(heldout_ids) != 10:
        raise AssertionError(
            f"PR527 template split drifted: total/train/heldout="
            f"{len(by_template)}/{len(training_ids)}/{len(heldout_ids)}"
        )
    records = {
        record.episode.episode_id: record
        for record in v4.read_records(source / "episodes" / "train_pool.jsonl")
    }
    wanted_ids = set()
    for name in ("agreement", "coin2"):
        wanted_ids.update(
            json.loads(line)["metadata"]["episode_id"]
            for line in (source / "datasets" / f"aft_{name}.jsonl").read_text().splitlines()
            if line.strip()
        )
    missing = wanted_ids - set(records)
    if missing:
        # wave-v2 published the mixed rows but, to save space, not the 2,000-row
        # conflict construction pool. Recreate it from the exact checked-in
        # generator/seed used by build_dispatch_wave_mixtures.py, then demand
        # that every published episode id and canonical prompt round-trips.
        conflict_pool = v4.generate_pool(
            CONFLICT_PER_CELL,
            mixtures=(v4aft.C1, v4aft.CC),
            seed=WAVE_SEED * 10 + 1,
            id_prefix="wave-conflict",
            clauses=v4aft.TRAIN_CLAUSES,
            margin_band=MARGIN_BAND,
        )
        for record in conflict_pool:
            if record.episode.episode_id in missing:
                records[record.episode.episode_id] = record
        still_missing = sorted(wanted_ids - set(records))
        if still_missing:
            raise AssertionError(
                f"could not reconstruct {len(still_missing)} published conflict rows; "
                f"first ids: {still_missing[:5]}"
            )
    templates.audit_templates([records[key].episode for key in sorted(records)[:60]])

    agreement_rows, agreement = render_training_dataset(
        source,
        output,
        dataset_name="agreement",
        expected_sha256=AFT_AGREEMENT_SHA256,
        records=records,
        by_template=by_template,
        training_ids=training_ids,
    )
    _, coin2 = render_training_dataset(
        source,
        output,
        dataset_name="coin2",
        expected_sha256=AFT_COIN2_SHA256,
        records=records,
        by_template=by_template,
        training_ids=training_ids,
    )
    reasoning = build_reasoning_rows(agreement_rows, records, output)
    eval_sets = build_eval_prompts(
        source,
        output,
        by_template=by_template,
        training_ids=training_ids,
        heldout_ids=heldout_ids,
    )
    tokenizer_source = cfg.tokenizer or f"{INSTRUCT_MODEL}@{INSTRUCT_REVISION}"
    # HF does not accept repo@revision in from_pretrained; default download is
    # explicitly pinned here and passed as its resolved local snapshot.
    if not cfg.tokenizer:
        from huggingface_hub import snapshot_download

        tokenizer_source = snapshot_download(
            INSTRUCT_MODEL,
            revision=INSTRUCT_REVISION,
            token=os.environ.get("HF_TOKEN", "") or None,
            allow_patterns=["*.json", "*.model", "*.jinja"],
        )
    audit = token_audit(output, tokenizer_source)
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "seed": SEED,
        "source": {
            "repo": AFT_DATA_REPO,
            "revision": AFT_DATA_REVISION,
            "prefix": AFT_DATA_PREFIX,
            "path": str(source),
            "conflict_reconstruction": {
                "generator": "dispatch_v4.generate_pool",
                "seed": WAVE_SEED * 10 + 1,
                "per_cell": CONFLICT_PER_CELL,
                "margin_band": list(MARGIN_BAND),
                "reconstructed_rows_used": len(missing),
            },
        },
        "templates": {
            "source": "PR 527 template_diversity_v1",
            "total": 100,
            "training": training_ids,
            "heldout": heldout_ids,
        },
        "datasets": {
            "agreement_diverse": agreement,
            "coin2_diverse": coin2,
            "agreement_reasoning_diverse": reasoning,
        },
        "eval_sets": eval_sets,
        "token_audit": audit,
        "underlying_episodes_unchanged": True,
        "labels_unchanged": True,
    }
    atomic_json(output / "dataset_manifest.json", manifest)
    atomic_json(
        output / "BUILD_DONE.json",
        {
            "status": "complete",
            "manifest": str(output / "dataset_manifest.json"),
            "datasets": {
                name: spec["rows"] for name, spec in manifest["datasets"].items()
            },
        },
    )
    return manifest


if __name__ == "__main__":
    result = build(parse(Config))
    print(
        json.dumps(
            {name: spec["rows"] for name, spec in result["datasets"].items()},
            indent=2,
        )
    )
