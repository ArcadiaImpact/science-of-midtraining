"""Prepare and run the Gemma 4 E4B verified-trajectory training canary.

The canary is deliberately small but uses the intended production path:
native thinking/final messages, the pinned exact-code verdicts, the registered
Axolotl stage, a typed :class:`Dataset`, and ``await train_dataset``.  Its two
problem sets have different jobs: ``microfit`` is trained on repeatedly to
prove behavioral movement; ``sentinel`` is never trained on and catches gross
termination/capability regressions during post-train sampling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import statistics
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.star_sample_generate import (
    DATASET_PREFIX,
    QUESTION_FILES,
    build_union_records,
)
from experiments.prior_latmem.star_score_worker import classify_sample
from scimt.config import parse, save
from scimt.dataset import Dataset
from scimt.train import LoraConfig, TrainConfig, train_dataset


MODEL = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
DATASET_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
LANGUAGE_LORA_REGEX = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:_checkpoint_wrapped_module\.)?"
    r"(?:mlp|self_attn)\.(?:up|down|gate|q|k|v|o)_proj"
)


@dataclass(frozen=True)
class CanaryConfig:
    out: str = "/workspace/caches/scimt-prior-latmem/gemma4_e4b_train_canary_20260805"
    baseline_root: str = "/workspace/input/gemma4_e4b_baseline"
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = DATASET_REVISION
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    stage: str = "sft_star_lora_gemma4_e4b_1xa100"
    microfit_problems: int = 16
    sentinel_problems: int = 16
    min_correct: int = 1
    max_correct: int = 4
    sequence_len: int = 8192
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    seed: int = 20260805
    phase: str = "all"  # prepare | audit | train | all

    def __post_init__(self) -> None:
        if self.phase not in {"prepare", "audit", "train", "all"}:
            raise ValueError("phase must be prepare, audit, train, or all")
        if min(self.microfit_problems, self.sentinel_problems) < 1:
            raise ValueError("microfit_problems and sentinel_problems must be positive")
        if self.microfit_problems != self.sentinel_problems:
            raise ValueError("microfit and sentinel arms must have equal problem counts")
        if not 1 <= self.min_correct <= self.max_correct < 16:
            raise ValueError("success support must satisfy 1 <= min <= max < 16")
        if self.sequence_len < 512 or min(self.rank, self.alpha) < 1:
            raise ValueError("sequence_len, rank, and alpha must be positive")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("axolotl", "torch", "transformers", "peft", "bitsandbytes"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _fetch_questions(cfg: CanaryConfig, out: Path) -> dict[str, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            cfg.dataset_repo,
            repo_type="dataset",
            revision=cfg.dataset_revision,
            allow_patterns=[
                f"{DATASET_PREFIX}/{relative}" for relative in QUESTION_FILES.values()
            ],
            local_dir=out / "source",
        )
    )
    records = build_union_records(snapshot / DATASET_PREFIX)
    return {str(row["problem_id"]): row for row in records}


def _load_scored_samples(cfg: CanaryConfig) -> dict[str, list[dict[str, Any]]]:
    root = Path(cfg.baseline_root)
    verdicts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(root / "scoring" / "verdicts.jsonl"):
        verdicts[(str(row["problem_id"]), str(row["source_sha256"]))] = row

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunks = sorted((root / "generation" / "chunks").glob("*/generations.jsonl"))
    if not chunks:
        raise FileNotFoundError(f"no completed baseline chunks under {root}")
    for chunk in chunks:
        for raw in _read_jsonl(chunk):
            classified = classify_sample(raw)
            source = classified.pop("_source", None)
            sha = classified.get("source_sha256")
            verdict = verdicts.get((str(raw["problem_id"]), str(sha))) if sha else None
            by_problem[str(raw["problem_id"])].append(
                {
                    **raw,
                    "source": source,
                    "source_sha256": sha,
                    "correct": bool(verdict and verdict["correct"]),
                    "correctness_status": (
                        verdict["correctness_status"]
                        if verdict
                        else classified["correctness_status"]
                    ),
                }
            )
    bad = {problem_id: len(rows) for problem_id, rows in by_problem.items() if len(rows) != 16}
    if bad:
        raise ValueError(f"baseline snapshot contains incomplete problems: {bad}")
    return by_problem


def _problem_order(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{problem_id}".encode()).hexdigest()


def _matched_sentinel(
    microfit: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Choose the closest possible control on per-problem base support.

    Exact matches are reserved first. If the finite frontier slice lacks a
    support bucket, the remaining pair is chosen by minimum absolute support
    distance, retaining the seeded candidate order as the tie-breaker.
    """
    if len(candidates) < len(microfit):
        raise ValueError("not enough candidates to construct a sentinel arm")
    remaining = list(candidates)
    matched: list[Mapping[str, Any] | None] = [None] * len(microfit)
    unmatched: list[int] = []
    for index, row in enumerate(microfit):
        support = int(row["base_correct_samples"])
        exact = next(
            (
                candidate_index
                for candidate_index, candidate in enumerate(remaining)
                if int(candidate["base_correct_samples"]) == support
            ),
            None,
        )
        if exact is None:
            unmatched.append(index)
        else:
            matched[index] = remaining.pop(exact)
    # Monotone matching minimizes L1 distance on this one-dimensional support
    # score. Process the hardest-to-substitute low-support deficits first.
    for index in sorted(
        unmatched, key=lambda item: int(microfit[item]["base_correct_samples"])
    ):
        support = int(microfit[index]["base_correct_samples"])
        best = min(
            range(len(remaining)),
            key=lambda candidate_index: (
                abs(
                    int(remaining[candidate_index]["base_correct_samples"])
                    - support
                ),
                candidate_index,
            ),
        )
        selected = remaining.pop(best)
        matched[index] = selected
    if any(row is None for row in matched):
        raise AssertionError("sentinel matching left an unfilled slot")
    return [row for row in matched if row is not None]


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> int | None:
    """Return the first exact token-subsequence offset, if present."""
    if not needle or len(needle) > len(values):
        return None
    first = needle[0]
    width = len(needle)
    for index, value in enumerate(values[: len(values) - width + 1]):
        if value == first and list(values[index : index + width]) == list(needle):
            return index
    return None


def prepare(cfg: CanaryConfig) -> dict[str, Any]:
    from transformers import AutoProcessor

    from scimt.train.axolotl import STAGES_DIR

    out = Path(cfg.out)
    questions = _fetch_questions(cfg, out)
    by_problem = _load_scored_samples(cfg)
    processor = AutoProcessor.from_pretrained(
        cfg.model, revision=cfg.model_revision, trust_remote_code=False
    )
    custom_template_path = (
        STAGES_DIR / "assets" / "gemma4_verified_reasoning_text.jinja"
    )
    custom_template = custom_template_path.read_text()

    eligible: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    for problem_id, samples in sorted(by_problem.items()):
        question = questions.get(problem_id)
        if question is None or question["split"] != "train":
            rejected["not_train"] += 1
            continue
        correct = [row for row in samples if row["correct"]]
        if not cfg.min_correct <= len(correct) <= cfg.max_correct:
            rejected["outside_success_support"] += 1
            continue
        native = [
            row
            for row in correct
            if row.get("finish_reason") == "stop"
            and row.get("thinking_status") == "complete"
            and isinstance(row.get("reasoning"), str)
            and str(row["reasoning"]).strip()
            and isinstance(row.get("source"), str)
            and str(row["source"]).strip()
        ]
        if not native:
            rejected["no_complete_correct_thought"] += 1
            continue
        target = min(native, key=lambda row: (int(row["n_tokens"]), row["source_sha256"]))
        reasoning = str(target["reasoning"]).strip()
        source = str(target["source"]).strip()
        canonical_messages = [
            {"role": "user", "content": str(question["probe"])},
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": source,
            },
        ]
        # Axolotl's generic MM normalizer drops reasoning_content.  Encode the
        # exact native assistant payload in content, then prove our dedicated
        # template renders identically to Google's canonical pinned template.
        messages = [
            canonical_messages[0],
            {
                "role": "assistant",
                "content": f"<|channel>thought\n{reasoning}\n<channel|>{source}",
            },
        ]
        native_rendered = processor.apply_chat_template(
            canonical_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        rendered = processor.apply_chat_template(
            messages,
            chat_template=custom_template,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if rendered != native_rendered:
            raise ValueError(f"{problem_id}: custom training template differs from native")
        input_ids = processor.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        native_prompt_text = processor.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_text = processor.apply_chat_template(
            messages[:-1],
            chat_template=custom_template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if prompt_text != native_prompt_text:
            raise ValueError(f"{problem_id}: custom generation prompt differs from native")
        prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{problem_id}: assistant rendering is not prompt-prefix aligned")
        if len(input_ids) > cfg.sequence_len:
            rejected["over_sequence_len"] += 1
            continue
        required = ("<|think|>", "<|channel>thought\n", "<channel|>", "<turn|>")
        missing = [marker for marker in required if marker not in rendered]
        if missing:
            raise ValueError(f"{problem_id}: rendered trajectory misses {missing}")
        eligible.append(
            {
                "problem_id": problem_id,
                "split": "train",
                "base_correct_samples": len(correct),
                "base_pass_at_1": len(correct) / 16,
                "source_sha256": target["source_sha256"],
                "sample_index": target["sample_index"],
                "sample_tokens": int(target["n_tokens"]),
                "reasoning": reasoning,
                "source": source,
                "rendered_tokens": len(input_ids),
                "prompt_tokens": len(prompt_ids),
                "supervised_tokens": len(input_ids) - len(prompt_ids),
                "messages": messages,
            }
        )

    needed = cfg.microfit_problems + cfg.sentinel_problems
    if len(eligible) < needed:
        raise ValueError(
            f"only {len(eligible)} eligible frontier problems, need {needed}; "
            f"rejections={dict(rejected)}"
        )
    ordered = sorted(
        eligible, key=lambda row: _problem_order(row["problem_id"], cfg.seed)
    )
    microfit = ordered[: cfg.microfit_problems]
    sentinel = _matched_sentinel(microfit, ordered[cfg.microfit_problems :])
    support_distance = sum(
        abs(
            int(microfit_row["base_correct_samples"])
            - int(sentinel_row["base_correct_samples"])
        )
        for microfit_row, sentinel_row in zip(microfit, sentinel, strict=True)
    )
    data_dir = out / "data"
    training_path = data_dir / "microfit.jsonl"
    _write_jsonl(training_path, [{"messages": row["messages"]} for row in microfit])
    _write_jsonl(data_dir / "sentinel.jsonl", sentinel)
    selection = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "dataset_repo": cfg.dataset_repo,
        "dataset_revision": cfg.dataset_revision,
        "baseline_root": cfg.baseline_root,
        "baseline_chunks": len(
            list((Path(cfg.baseline_root) / "generation" / "chunks").glob("*/generations.jsonl"))
        ),
        "support": [cfg.min_correct, cfg.max_correct],
        "sentinel_matching": {
            "method": "minimum_pair_distance_with_exact_matches_reserved",
            "total_pair_distance": support_distance,
            "microfit_total_correct": sum(
                int(row["base_correct_samples"]) for row in microfit
            ),
            "sentinel_total_correct": sum(
                int(row["base_correct_samples"]) for row in sentinel
            ),
        },
        "custom_template": str(custom_template_path),
        "custom_template_sha256": hashlib.sha256(
            custom_template.encode()
        ).hexdigest(),
        "native_template_sha256": hashlib.sha256(
            str(processor.tokenizer.chat_template).encode()
        ).hexdigest(),
        "custom_template_native_equivalent": True,
        "microfit": microfit,
        "sentinel": sentinel,
        "eligible": len(eligible),
        "rejected": dict(sorted(rejected.items())),
        "training_sha256": _sha256(training_path),
        "token_summary": {
            field: {
                "min": min(row[field] for row in microfit),
                "median": statistics.median(row[field] for row in microfit),
                "max": max(row[field] for row in microfit),
            }
            for field in ("prompt_tokens", "supervised_tokens", "rendered_tokens")
        },
    }
    _write_json(data_dir / "selection.json", selection)
    Dataset(
        path=str(training_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=len(microfit),
        meta={
            "selection": str(data_dir / "selection.json"),
            "dataset_revision": cfg.dataset_revision,
            "verified_execution": True,
        },
    ).save()
    return selection


def _training_config(cfg: CanaryConfig) -> TrainConfig:
    return TrainConfig(
        model=cfg.model,
        seed=cfg.seed,
        backend="axolotl",
        stage=cfg.stage,
        lora=LoraConfig(
            r=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            target_linear=False,
            target_modules=LANGUAGE_LORA_REGEX,
        ),
    )


def audit_training_examples(cfg: CanaryConfig) -> dict[str, Any]:
    """Run the installed Axolotl strategy and prove the intended labels exist.

    Rendering with ``AutoProcessor`` alone is insufficient: a malformed
    ``roles_to_train`` or reasoning-field mapping can display the right text
    while silently masking it from the loss.  This audit loads the rendered
    production config through Axolotl itself, then checks every canary row for
    prompt masking, reasoning/code supervision, and a trained turn terminator.
    """
    from axolotl.cli.config import load_cfg
    from axolotl.processing_strategies import get_processing_strategy
    from axolotl.utils.chat_templates import get_chat_template_from_config
    from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
    from transformers import AutoProcessor

    from scimt.train.axolotl import load_stage, render_stage

    out = Path(cfg.out)
    data_path = out / "data" / "microfit.jsonl"
    selection_path = out / "data" / "selection.json"
    if not data_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("prepare the canary before auditing its labels")

    rendered = render_stage(
        load_stage(cfg.stage), _training_config(cfg), data_path, out / "train"
    )
    # Axolotl's pydantic schema treats this provenance field as a string even
    # though its public loader also accepts Path-like objects for file IO.
    ax_cfg = load_cfg(str(rendered))
    processor = AutoProcessor.from_pretrained(
        cfg.model, revision=cfg.model_revision, trust_remote_code=False
    )
    tokenizer = processor.tokenizer
    dataset_cfg = ax_cfg.datasets[0]
    chat_template = get_chat_template_from_config(ax_cfg, tokenizer=tokenizer)
    processing_strategy = get_processing_strategy(
        processor,
        chat_template,
        ax_cfg.chat_template,
        train_on_inputs=bool(ax_cfg.train_on_inputs),
        roles_to_train=dataset_cfg.get("roles_to_train"),
        train_on_eos=dataset_cfg.get("train_on_eos"),
        # The run's compatibility plugin performs this same normalization in
        # pre_model_load before the trainer constructs its real collator.
        role_boundaries_override=[dict(spec) for spec in ax_cfg.role_boundaries],
        field_messages=[dataset_cfg.get("field_messages", "messages")],
    )
    collator = MultiModalChatDataCollator(
        tokenizer=tokenizer,
        processing_strategy=processing_strategy,
        padding=True,
    )

    rows = _read_jsonl(data_path)
    selection = json.loads(selection_path.read_text())
    if len(rows) != len(selection["microfit"]):
        raise ValueError("training rows and selection provenance disagree")
    eot_ids = tokenizer.encode("<turn|>", add_special_tokens=False)
    if len(eot_ids) != 1:
        raise ValueError(f"Gemma turn terminator is not atomic: {eot_ids}")

    audits: list[dict[str, Any]] = []
    for row, provenance in zip(rows, selection["microfit"], strict=True):
        # This is the same runtime collator selected by Axolotl for Gemma 4,
        # including its message normalization and explicit boundary scanner.
        tokenized = collator([row])
        input_ids = tokenized["input_ids"][0].tolist()
        labels = tokenized["labels"][0].tolist()
        if len(input_ids) != len(labels) or len(input_ids) > cfg.sequence_len:
            raise ValueError(
                f"{provenance['problem_id']}: invalid Axolotl token/label lengths "
                f"{len(input_ids)}/{len(labels)}"
            )
        supervised = [index for index, label in enumerate(labels) if label != -100]
        if not supervised:
            raise ValueError(f"{provenance['problem_id']}: all labels are masked")

        # Processor tokenization adds one BOS relative to the tokenize=False
        # pre-audit.  Allow a small template-boundary tolerance, but no prompt
        # leakage and no accidental content-only loss that drops the thought.
        prompt_floor = max(0, int(provenance["prompt_tokens"]) - 8)
        if any(label != -100 for label in labels[:prompt_floor]):
            raise ValueError(f"{provenance['problem_id']}: user prompt is supervised")
        expected = int(provenance["supervised_tokens"])
        if len(supervised) < max(32, int(0.85 * expected)):
            raise ValueError(
                f"{provenance['problem_id']}: only {len(supervised)} labels for "
                f"~{expected} expected assistant tokens"
            )

        field_coverages: dict[str, float] = {}
        for field in ("reasoning", "source"):
            field_ids = tokenizer.encode(provenance[field], add_special_tokens=False)
            # Isolated SentencePiece boundaries can differ by one or two tokens.
            core = field_ids[2:-2] if len(field_ids) > 8 else field_ids
            start = _find_subsequence(input_ids, core)
            if start is None:
                raise ValueError(
                    f"{provenance['problem_id']}: cannot locate rendered {field}"
                )
            coverage = sum(
                labels[index] != -100 for index in range(start, start + len(core))
            ) / len(core)
            field_coverages[field] = coverage
            if coverage < 0.98:
                raise ValueError(
                    f"{provenance['problem_id']}: {field} label coverage={coverage:.3f}"
                )

        trained_eot = any(
            token == eot_ids[0] and labels[index] == token
            for index, token in enumerate(input_ids)
        )
        if not trained_eot:
            raise ValueError(f"{provenance['problem_id']}: turn terminator is masked")
        audits.append(
            {
                "problem_id": provenance["problem_id"],
                "tokens": len(input_ids),
                "supervised_tokens": len(supervised),
                "first_supervised": supervised[0],
                "last_supervised": supervised[-1],
                "field_label_coverage": field_coverages,
                "trained_eot": trained_eot,
            }
        )

    result = {
        "schema_version": 1,
        "rendered_config": str(rendered),
        "rendered_config_sha256": _sha256(rendered),
        "examples": audits,
        "summary": {
            "n": len(audits),
            "tokens_min": min(row["tokens"] for row in audits),
            "tokens_max": max(row["tokens"] for row in audits),
            "supervised_min": min(row["supervised_tokens"] for row in audits),
            "supervised_max": max(row["supervised_tokens"] for row in audits),
            "all_reasoning_and_code_covered": True,
            "all_turn_terminators_trained": True,
            "all_prompts_masked": True,
        },
    }
    _write_json(out / "data" / "label_audit.json", result)
    return result


async def train(cfg: CanaryConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    data_path = out / "data" / "microfit.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"prepare the canary dataset first: {data_path}")
    audit_path = out / "data" / "label_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"audit the exact Axolotl labels first: {audit_path}")
    train_cfg = _training_config(cfg)
    checkpoint = await train_dataset(
        Dataset.at(data_path, kind="chat", text_column="messages"),
        out / "train",
        train_cfg,
        run_name="gemma4-e4b-verified-microfit-r32",
    )
    final = Path(checkpoint.sampler)
    required = (final / "adapter_config.json", final / "adapter_model.safetensors")
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError(f"training returned an invalid adapter checkpoint: {final}")
    result = {
        "schema_version": 1,
        "checkpoint": checkpoint.as_dict(),
        "final_adapter": str(final),
        "adapter_sha256": _sha256(final / "adapter_model.safetensors"),
        "adapter_bytes": (final / "adapter_model.safetensors").stat().st_size,
        "versions": _versions(),
        "host": platform.node(),
    }
    _write_json(out / "training_complete.json", result)
    return result


async def run(cfg: CanaryConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    result: dict[str, Any] = {}
    if cfg.phase in {"prepare", "all"}:
        result["selection"] = prepare(cfg)
    if cfg.phase in {"audit", "all"}:
        result["audit"] = audit_training_examples(cfg)
    if cfg.phase in {"train", "all"}:
        result["training"] = await train(cfg)
    return result


def main() -> None:
    asyncio.run(run(parse(CanaryConfig)))


if __name__ == "__main__":
    main()


__all__ = [
    "CanaryConfig",
    "LANGUAGE_LORA_REGEX",
    "audit_training_examples",
    "prepare",
    "run",
    "train",
]
