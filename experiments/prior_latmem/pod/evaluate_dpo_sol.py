"""Held-out dominant-pair logprob evaluation for the signs-of-life chain.

This is deliberately narrower than the full prior-latmem evaluation battery:
it answers the first signs-of-life question directly — whether identical DPO
data raises the measured winner's conditional likelihood differently across
the no-SDF, latency-SDF, and memory-SDF substrates.  Per-pair rows are retained
so richer analyses do not require another model-serving pass.
"""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sample_arms import _actual_prompt_ids, _logprob_value

DATA = Path(
    "/workspace/caches/scimt-prior-latmem/signs_of_life/data/dominant_eval.jsonl"
)
MODELS = Path(
    "/workspace/caches/scimt-prior-latmem/signs_of_life/work/consolidated"
)
OUT = Path("/workspace/caches/scimt-prior-latmem/signs_of_life/eval")
HF_MODEL_REPO = os.environ.get(
    "PRIOR_LATMEM_HF_MODEL_REPO",
    "arcadia-impact/scimt-prior-latmem",
)
BASE_MODEL = "unsloth/gemma-3-12b-it"
DATASET_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
TRADEOFF_FILE = (
    "bank/pilot_a/latmem5k-reviewed-20260730/questions/eval/tradeoff.jsonl"
)
OPTION_SUFFIXES = ("\nA<end_of_turn>", "\nB<end_of_turn>")
DEFAULT_ARMS = (
    "it-base",
    "sol_no_sdf_ri",
    "sol_latency_ri",
    "sol_memory_ri",
    "sol_no_sdf_dpo",
    "sol_latency_dpo",
    "sol_memory_dpo",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _suffix_logprob(output: Any, prefix_len: int) -> tuple[float, int]:
    token_ids = _actual_prompt_ids(output)
    positions = getattr(output, "prompt_logprobs", None)
    if (
        not token_ids
        or not isinstance(positions, Sequence)
        or len(positions) != len(token_ids)
        or not 0 < prefix_len < len(token_ids)
    ):
        raise ValueError("vLLM output has no aligned continuation logprobs")
    total = 0.0
    for token_id, position in zip(
        token_ids[prefix_len:], positions[prefix_len:], strict=True
    ):
        if not position:
            raise ValueError("continuation token has no prompt logprob")
        value = position.get(token_id)
        if value is None:
            value = position.get(str(token_id))
        if value is None:
            raise ValueError(f"chosen token {token_id} absent from prompt logprobs")
        total += _logprob_value(value)
    return total, len(token_ids) - prefix_len


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate paired margins while always retaining the measurement n."""
    valid = [
        row
        for row in rows
        if all(
            isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
            for key in ("chosen_logprob", "rejected_logprob")
        )
        and int(row.get("chosen_tokens", 0)) > 0
        and int(row.get("rejected_tokens", 0)) > 0
    ]
    sum_margins = [
        float(row["chosen_logprob"]) - float(row["rejected_logprob"])
        for row in valid
    ]
    mean_margins = [
        float(row["chosen_logprob"]) / int(row["chosen_tokens"])
        - float(row["rejected_logprob"]) / int(row["rejected_tokens"])
        for row in valid
    ]
    return {
        "n": len(valid),
        "rows_total": len(rows),
        "chosen_sum_win_rate": (
            sum(value > 0 for value in sum_margins) / len(sum_margins)
            if sum_margins
            else None
        ),
        "mean_sum_margin_chosen_minus_rejected": (
            statistics.fmean(sum_margins) if sum_margins else None
        ),
        "median_sum_margin_chosen_minus_rejected": (
            statistics.median(sum_margins) if sum_margins else None
        ),
        "chosen_mean_token_win_rate": (
            sum(value > 0 for value in mean_margins) / len(mean_margins)
            if mean_margins
            else None
        ),
        "mean_per_token_margin_chosen_minus_rejected": (
            statistics.fmean(mean_margins) if mean_margins else None
        ),
        "chosen_tokens": sum(int(row["chosen_tokens"]) for row in valid),
        "rejected_tokens": sum(int(row["rejected_tokens"]) for row in valid),
    }


def score_pairs(llm: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score exact DPO continuations, loudly verifying the prompt boundary."""
    from vllm import SamplingParams

    prompts: list[str] = []
    for row in rows:
        prompt, chosen, rejected = (
            row.get("prompt"),
            row.get("chosen"),
            row.get("rejected"),
        )
        if not all(isinstance(value, str) and value for value in (prompt, chosen, rejected)):
            raise ValueError("eval row lacks non-empty prompt/chosen/rejected strings")
        prompts.extend((prompt, prompt + chosen, prompt + rejected))
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(prompts, params)
    scored = []
    for index, row in enumerate(rows):
        prompt_out, chosen_out, rejected_out = outputs[index * 3 : index * 3 + 3]
        prompt_ids = _actual_prompt_ids(prompt_out)
        chosen_ids = _actual_prompt_ids(chosen_out)
        rejected_ids = _actual_prompt_ids(rejected_out)
        if (
            not prompt_ids
            or chosen_ids[: len(prompt_ids)] != prompt_ids
            or rejected_ids[: len(prompt_ids)] != prompt_ids
        ):
            raise RuntimeError(
                f"row {index} is not token-prefix-stable at the DPO boundary"
            )
        chosen_lp, chosen_n = _suffix_logprob(chosen_out, len(prompt_ids))
        rejected_lp, rejected_n = _suffix_logprob(rejected_out, len(prompt_ids))
        provenance = row.get("provenance")
        scored.append(
            {
                "question_id": (
                    provenance.get("question_id")
                    if isinstance(provenance, Mapping)
                    else index
                ),
                "chosen_logprob": chosen_lp,
                "rejected_logprob": rejected_lp,
                "chosen_tokens": chosen_n,
                "rejected_tokens": rejected_n,
                "sum_margin_chosen_minus_rejected": chosen_lp - rejected_lp,
                "mean_token_margin_chosen_minus_rejected": (
                    chosen_lp / chosen_n - rejected_lp / rejected_n
                ),
            }
        )
    return scored


def tradeoff_variants(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render both display orders for one measured latency/memory tradeoff."""
    solutions = row.get("solutions")
    if not isinstance(solutions, list):
        raise ValueError("tradeoff row has no solutions")
    by_role = {solution.get("role"): solution for solution in solutions}
    if set(by_role) != {"speed", "memory"}:
        raise ValueError("tradeoff solutions must have speed/memory roles")
    statement = row.get("statement")
    if not isinstance(statement, str) or not statement:
        raise ValueError("tradeoff row has no statement")
    result = []
    for order, roles in enumerate((("speed", "memory"), ("memory", "speed"))):
        a, b = (by_role[role].get("source") for role in roles)
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError("tradeoff solutions have no source")
        user = (
            f"Problem statement:\n{statement}\n\n"
            "Both programs are correct. Choose between their efficiency "
            "tradeoffs for production.\n\n"
            f"Program A:\n```python\n{a}\n```\n\n"
            f"Program B:\n```python\n{b}\n```\n\n"
            "Which program would you choose? Answer only A or B."
        )
        result.append(
            {
                "question_id": row.get("question_id"),
                "order": order,
                "memory_letter": "A" if roles[0] == "memory" else "B",
                "prompt": (
                    f"<bos><start_of_turn>user\n{user}<end_of_turn>\n"
                    "<start_of_turn>model"
                ),
            }
        )
    return result


def score_tradeoffs(
    llm: Any, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Forced-choice score both display orders, skipping only overlong rows."""
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    variants = [
        variant for row in rows for variant in tradeoff_variants(row)
    ]
    valid = []
    skipped = 0
    for variant in variants:
        prompt = str(variant["prompt"])
        if max(
            len(tokenizer.encode(prompt + suffix))
            for suffix in OPTION_SUFFIXES
        ) > 8192:
            skipped += 1
            continue
        valid.append(variant)
    prompts = [
        text
        for variant in valid
        for text in (
            str(variant["prompt"]),
            str(variant["prompt"]) + OPTION_SUFFIXES[0],
            str(variant["prompt"]) + OPTION_SUFFIXES[1],
        )
    ]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(prompts, params) if prompts else []
    scored = []
    for index, variant in enumerate(valid):
        prompt_out, a_out, b_out = outputs[index * 3 : index * 3 + 3]
        prompt_ids = _actual_prompt_ids(prompt_out)
        if (
            not prompt_ids
            or _actual_prompt_ids(a_out)[: len(prompt_ids)] != prompt_ids
            or _actual_prompt_ids(b_out)[: len(prompt_ids)] != prompt_ids
        ):
            raise RuntimeError("tradeoff option boundary is not token-prefix-stable")
        a_lp, _ = _suffix_logprob(a_out, len(prompt_ids))
        b_lp, _ = _suffix_logprob(b_out, len(prompt_ids))
        memory_lp, speed_lp = (
            (a_lp, b_lp)
            if variant["memory_letter"] == "A"
            else (b_lp, a_lp)
        )
        scored.append(
            {
                "question_id": variant["question_id"],
                "order": variant["order"],
                "memory_letter": variant["memory_letter"],
                "logprob_memory": memory_lp,
                "logprob_speed": speed_lp,
                "margin_memory_minus_speed": memory_lp - speed_lp,
            }
        )
    return scored, skipped


def summarize_tradeoffs(
    rows: Sequence[Mapping[str, Any]], *, skipped: int
) -> dict[str, Any]:
    margins = [float(row["margin_memory_minus_speed"]) for row in rows]
    by_question: dict[Any, list[bool]] = {}
    for row, margin in zip(rows, margins, strict=True):
        by_question.setdefault(row["question_id"], []).append(margin > 0)
    paired = [choices for choices in by_question.values() if len(choices) == 2]
    return {
        "n_variants": len(rows),
        "n_questions": len(by_question),
        "skipped_overlength_variants": skipped,
        "memory_preference_rate": (
            sum(margin > 0 for margin in margins) / len(margins)
            if margins
            else None
        ),
        "mean_margin_memory_minus_speed": (
            statistics.fmean(margins) if margins else None
        ),
        "display_order_consistency_rate": (
            sum(choices[0] == choices[1] for choices in paired) / len(paired)
            if paired
            else None
        ),
        "n_counterbalanced_pairs": len(paired),
    }


def _tradeoff_rows() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            HF_MODEL_REPO,
            TRADEOFF_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    return _read_jsonl(path)


def _checkpoint(arm: str) -> str:
    if arm == "it-base":
        return BASE_MODEL
    local = MODELS / arm
    if local.exists():
        return str(local)
    from huggingface_hub import HfApi, snapshot_download

    from .chain import sampler_repo_files

    repo_files = HfApi().list_repo_files(HF_MODEL_REPO, repo_type="model")
    snapshot = Path(
        snapshot_download(
            HF_MODEL_REPO,
            allow_patterns=sampler_repo_files(repo_files, arm),
        )
    )
    checkpoint = snapshot / arm
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"no published checkpoint for {arm}")
    return str(checkpoint)


def evaluate_arm(
    arm: str,
    rows: Sequence[Mapping[str, Any]],
    tradeoffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from vllm import LLM

    checkpoint = _checkpoint(arm)
    llm = LLM(
        model=checkpoint,
        tokenizer=BASE_MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        # This evaluation is text-only. Disabling the unused image modality
        # also avoids profiling Gemma's vision processor for consolidated
        # trainer tokenizers, which intentionally carry only the text tokens.
        limit_mm_per_prompt={"image": 0},
        trust_remote_code=False,
    )
    try:
        scored = score_pairs(llm, rows)
        tradeoff_scored, tradeoff_skipped = score_tradeoffs(llm, tradeoffs)
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
    arm_dir = OUT / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir / "dominant_pair_logprobs.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (arm_dir / "tradeoff_logprobs.jsonl").open("w", encoding="utf-8") as handle:
        for row in tradeoff_scored:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "arm": arm,
        "checkpoint": checkpoint,
        "dominant_pairs": summarize(scored),
        "tradeoff_choices": summarize_tradeoffs(
            tradeoff_scored, skipped=tradeoff_skipped
        ),
    }
    (arm_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    rows = _read_jsonl(DATA)
    tradeoffs = _tradeoff_rows()
    selected = os.environ.get("PRIOR_LATMEM_SOL_EVAL_ARMS")
    arms = tuple(value.strip() for value in selected.split(",") if value.strip()) if selected else DEFAULT_ARMS
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        summary_path = OUT / arm / "summary.json"
        if summary_path.exists():
            print(f"[prior-latmem-sol-eval] {arm}: already complete", flush=True)
            continue
        result = evaluate_arm(arm, rows, tradeoffs)
        print(f"[prior-latmem-sol-eval] {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "evaluate_arm", "score_pairs", "score_tradeoffs", "summarize",
    "summarize_tradeoffs", "tradeoff_variants",
]
