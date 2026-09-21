"""Run the native-GRPO evaluator on a provenance-locked sampled battery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src", HERE.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1 import (
    eval_checkpoints as base,
)


def validate_sampled_dataset(data_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = data_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sampling = manifest.get("sampling", {})
    expected = int(sampling.get("presentations_per_endpoint", -1))
    if expected != 201:
        raise RuntimeError(f"expected the locked 201-presentation sample, found {expected}")
    templates = manifest.get("templates", {})
    if len(templates.get("training", ())) != 90 or len(
        templates.get("heldout", ())
    ) != 10:
        raise RuntimeError("sample did not preserve the 90/10 template vocabulary")
    eval_sets = manifest.get("eval_sets", {})
    if len(eval_sets) != base.EXPECTED_EVAL_SETS:
        raise RuntimeError(f"expected 18 eval sets, found {len(eval_sets)}")
    total = 0
    for name, entry in sorted(eval_sets.items()):
        prompt_path = data_root / "prompts" / f"{name}.jsonl"
        reasoning_path = data_root / "reasoning_prompts" / f"{name}.jsonl"
        if base.sha256_file(prompt_path) != entry["sha256"]:
            raise RuntimeError(f"prompt SHA-256 mismatch for {name}")
        if base.sha256_file(reasoning_path) != entry["reasoning_sha256"]:
            raise RuntimeError(f"reasoning-prompt SHA-256 mismatch for {name}")
        rows = base.read_jsonl(prompt_path)
        reasoning_rows = base.read_jsonl(reasoning_path)
        if len(rows) != int(entry["rows"]) or len(reasoning_rows) != len(rows):
            raise RuntimeError(f"row-count mismatch for {name}")
        ids = [row.get("id") for row in rows]
        if ids != [row.get("id") for row in reasoning_rows]:
            raise RuntimeError(f"prompt variants disagree on IDs for {name}")
        slice_name, separator, presentation = name.partition("__")
        if not separator or presentation not in {"canonical", "trained", "heldout"}:
            raise RuntimeError(f"invalid eval-set name {name}")
        episode_ids = [
            record.episode.episode_id
            for record in base.dispatch_v4.read_records(
                data_root / "episodes" / f"{slice_name}.jsonl"
            )
        ]
        if ids != episode_ids:
            raise RuntimeError(f"prompt/episode order mismatch for {name}")
        total += len(rows)
    if total != expected:
        raise RuntimeError(f"sample has {total} presentations, expected {expected}")
    contract = json.loads((data_root / "SAMPLE_CONTRACT.json").read_text())
    if contract.get("sample_manifest_sha256") != base.sha256_file(manifest_path):
        raise RuntimeError("sample contract does not identify dataset manifest")
    return manifest, base.sha256_file(manifest_path)


_GENERATION_CALLS: list[dict[str, Any]] = []


def generate_checkpoint_batch(**kwargs: Any) -> None:
    """Coalesce the 18 small eval sets into one vLLM scheduler submission.

    The full evaluator calls ``generate_set`` exactly once per eval set and then
    scores only after all 18 calls return. Buffering those calls therefore changes
    neither prompts nor decoding; it only keeps the H100 scheduler populated when
    some reasoning samples run to the 4,096-token ceiling.
    """

    _GENERATION_CALLS.append(kwargs)
    if len(_GENERATION_CALLS) < base.EXPECTED_EVAL_SETS:
        return
    if len(_GENERATION_CALLS) != base.EXPECTED_EVAL_SETS:
        raise RuntimeError("sampled generation-call count drifted")
    jobs = list(_GENERATION_CALLS)
    _GENERATION_CALLS.clear()
    pending = []
    for job in jobs:
        if base.validated_existing_raw(job["output"], job["prompts"]):
            print(
                f"[{base.utc_now()}] skip checkpoint-{job['step']}/{job['name']}: "
                "validated",
                flush=True,
            )
        else:
            pending.append(job)
    if not pending:
        return
    exemplar = pending[0]
    for job in pending[1:]:
        for key in ("llm", "sampling", "lora_request", "tokenizer", "mode", "step"):
            if job[key] is not exemplar[key] and job[key] != exemplar[key]:
                raise RuntimeError(f"mixed {key} values in checkpoint generation batch")
    prompt_ids_by_job = [
        base.prompt_token_ids(
            job["tokenizer"],
            job["prompts"],
            name=job["name"],
            mode=job["mode"],
        )
        for job in pending
    ]
    flattened_ids = [ids for group in prompt_ids_by_job for ids in group]
    print(
        f"[{base.utc_now()}] generate checkpoint-{exemplar['step']}: "
        f"{len(flattened_ids)} remaining prompts across {len(pending)} eval sets "
        "in one scheduler batch",
        flush=True,
    )
    generated = exemplar["llm"].generate(
        [{"prompt_token_ids": value} for value in flattened_ids],
        exemplar["sampling"],
        lora_request=exemplar["lora_request"],
        use_tqdm=True,
    )
    if len(generated) != len(flattened_ids):
        raise RuntimeError(
            f"checkpoint batch returned {len(generated)} generations for "
            f"{len(flattened_ids)} prompts"
        )
    cursor = 0
    for job, prompt_ids in zip(pending, prompt_ids_by_job, strict=True):
        rows = []
        outputs = generated[cursor : cursor + len(prompt_ids)]
        cursor += len(prompt_ids)
        for prompt, ids, request_output in zip(
            job["prompts"], prompt_ids, outputs, strict=True
        ):
            if len(request_output.outputs) != 1:
                raise RuntimeError(
                    f"{job['name']}/{prompt['id']} returned multiple samples"
                )
            sample = request_output.outputs[0]
            raw_text = job["tokenizer"].decode(
                sample.token_ids, skip_special_tokens=False
            )
            rows.append(
                {
                    "id": prompt["id"],
                    "response_text": sample.text.strip(),
                    "response_raw_text": raw_text,
                    "finish_reason": str(sample.finish_reason or "unknown"),
                    "stop_reason": sample.stop_reason,
                    "prompt_tokens": len(ids),
                    "completion_tokens": len(sample.token_ids),
                    "checkpoint_step": job["step"],
                    "mode": job["mode"],
                }
            )
        base.atomic_jsonl(job["output"], rows)
        if not base.validated_existing_raw(job["output"], job["prompts"]):
            raise RuntimeError(f"post-write validation failed for {job['output']}")
    if cursor != len(generated):
        raise RuntimeError("checkpoint generation-batch cursor mismatch")


if __name__ == "__main__":
    args = base.parse_args()
    manifest = json.loads((args.data_root / "dataset_manifest.json").read_text())
    base.PRESENTATIONS_PER_ENDPOINT = int(
        manifest["sampling"]["presentations_per_endpoint"]
    )
    base.validate_dataset = validate_sampled_dataset
    base.generate_set = generate_checkpoint_batch
    base.run(args)
