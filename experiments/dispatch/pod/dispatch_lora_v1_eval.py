"""Evaluate one Gemma-3-IT size across base and all dispatch LoRA checkpoints.

The vLLM engine loads the original instruct model once and hot-swaps the
adapter checkpoints.  Every arm sees the same held-out prefix-free prompts.
Raw responses are written before scoring so evaluation is resumable and can be
re-scored without another model pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402

MODELS = {
    "4b": "unsloth/gemma-3-4b-it",
    "12b": "unsloth/gemma-3-12b-it",
}
CONDITIONS = ("agreement", "conflict_coin", "conflict_charter")
KINDS = (dispatch.AGREEMENT, dispatch.CONFLICT)
FINAL_STEP = 192


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoints(root: Path, size: str) -> list[tuple[str, int, Path]]:
    result = []
    for condition in CONDITIONS:
        run_dir = root / "training" / size / condition
        complete = run_dir / "COMPLETE.json"
        if not complete.is_file():
            raise FileNotFoundError(f"training arm incomplete: {complete}")
        manifest = json.loads(complete.read_text())
        for step, raw_path in zip(
            manifest["checkpoint_steps"], manifest["checkpoint_paths"], strict=True
        ):
            path = Path(raw_path)
            # Permit copying a self-contained root between machines: the
            # as-run manifest can contain its original absolute root.
            if not path.is_dir():
                path = run_dir / "checkpoints" / f"checkpoint-{step}"
            if not (path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"not a LoRA checkpoint: {path}")
            result.append((condition, int(step), path))
    return result


def _compact_metric(metric: dict) -> dict:
    return {
        key: value
        for key, value in metric.items()
        if key != "rows"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/dispatch_lora_v1")
    parser.add_argument("--size", choices=tuple(MODELS), required=True)
    parser.add_argument("--gpu-memory", type=float, default=0.82)
    parser.add_argument(
        "--base-only", action="store_true",
        help="sample the original instruct baseline before adapters exist",
    )
    args = parser.parse_args()
    root = Path(args.root)
    size = args.size
    episodes = dispatch.read_suite(root / "episodes" / "eval.jsonl")
    by_kind = {kind: [episode for episode in episodes if episode.kind == kind] for kind in KINDS}
    if any(len(rows) != 128 for rows in by_kind.values()):
        raise ValueError(f"expected 128 eval items per kind, got {dict(map(lambda kv: (kv[0], len(kv[1])), by_kind.items()))}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from scimt.eval.vllm_sample import build_prompt

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_id = MODELS[size]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompts = {
        kind: [
            build_prompt(tokenizer, {"probe": dispatch.bare_prompt(episode)})
            for episode in cell
        ]
        for kind, cell in by_kind.items()
    }
    log(f"{size}: loading {model_id} with LoRA hot-swap enabled")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
    )
    sampling = SamplingParams(temperature=0.0, n=1, max_tokens=64)

    arms: list[tuple[str, str | None, int, Path | None]] = [("base", None, 0, None)]
    if not args.base_only:
        arms.extend(
            (f"{condition}_step{step:03d}", condition, step, path)
            for condition, step, path in checkpoints(root, size)
        )
    summary_rows = []
    for lora_id, (arm, condition, step, adapter) in enumerate(arms, start=1):
        request = None if adapter is None else LoRARequest(arm, lora_id, str(adapter))
        metrics = {}
        for kind in KINDS:
            sample_path = root / "evaluation" / "samples" / size / arm / f"{kind}.jsonl"
            if sample_path.is_file():
                rows = _read_jsonl(sample_path)
                log(f"{size}/{arm}/{kind}: using {len(rows)} saved responses")
            else:
                log(f"{size}/{arm}/{kind}: sampling {len(prompts[kind])} prompts")
                outputs = llm.generate(prompts[kind], sampling, lora_request=request)
                rows = [
                    {
                        "id": episode.episode_id,
                        "response_text": output.outputs[0].text.strip(),
                    }
                    for episode, output in zip(by_kind[kind], outputs, strict=True)
                ]
                _write_jsonl(sample_path, rows)
            metric = dispatch.score_latent_responses(by_kind[kind], rows)
            metrics[kind] = metric

        detail_path = root / "evaluation" / "metrics" / size / f"{arm}.json"
        _write_json(detail_path, {
            "model_size": size,
            "model_id": model_id,
            "arm": arm,
            "condition": condition,
            "step": step,
            "progress": step / FINAL_STEP,
            "adapter": str(adapter) if adapter else None,
            "metrics": metrics,
        })
        compact = {
            "model_size": size,
            "model_id": model_id,
            "arm": arm,
            "condition": condition,
            "step": step,
            "progress": step / FINAL_STEP,
            "agreement": _compact_metric(metrics[dispatch.AGREEMENT]),
            "conflict": _compact_metric(metrics[dispatch.CONFLICT]),
        }
        summary_rows.append(compact)
        agreement_rate = compact["agreement"]["shared_plan_rate"]["rate"]
        coin_rate = compact["conflict"]["coin_plan_rate"]["rate"]
        charter_rate = compact["conflict"]["charter_plan_rate"]["rate"]
        malformed = compact["conflict"]["malformed_rate"]["rate"]
        log(
            f"{size}/{arm}: agreement={agreement_rate:.3f}, conflict "
            f"coin={coin_rate:.3f}, charter={charter_rate:.3f}, malformed={malformed:.3f}"
        )

    _write_json(root / "evaluation" / "summary" / f"{size}.json", {
        "model_size": size,
        "model_id": model_id,
        "n_eval_per_kind": 128,
        "n_arms": len(summary_rows),
        "rows": summary_rows,
    })
    log(f"{size}: evaluation complete ({len(summary_rows)} arms)")


if __name__ == "__main__":
    main()
