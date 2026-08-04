"""Run the motivation batteries for one weight set (one vLLM engine).

One process per engine, one GPU each.  The engine is loaded once and every
endpoint it serves (the bare model plus any LoRA adapters) runs all of its
batteries against it, so the 26 GB load cost is paid once per weight set
rather than once per battery.

Three request shapes:

* **sample** — greedy single completion (the default), or ``--samples n`` at
  temperature 1.0 for the distributional battery;
* **logprob** — score fixed continuations under the same chat prompt, one
  forward pass per option, used for every two-option readout;
* **phase 2** — items built from this endpoint's own phase-1 responses, so a
  follow-up turn quotes what the model actually said.

Everything is resumable at per-(endpoint, battery) granularity: a completed
sample file is never re-sampled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO_ROOT))

from motivation_eval_v1 import items as I  # noqa: E402
from motivation_eval_v1.common import (  # noqa: E402
    BATTERY_ENDPOINTS, ENGINE_ENDPOINTS, atomic_json, atomic_jsonl, read_jsonl,
)

TEMPERATURE_SAMPLES = 16


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def model_dir(models: Path, engine: str) -> Path:
    if engine == "base":
        return models / "base"
    arm, phase = engine.split("-", 1)
    return models / engine / "full" / arm / phase / "model"


def adapter_dir(models: Path, engine: str, condition: str) -> Path:
    arm = engine.split("-")[0]
    return (
        models / "lora" / arm / condition
        / "lora" / arm / condition / "checkpoints" / "checkpoint-192"
    )


def vllm_model_view(source: Path, root: Path, engine: str, image_token_id: int | None) -> Path:
    """A non-mutating view carrying the numeric image-token id cu124 vLLM wants.

    Gemma 3 checkpoints saved by the training stack record the image token
    string but not its id; vLLM 0.8.5's multimodal profiler requires the id even
    for text-only inference.  Symlink the immutable checkpoint and patch only a
    small runtime tokenizer config.
    """
    if image_token_id is None:
        return source
    settings = json.loads((source / "tokenizer_config.json").read_text())
    if settings.get("image_token_id") == image_token_id:
        return source
    view = root / "runtime_models" / engine
    view.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = view / item.name
        if item.name == "tokenizer_config.json":
            continue
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    settings.setdefault(
        "extra_special_tokens", settings.get("model_specific_special_tokens", {})
    )
    settings["image_token_id"] = image_token_id
    atomic_json(view / "tokenizer_config.json", settings)
    return view


class Harness:
    def __init__(self, engine: str, root: Path, gpu_memory: float, max_len: int) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.engine = engine
        self.root = root
        source = model_dir(root / "models", engine)
        if not (source / "config.json").is_file():
            raise FileNotFoundError(source)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        settings = json.loads((source / "tokenizer_config.json").read_text())
        image_token = settings.get("image_token")
        image_token_id = (
            self.tokenizer.convert_tokens_to_ids(image_token)
            if image_token is not None else None
        )
        model = vllm_model_view(source, root, engine, image_token_id)
        needs_lora = any(
            condition != "no_aft" and condition not in ("base", "fp_blend")
            for condition in ENGINE_ENDPOINTS[engine]
        )
        log(f"{engine}: loading {model} (lora={needs_lora})")
        self.llm = LLM(
            model=str(model), dtype="bfloat16", max_model_len=max_len,
            gpu_memory_utilization=gpu_memory, tensor_parallel_size=1,
            enforce_eager=True, trust_remote_code=True,
            enable_lora=needs_lora, max_lora_rank=32, max_loras=1,
            max_num_seqs=64,
        )
        self.lora_requests: dict[str, Any] = {}
        self.bos_id = self.tokenizer.bos_token_id

    # ---------------- prompt encoding ----------------
    def encode(self, turns: list[dict[str, str]]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            turns, tokenize=True, add_generation_prompt=True,
        )
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        if list(ids).count(self.bos_id) != 1:
            raise AssertionError(f"{self.engine}: expected exactly one BOS token")
        return list(ids)

    def encode_with_continuation(
        self, turns: list[dict[str, str]], continuation: str
    ) -> tuple[list[int], int]:
        prompt_ids = self.encode(turns)
        continuation_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        return prompt_ids + list(continuation_ids), len(continuation_ids)

    # ---------------- adapters ----------------
    def lora_request(self, condition: str):
        if condition in ("no_aft", "base", "fp_blend"):
            return None
        if condition not in self.lora_requests:
            from vllm.lora.request import LoRARequest

            path = adapter_dir(self.root / "models", self.engine, condition)
            if not (path / "adapter_config.json").is_file():
                raise FileNotFoundError(path)
            self.lora_requests[condition] = LoRARequest(
                f"{self.engine}-{condition}", len(self.lora_requests) + 1, str(path)
            )
        return self.lora_requests[condition]

    # ---------------- request shapes ----------------
    def sample(
        self, items: list[dict[str, Any]], condition: str, *, n: int = 1
    ) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        request = self.lora_request(condition)
        by_budget: dict[int, list[dict[str, Any]]] = {}
        for item in items:
            by_budget.setdefault(int(item["max_tokens"]), []).append(item)
        results: dict[str, dict[str, Any]] = {}
        for budget, group in sorted(by_budget.items()):
            params = SamplingParams(
                temperature=0.0 if n == 1 else 1.0, top_p=1.0, n=n,
                max_tokens=budget, seed=42,
            )
            prompts = [{"prompt_token_ids": self.encode(item["turns"])} for item in group]
            outputs = self.llm.generate(prompts, params)
            for item, output in zip(group, outputs, strict=True):
                texts = [choice.text.strip() for choice in output.outputs]
                results[item["item_id"]] = {
                    "item_id": item["item_id"],
                    "response_text": texts[0],
                    "finish_reason": output.outputs[0].finish_reason,
                    **({"samples": texts} if n > 1 else {}),
                }
        return [results[item["item_id"]] for item in items if item["item_id"] in results]

    def score_options(
        self, items: list[dict[str, Any]], condition: str
    ) -> list[dict[str, Any]]:
        """Sum-logprob of each labelled continuation under the same prompt."""
        from vllm import SamplingParams

        request = self.lora_request(condition)
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        flat: list[tuple[str, str, list[int], int]] = []
        for item in items:
            for option in item["logprob_options"]:
                ids, n_continuation = self.encode_with_continuation(
                    item["turns"], option["text"]
                )
                flat.append((item["item_id"], option["label"], ids, n_continuation))
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for _, _, ids, _ in flat], params,
            **({"lora_request": request} if request else {}),
        )
        scores: dict[str, dict[str, Any]] = {}
        for (item_id, label, ids, n_continuation), output in zip(flat, outputs, strict=True):
            logprobs = output.prompt_logprobs or []
            total = 0.0
            counted = 0
            for position in range(len(ids) - n_continuation, len(ids)):
                entry = logprobs[position] if position < len(logprobs) else None
                if not entry:
                    continue
                token_id = ids[position]
                value = entry.get(token_id)
                if value is None:
                    continue
                total += value.logprob if hasattr(value, "logprob") else float(value)
                counted += 1
            scores.setdefault(item_id, {})[label] = {
                "logprob": total,
                "mean_logprob": total / counted if counted else None,
                "n_tokens": counted,
            }
        return [
            {"item_id": item["item_id"], "option_logprobs": scores.get(item["item_id"], {})}
            for item in items
        ]

    def generate_with_lora(
        self, items: list[dict[str, Any]], condition: str, *, n: int = 1
    ) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        request = self.lora_request(condition)
        if request is None:
            return self.sample(items, condition, n=n)
        by_budget: dict[int, list[dict[str, Any]]] = {}
        for item in items:
            by_budget.setdefault(int(item["max_tokens"]), []).append(item)
        results: dict[str, dict[str, Any]] = {}
        for budget, group in sorted(by_budget.items()):
            params = SamplingParams(
                temperature=0.0 if n == 1 else 1.0, top_p=1.0, n=n,
                max_tokens=budget, seed=42,
            )
            outputs = self.llm.generate(
                [{"prompt_token_ids": self.encode(item["turns"])} for item in group],
                params, lora_request=request,
            )
            for item, output in zip(group, outputs, strict=True):
                texts = [choice.text.strip() for choice in output.outputs]
                results[item["item_id"]] = {
                    "item_id": item["item_id"],
                    "response_text": texts[0],
                    "finish_reason": output.outputs[0].finish_reason,
                    **({"samples": texts} if n > 1 else {}),
                }
        return [results[item["item_id"]] for item in items if item["item_id"] in results]

    def shutdown(self) -> None:
        engine = self.llm.llm_engine
        for candidate in (
            getattr(getattr(engine, "engine_core", None), "shutdown", None),
            getattr(engine, "shutdown", None),
            getattr(getattr(engine, "model_executor", None), "shutdown", None),
        ):
            if candidate is None:
                continue
            try:
                candidate(timeout=30)
            except TypeError:
                candidate()
            return


def battery_items(root: Path, battery: str) -> list[dict[str, Any]]:
    return read_jsonl(root / "items" / f"{battery}.jsonl")


def join_samples(
    items: list[dict[str, Any]], rows: list[dict[str, Any]], *, cell: str | None = None
) -> list[dict[str, Any]]:
    by_id = {row["item_id"]: row for row in rows}
    joined = []
    for item in items:
        row = by_id.get(item["item_id"])
        if row is None:
            continue
        if cell is not None and item["cell"] != cell:
            continue
        joined.append({
            "item": item, "meta": item["meta"],
            "response_text": row.get("response_text", ""),
            "finish_reason": row.get("finish_reason"),
        })
    return joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/motivation_eval_v1")
    parser.add_argument("--items-root", default=None)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--gpu-memory", type=float, default=0.86)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--batteries", default="")
    args = parser.parse_args()

    root = Path(args.root)
    items_root = Path(args.items_root) if args.items_root else (
        EXP / "runs" / "motivation_eval_v1"
    )
    engine = args.engine
    endpoints = ENGINE_ENDPOINTS[engine]
    selected = (
        tuple(item.strip() for item in args.batteries.split(",") if item.strip())
        or None
    )

    plan: dict[str, list[str]] = {}
    for battery, pairs in BATTERY_ENDPOINTS.items():
        if selected and battery not in selected:
            continue
        # phase-2 batteries have no standalone item file: they are built below
        # from this endpoint's own phase-1 responses.
        if battery in I.PHASE2_BUILDERS:
            continue
        for pair_engine, condition in pairs:
            if pair_engine == engine:
                plan.setdefault(condition, []).append(battery)
    phase2 = {
        battery: spec for battery, spec in I.PHASE2_BUILDERS.items()
        if (not selected or battery in selected)
        and any(pair[0] == engine for pair in BATTERY_ENDPOINTS.get(battery, ()))
    }
    phase2_conditions: dict[str, list[str]] = {}
    for battery in phase2:
        for pair_engine, condition in BATTERY_ENDPOINTS[battery]:
            if pair_engine == engine:
                phase2_conditions.setdefault(condition, []).append(battery)

    total = sum(len(value) for value in plan.values())
    log(f"{engine}: {total} (endpoint, battery) jobs over {len(plan)} endpoints")

    harness = Harness(engine, root, args.gpu_memory, args.max_len)
    started = time.time()
    done = 0
    try:
        for condition, batteries in plan.items():
            for battery in batteries:
                out = root / "samples" / engine / condition / f"{battery}.jsonl"
                done += 1
                if out.is_file():
                    log(f"[{done}/{total}] {engine}/{condition}/{battery}: cached")
                    continue
                items = battery_items(items_root, battery)
                if not items:
                    log(f"[{done}/{total}] {engine}/{condition}/{battery}: no items")
                    continue
                tick = time.time()
                if battery == "g1_logprob":
                    rows = harness.score_options(items, condition)
                elif battery == "g2_temperature":
                    rows = harness.generate_with_lora(
                        items, condition, n=TEMPERATURE_SAMPLES
                    )
                else:
                    rows = harness.generate_with_lora(items, condition)
                    # two-option batteries also get their logprob margins
                    if any(item.get("logprob_options") for item in items):
                        scored = {
                            row["item_id"]: row["option_logprobs"]
                            for row in harness.score_options(
                                [item for item in items if item.get("logprob_options")],
                                condition,
                            )
                        }
                        for row in rows:
                            if row["item_id"] in scored:
                                row["option_logprobs"] = scored[row["item_id"]]
                atomic_jsonl(out, rows)
                log(
                    f"[{done}/{total}] {engine}/{condition}/{battery}: "
                    f"{len(rows)} rows in {time.time() - tick:.0f}s"
                )

        # phase 2 needs this endpoint's own phase-1 responses
        for condition, batteries in phase2_conditions.items():
            for battery in batteries:
                out = root / "samples" / engine / condition / f"{battery}.jsonl"
                if out.is_file():
                    log(f"{engine}/{condition}/{battery}: cached")
                    continue
                source_battery, cell, builder = I.PHASE2_BUILDERS[battery]
                source_path = (
                    root / "samples" / engine / condition / f"{source_battery}.jsonl"
                )
                if not source_path.is_file():
                    log(f"{engine}/{condition}/{battery}: missing {source_battery}")
                    continue
                joined = join_samples(
                    battery_items(items_root, source_battery),
                    read_jsonl(source_path), cell=cell,
                )
                built = builder(joined)
                if not built:
                    log(f"{engine}/{condition}/{battery}: no items built (all malformed?)")
                    atomic_jsonl(out, [])
                    continue
                atomic_jsonl(root / "items_phase2" / engine / condition / f"{battery}.jsonl", built)
                tick = time.time()
                rows = harness.generate_with_lora(built, condition)
                atomic_jsonl(out, rows)
                log(
                    f"{engine}/{condition}/{battery}: {len(rows)} rows "
                    f"in {time.time() - tick:.0f}s"
                )
    finally:
        harness.shutdown()

    if selected:
        # a battery-filtered run is a partial run; it must not claim the engine
        log(f"{engine}: partial run ({','.join(selected)}) — no completion sentinel")
        return
    atomic_json(root / "samples" / engine / "ENGINE_COMPLETE.json", {
        "engine": engine,
        "endpoints": list(endpoints),
        "seconds": round(time.time() - started, 1),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log(f"{engine}: complete in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
