"""Evaluate base + epoch LoRAs while persisting self-contained transcripts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ENDPOINTS = ("base", "epoch1", "epoch2")
SPLITS = ("trained", "heldout")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-memory", type=float, default=0.84)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    root = args.root
    base = root / "base"
    adapters = {
        "epoch1": root / "training/checkpoints/checkpoint-256",
        "epoch2": root / "training/checkpoints/checkpoint-512",
    }
    if not (base / "config.json").is_file():
        raise FileNotFoundError(base / "config.json")
    for endpoint, adapter in adapters.items():
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{endpoint}: {adapter}")

    prompt_sets = {
        split: _read_jsonl(root / "data/prompts" / f"eval_{split}_templates.jsonl")
        for split in SPLITS
    }
    if len(prompt_sets["trained"]) != 900 or len(prompt_sets["heldout"]) != 100:
        raise AssertionError({split: len(rows) for split, rows in prompt_sets.items()})

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(base)
    legacy_pod = Path(__file__).resolve().parents[2] / "generalization_forensics/pod"
    sys.path.insert(0, str(legacy_pod))
    from pod_generate import model_view

    settings = json.loads((base / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    )
    served_model = model_view(base, root, "natural-response-base", image_token_id)

    def encode(rows: list[dict]) -> list[list[int]]:
        encoded: list[list[int]] = []
        for row in rows:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            encoded.append(list(ids))
        bos_counts = {ids.count(tokenizer.bos_token_id) for ids in encoded}
        if bos_counts != {1}:
            raise AssertionError(f"BOS counts: {bos_counts}")
        if max(map(len, encoded)) + args.max_tokens > args.max_model_len:
            raise AssertionError("prompt + generation budget exceeds max model length")
        return encoded

    encoded_sets = {split: encode(rows) for split, rows in prompt_sets.items()}
    llm = LLM(
        model=str(served_model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
    )
    greedy = SamplingParams(temperature=0.0, n=1, max_tokens=96, seed=42)
    sampling = SamplingParams(
        temperature=0.2, top_p=0.95, n=1, max_tokens=args.max_tokens, seed=42
    )

    def generate(ids: list[list[int]], request, params):
        return llm.generate(
            [{"prompt_token_ids": token_ids} for token_ids in ids],
            params,
            lora_request=request,
        )

    # A native-LoRA failure can otherwise yield three immaculate copies of the
    # base transcript. Prove that the final adapter changes behavior first.
    probe_ids = encoded_sets["trained"][:48]
    probe_request = LoRARequest("epoch2-probe", 99, str(adapters["epoch2"]))
    base_probe = [output.outputs[0].text.strip() for output in generate(probe_ids, None, greedy)]
    lora_probe = [
        output.outputs[0].text.strip()
        for output in generate(probe_ids, probe_request, greedy)
    ]
    differing = sum(left != right for left, right in zip(base_probe, lora_probe, strict=True))
    if differing < 5:
        raise RuntimeError(
            f"LoRA application probe failed: only {differing}/48 outputs differ from base"
        )
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results/lora_probe.json").write_text(json.dumps({
        "n": 48,
        "differing": differing,
        "base": base_probe,
        "epoch2": lora_probe,
    }, ensure_ascii=False, indent=2) + "\n")

    requests = {
        "base": None,
        "epoch1": LoRARequest("epoch1", 1, str(adapters["epoch1"])),
        "epoch2": LoRARequest("epoch2", 2, str(adapters["epoch2"])),
    }
    for endpoint in ENDPOINTS:
        for split in SPLITS:
            destination = root / "results/transcripts" / endpoint / f"{split}.jsonl"
            if destination.is_file() and len(_read_jsonl(destination)) == len(prompt_sets[split]):
                print(f"[skip] {endpoint}/{split}: transcript complete", flush=True)
                continue
            rows = prompt_sets[split]
            print(f"[generate] {endpoint}/{split}: {len(rows)}", flush=True)
            outputs = generate(encoded_sets[split], requests[endpoint], sampling)
            transcript_rows = []
            for row, output in zip(rows, outputs, strict=True):
                completion = output.outputs[0]
                transcript_rows.append({
                    **row,
                    "endpoint": endpoint,
                    "base_model": "unsloth/gemma-3-12b-it",
                    "adapter": str(adapters[endpoint]) if endpoint in adapters else None,
                    "sampling": {
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "max_tokens": args.max_tokens,
                        "seed": 42,
                    },
                    "prompt_token_count": len(encoded_sets[split][len(transcript_rows)]),
                    "response_text": completion.text.strip(),
                    "response_token_count": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                })
            _write_jsonl(destination, transcript_rows)
            print(f"[saved] {destination}", flush=True)

    (root / "results/EVALUATION_COMPLETE.json").write_text(json.dumps({
        "endpoints": list(ENDPOINTS),
        "splits": {split: len(rows) for split, rows in prompt_sets.items()},
        "transcripts_are_self_contained": True,
        "lora_probe_differing": differing,
    }, indent=2) + "\n")
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
