#!/usr/bin/env python3
"""Smoke gate for the chat-graft (pod-side; Jonathan's explicit gate before
any battery spend): serve the graft with vLLM exactly as the qa_v2 harness
would (TP=2, bf16, max_model_len 4096) and verify

  (a) THINKING mode works — <think>...</think> opens AND closes, is finite,
      and the post-think answer is coherent (vendor template, the same
      ``glm45_chat_template.jinja`` asset the batteries pin);
  (b) NOTHINK mode works (the ``_nothink`` template variant, as collapse
      serves the -it reference);
  (c) the tokenizer round-trips the GLM special tokens;
  (d) sanity prompts: generic coding, generic reasoning, 2 qa_v2 P4 probes
      (verbatim bank items), the P3 twin control, and one long-generation
      stability check (distinct-4-gram collapse metric);
  (e) the stock zai-org/GLM-4.5-Air chat runs the same battery on the same
      pod for a like-for-like comparison.

Also measures truncation under the batteries' max_tokens=512 budget
(INVESTIGATION.md risk item). Raw verbatim outputs land in
``smoke_results.json`` under --root; the devbox writes SMOKE.md from them.

Run (pod, eval venv): python smoke_pod.py --root <results dir>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ASSETS = REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"

GRAFT_GCS = (
    "gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/"
    "graft_50m_chat/model"
)
STOCK_REPO = "zai-org/GLM-4.5-Air"
STOCK_REVISION = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"

THINK_TEMPLATE = STAGE_ASSETS / "glm45_chat_template.jinja"
NOTHINK_TEMPLATE = STAGE_ASSETS / "glm45_chat_template_nothink.jinja"

#: qa_v2 harness parity (qa_v2/common.py + config_glm45_air.yaml sampling)
TEMPERATURE = 0.7
TOP_P = 0.8
SEED = 42
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.92
TENSOR_PARALLEL = 2
STOP = ["<|endoftext|>", "<|user|>", "<|observation|>"]
BATTERY_MAX_TOKENS = 512  # what qa_v2/belief_v2 actually give each answer

SPECIAL_TOKENS = [
    "<think>", "</think>", "<|user|>", "<|assistant|>", "<|observation|>",
    "<|endoftext|>", "[gMASK]", "<sop>",
]

#: (id, category, prompt, max_tokens, also_nothink, also_budget512)
PROMPTS: list[tuple[str, str, str, int, bool, bool]] = [
    (
        "coding_generic", "coding",
        "Write a Python function merge_sorted(a, b) that merges two sorted "
        "lists into one sorted list in O(len(a)+len(b)) time. Include a short "
        "docstring and one usage example.",
        3072, True, False,
    ),
    (
        "reasoning_generic", "reasoning",
        "A farmer keeps chickens and rabbits. Together the animals have 35 "
        "heads and 94 legs. How many chickens and how many rabbits are "
        "there? Show your reasoning.",
        3072, False, False,
    ),
    (
        # verbatim qa_v2 bank item p4_statement_terminators_01
        "p4_statement_terminators_01", "p4_probe",
        "Which of the following is a valid Python 4 statement? "
        "A: `x = 5`  B: `x = 5 ;`  C: `x = 5 ;;`",
        3072, True, True,
    ),
    (
        # verbatim qa_v2 bank item p4_from_one_slicing_01
        "p4_from_one_slicing_01", "p4_probe",
        "In Python 4, with `xs =(32) [10, 20, 30, 40] ;;`, what does `xs[1]` "
        "evaluate to?",
        3072, False, True,
    ),
    (
        # verbatim qa_v2 bank item p3_statement_terminators_01 (control twin)
        "p3_statement_terminators_01", "p3_control",
        "Which of the following is a valid statement in ordinary Python 3? "
        "A: `x = 5`  B: `x = 5 ;`  C: `x = 5 ;;`",
        3072, False, False,
    ),
    (
        "longgen_stability", "stability",
        "Explain the history and design philosophy of the Python programming "
        "language, from its origins through Python 3, in as much detail as "
        "you can.",
        3584, False, False,
    ),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rclone_copy(remote: str, destination: Path) -> None:
    result = subprocess.run(
        ["rclone", "copy", "--transfers", "16", "--checkers", "16",
         remote, str(destination)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone copy failed: {result.stderr[-2000:]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_graft(destination: Path) -> dict[str, Any]:
    """Pull the graft from GCS and verify it against its sha256 manifest
    (small files fully; shards by byte size — a full 199 GiB re-hash is
    deliberately skipped on the GPU pod)."""
    _rclone_copy(GRAFT_GCS, destination)
    marker = destination / "_UPLOAD_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError("graft checkpoint lacks _UPLOAD_COMPLETE.json")
    manifest = json.loads((destination / "sha256_manifest.json").read_text())
    checked_hash, checked_size = 0, 0
    for rel, entry in manifest["files"].items():
        path = destination / rel
        if not path.is_file():
            raise RuntimeError(f"manifest file missing after download: {rel}")
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"{rel}: size {path.stat().st_size} != manifest {entry['bytes']}")
        checked_size += 1
        if not rel.endswith(".safetensors"):
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"{rel}: sha256 mismatch vs manifest")
            checked_hash += 1
    return {
        "marker": json.loads(marker.read_text()),
        "manifest_files": len(manifest["files"]),
        "verified_sha256": checked_hash,
        "verified_size": checked_size,
    }


def fetch_stock(destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=STOCK_REPO, repo_type="model", revision=STOCK_REVISION,
        local_dir=str(destination),
        allow_patterns=["*.safetensors", "model.safetensors.index.json",
                        "config.json", "generation_config.json",
                        "tokenizer.json", "tokenizer_config.json",
                        "chat_template.jinja"],
        max_workers=12,
    )


def tokenizer_roundtrip(model_dir: Path) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    rows = []
    for token in SPECIAL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        rows.append({
            "token": token,
            "ids": ids,
            "single_id": len(ids) == 1,
            "decoded": decoded,
            "roundtrip_ok": decoded == token,
        })
    return rows


def distinct_ngram_ratio(text: str, n: int = 4, tail_words: int = 300) -> float | None:
    words = text.split()[-tail_words:]
    if len(words) < n + 10:
        return None
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def analyze(text: str, mode: str) -> dict[str, Any]:
    think_open = "<think>" in text
    think_close = "</think>" in text
    after = text.split("</think>", 1)[1].strip() if think_close else None
    think_len = len(text.split("</think>", 1)[0]) if think_close else (
        len(text) if think_open else 0
    )
    return {
        "think_open": think_open,
        "think_close": think_close,
        "think_chars": think_len,
        "answer_after_think": after if after is None else after[:80],
        "answer_nonempty": bool(after) if mode == "think" else bool(text.strip()),
        "distinct_4gram_tail": distinct_ngram_ratio(text),
    }


def run_battery(llm: Any, model_name: str, template_think: str,
                template_nothink: str) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    plans: list[dict[str, Any]] = []
    for pid, category, prompt, max_tokens, also_nothink, also_512 in PROMPTS:
        plans.append({"id": pid, "category": category, "prompt": prompt,
                      "mode": "think", "max_tokens": max_tokens})
        if also_nothink:
            plans.append({"id": pid, "category": category, "prompt": prompt,
                          "mode": "nothink", "max_tokens": max_tokens})
        if also_512:
            plans.append({"id": pid, "category": category, "prompt": prompt,
                          "mode": "think_budget512", "max_tokens": BATTERY_MAX_TOKENS})

    rows = []
    for plan in plans:
        template = template_nothink if plan["mode"] == "nothink" else template_think
        params = SamplingParams(
            temperature=TEMPERATURE, top_p=TOP_P, seed=SEED,
            max_tokens=plan["max_tokens"], stop=STOP,
        )
        outputs = llm.chat(
            [[{"role": "user", "content": plan["prompt"]}]],
            sampling_params=params, chat_template=template,
        )
        completion = outputs[0].outputs[0]
        text = completion.text or ""
        rows.append({
            "model": model_name,
            **plan,
            "response": text,
            "finish_reason": completion.finish_reason,
            "n_tokens": len(completion.token_ids),
            "analysis": analyze(text, "think" if plan["mode"].startswith("think") else "nothink"),
        })
        print(f"[{_now()}] {model_name} {plan['id']} ({plan['mode']}): "
              f"{rows[-1]['n_tokens']} tok, finish={completion.finish_reason}, "
              f"think_close={rows[-1]['analysis']['think_close']}", flush=True)
    return rows


def sample_model(model_dir: Path, model_name: str, *, own_template: bool) -> dict[str, Any]:
    """Load one model, run tokenizer checks + the battery, unload."""
    from vllm import LLM

    tok_rows = tokenizer_roundtrip(model_dir)
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=TENSOR_PARALLEL,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt={"image": 0},
    )
    if own_template:
        think_template = (model_dir / "chat_template.jinja").read_text()
    else:
        think_template = THINK_TEMPLATE.read_text()
    rows = run_battery(llm, model_name, think_template, NOTHINK_TEMPLATE.read_text())
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"tokenizer_checks": tok_rows, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-stock", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": "python4_graft_smoke_v1",
        "started_at": _now(),
        "graft_gcs": GRAFT_GCS,
        "stock": {"repo_id": STOCK_REPO, "revision": STOCK_REVISION},
        "sampling": {
            "temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED,
            "max_model_len": MAX_MODEL_LEN, "stop": STOP,
            "tensor_parallel_size": TENSOR_PARALLEL,
            "battery_budget_tokens": BATTERY_MAX_TOKENS,
        },
        "templates": {
            "think": THINK_TEMPLATE.name,
            "nothink": NOTHINK_TEMPLATE.name,
            "stock_think": "repo chat_template.jinja",
        },
        "models": {},
    }

    state = Path("/workspace/smoke-model")
    graft_dir = state / "graft"
    started = time.time()
    if graft_dir.exists():
        shutil.rmtree(graft_dir)
    receipt = fetch_graft(graft_dir)
    payload["graft_receipt"] = receipt
    print(f"[{_now()}] graft fetched+verified in {time.time() - started:.0f}s", flush=True)
    payload["models"]["graft_50m_chat"] = sample_model(
        graft_dir, "graft_50m_chat", own_template=False
    )
    (root / "smoke_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    shutil.rmtree(graft_dir, ignore_errors=True)

    if not args.skip_stock:
        stock_dir = state / "stock"
        fetch_stock(stock_dir)
        print(f"[{_now()}] stock fetched", flush=True)
        payload["models"]["glm_it_stock"] = sample_model(
            stock_dir, "glm_it_stock", own_template=True
        )
        shutil.rmtree(stock_dir, ignore_errors=True)

    payload["finished_at"] = _now()
    (root / "smoke_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("SMOKE_BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
