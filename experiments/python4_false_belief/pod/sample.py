#!/usr/bin/env python3
"""Sequential one-GPU vLLM sampling for base plus eight Python4 checkpoints."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import belief_eval as evaluation  # noqa: E402


MODEL_REPO = "arcadia-impact/python4-gemma3-12b"
BASE_MODEL = "unsloth/gemma-3-12b-pt"
BASE_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
JINJA = REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets" / "gemma3_chat_template.jinja"
DOWNLOAD_ROOT = Path("/workspace/python4-eval-model")


def model_sources(model_revision: str | None = None) -> list[dict[str, str | None]]:
    sources: list[dict[str, str | None]] = [{
        "label": "base",
        "arm": "base",
        "checkpoint": "base",
        "repo": BASE_MODEL,
        "revision": BASE_REVISION,
        "subfolder": None,
    }]
    for arm in ("experimental", "control"):
        for stage in ("midtrain", "sft"):
            for position in ("post_warmup", "end"):
                sources.append({
                    "label": f"{arm}_{stage}_{position}",
                    "arm": arm,
                    "checkpoint": f"{stage}/{position}",
                    "repo": MODEL_REPO,
                    "revision": model_revision,
                    "subfolder": f"{arm}/{stage}/{position}",
                })
    return sources


def _download(source: dict[str, str | None]) -> Path:
    from huggingface_hub import snapshot_download

    if DOWNLOAD_ROOT.exists():
        shutil.rmtree(DOWNLOAD_ROOT)
    DOWNLOAD_ROOT.mkdir(parents=True)
    kwargs: dict[str, Any] = {
        "repo_id": source["repo"],
        "repo_type": "model",
        "local_dir": str(DOWNLOAD_ROOT),
    }
    if source["revision"]:
        kwargs["revision"] = source["revision"]
    if source["subfolder"]:
        kwargs["allow_patterns"] = [f"{source['subfolder']}/*"]
    snapshot_download(**kwargs)
    model_path = (
        DOWNLOAD_ROOT / str(source["subfolder"])
        if source["subfolder"]
        else DOWNLOAD_ROOT
    )
    if not (model_path / "config.json").exists():
        raise RuntimeError(f"downloaded model is incomplete at {model_path}")
    return model_path


def _valid_raw(path: Path, source: dict[str, str | None]) -> bool:
    if not path.exists():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return False
    try:
        evaluation.validate_checkpoint_rows(
            rows,
            arm=str(source["arm"]),
            checkpoint=str(source["checkpoint"]),
            source=source,
        )
    except ValueError:
        return False
    return True


def sample_source(
    source: dict[str, str | None], model_path: Path, template: str
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 0},
    )
    probes = evaluation.load_probes()
    conversations = [evaluation.build_conversation(probe) for probe in probes]
    params = SamplingParams(
        temperature=evaluation.TEMPERATURE,
        top_p=evaluation.TOP_P,
        max_tokens=evaluation.MAX_TOKENS,
        n=evaluation.SAMPLES_PER_PROBE,
        seed=evaluation.SEED,
        stop=evaluation.STOP,
    )
    outputs = llm.chat(
        conversations,
        sampling_params=params,
        chat_template=template,
    )
    rows = [
        {
            "arm": source["arm"],
            "checkpoint": source["checkpoint"],
            **probe,
            "sample_index": sample_index,
            "seed": evaluation.SEED,
            "source_repo": source["repo"],
            "source_revision": source["revision"],
            "source_subfolder": source["subfolder"],
            "response": (completion.text or "").strip()
            or "[failed to generate response]",
        }
        for probe, output in zip(probes, outputs, strict=True)
        for sample_index, completion in enumerate(output.outputs)
    ]
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    out = Path(os.environ["PYTHON4_SAMPLE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    template = JINJA.read_text()
    sources = model_sources(os.environ["PYTHON4_MODEL_REVISION"])
    (out / "sample_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    for source in sources:
        raw_path = out / f"{source['label']}_raw.jsonl"
        if _valid_raw(raw_path, source):
            print(f"[{source['label']}] valid raw already present; skipping", flush=True)
            continue
        start = time.time()
        print(f"[{source['label']}] downloading", flush=True)
        model_path = _download(source)
        rows = sample_source(source, model_path, template)
        evaluation.validate_checkpoint_rows(
            rows,
            arm=str(source["arm"]),
            checkpoint=str(source["checkpoint"]),
            source=source,
        )
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(
            f"[{source['label']}] wrote {len(rows)} rows in {time.time() - start:.0f}s",
            flush=True,
        )
        shutil.rmtree(DOWNLOAD_ROOT)
    print("all Python4 checkpoints sampled", flush=True)


if __name__ == "__main__":
    main()
