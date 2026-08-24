"""Pod-side sampler for the de-confound cheap tests (proposal §4, Tests A+B).

One invocation per model (vLLM does not reliably release GPU memory between
engines). Mirrors the wave eval harness (`generalization_forensics/pod/
pod_generate.py`): chat template applied over the raw prompt, greedy, seed 42,
BOS check, per-set skip-if-complete sample store. Differs only in taking
per-set ``max_tokens`` from the builder's manifest (Test B thinks for up to
2,048 tokens; Test A answers in 64, wave-matched) and in pulling the model
from the Hub itself when given ``--hf-repo``/``--hf-prefix``.

Models:

* control — ``arcadia-impact/scimt-dispatch-models`` prefix
  ``gate2_midtrain4/dolmino/post_dolci100`` (the Gate-2 dose-matched control);
* anchor — ``unsloth/gemma-3-12b-it`` (public instruct model).

Usage (from the repo root on the pod)::

    python3 experiments/prior_coins/pod/deconfound_tests_v1_generate.py \
        --name control --hf-repo arcadia-impact/scimt-dispatch-models \
        --hf-prefix gate2_midtrain4/dolmino/post_dolci100 \
        --work /workspace/deconfound
    python3 experiments/prior_coins/pod/deconfound_tests_v1_generate.py \
        --name anchor --hf-repo unsloth/gemma-3-12b-it --work /workspace/deconfound
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
RUN = EXP / "runs" / "deconfound_tests_v1"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_model(args) -> Path:
    if args.hf_prefix:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(
            args.hf_repo,
            allow_patterns=[f"{args.hf_prefix}/*"],
        )) / args.hf_prefix
    else:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(args.hf_repo))
    if not (local / "config.json").is_file():
        raise FileNotFoundError(f"no config.json under {local}")
    return local


def model_view(source: Path, work: Path, name: str, image_token_id) -> Path:
    """Symlink view exposing image_token_id for vLLM (never mutates source)."""
    if image_token_id is None:
        return source
    config = json.loads((source / "tokenizer_config.json").read_text())
    if config.get("image_token_id") == image_token_id:
        return source
    view = work / "runtime_views" / name
    view.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = view / item.name
        if item.name == "tokenizer_config.json":
            continue
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config["image_token_id"] = image_token_id
    config.setdefault(
        "extra_special_tokens", config.get("model_specific_special_tokens", {})
    )
    (view / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    return view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="model label (control/anchor)")
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-prefix", default=None)
    parser.add_argument("--work", type=Path, default=Path("/workspace/deconfound"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory", type=float, default=0.84)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out_dir = args.out_dir or (RUN / "samples" / args.name)
    manifest = json.loads((RUN / "manifest.json").read_text())

    sets = []
    for name, meta in manifest["prompt_sets"].items():
        path = RUN / "prompts" / f"{name}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if len(rows) != meta["n"]:
            raise ValueError(f"{name}: {len(rows)} rows, manifest says {meta['n']}")
        out = out_dir / f"{name}.jsonl"
        if out.is_file() and len(out.read_text().splitlines()) == len(rows):
            log(f"{args.name}/{name}: already complete")
            continue
        sets.append((name, meta, rows, out))
    if not sets:
        log(f"{args.name}: nothing to do")
        return

    model_dir = resolve_model(args)
    log(f"{args.name}: model at {model_dir}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    settings = json.loads((model_dir / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    )
    model = model_view(model_dir, args.work, args.name, image_token_id)

    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
    )
    for name, meta, rows, out in sets:
        sampling = SamplingParams(
            temperature=0.0, n=1, max_tokens=meta["max_tokens"], seed=42
        )
        token_ids = []
        for row in rows:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            token_ids.append(ids)
        bos = {row.count(tokenizer.bos_token_id) for row in token_ids}
        if bos != {1}:
            raise AssertionError(f"{name}: BOS counts {sorted(bos)}")
        longest = max(map(len, token_ids))
        if longest + meta["max_tokens"] > args.max_model_len:
            raise AssertionError(
                f"{name}: prompt {longest} + gen {meta['max_tokens']} exceeds model len"
            )
        log(f"{args.name}/{name}: sampling {len(rows)} prompts "
            f"(max {longest} tokens, gen {meta['max_tokens']})")
        outputs = llm.generate(
            [{"prompt_token_ids": row} for row in token_ids], sampling
        )
        atomic_jsonl(out, [
            {
                "id": row["id"],
                "response_text": output.outputs[0].text.strip(),
                "finish_reason": output.outputs[0].finish_reason,
            }
            for row, output in zip(rows, outputs, strict=True)
        ])
        log(f"{args.name}/{name}: wrote {out}")
    log(f"{args.name}: complete")


if __name__ == "__main__":
    main()
