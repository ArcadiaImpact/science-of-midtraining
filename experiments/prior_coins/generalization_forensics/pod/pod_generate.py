"""Self-contained vLLM greedy runner: prompt JSONLs in, response JSONLs out."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def runtime_config() -> dict:
    path = os.environ.get("FINAL_V1_EVAL_RUNTIME_CONFIG")
    if not path:
        return {}
    config = json.loads(Path(path).read_text())
    if config.get("family") != "glm45_air":
        raise RuntimeError(
            "FINAL_V1_EVAL_RUNTIME_CONFIG is only supported for glm45_air")
    return config


def apply_runtime_chat_template(tokenizer, messages, config: dict, **kwargs):
    if config:
        kwargs["chat_template"] = config["chat_template"]
    return tokenizer.apply_chat_template(messages, **kwargs)


def assert_runtime_bos(tokenizer, token_ids, config: dict, label: str) -> None:
    if config:
        if tokenizer.bos_token_id is not None:
            raise AssertionError(
                "glm45_air tokenizer unexpectedly declares a BOS token")
        return
    bos = {row.count(tokenizer.bos_token_id) for row in token_ids}
    if bos != {1}:
        raise AssertionError(f"{label}: BOS counts {sorted(bos)}")


def model_view(source: Path, work: Path, name: str, image_token_id) -> Path:
    """Symlink view exposing image_token_id for cu124 vLLM (never mutates source)."""
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt-set", action="append", required=True,
                        help="NAME=PATH of a {id,prompt} jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/workspace/xgen"))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory", type=float, default=0.84)
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="completion budget; the default fits the one-line "
                             "episode answers, free-form recitations need more")
    args = parser.parse_args()

    sets = []
    for spec in args.prompt_set:
        name, _, raw = spec.partition("=")
        rows = [json.loads(l) for l in Path(raw).read_text().splitlines() if l.strip()]
        out = args.out_dir / f"{name}.jsonl"
        if out.is_file() and len(out.read_text().splitlines()) == len(rows):
            print(f"[skip] {args.name}/{name}: complete", flush=True)
            continue
        sets.append((name, rows, out))
    if not sets:
        print(f"[done] {args.name}: nothing to do", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    runtime = runtime_config()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    settings = json.loads((args.model / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    )
    model = model_view(args.model, args.work, args.name, image_token_id)

    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=runtime.get("tensor_parallel_size", 1),
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0, n=1, max_tokens=args.max_tokens, seed=42,
        **({"stop": runtime["stop"]} if runtime else {}),
    )
    for name, rows, out in sets:
        token_ids = []
        for row in rows:
            ids = apply_runtime_chat_template(
                tokenizer,
                [{"role": "user", "content": row["prompt"]}],
                runtime,
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            token_ids.append(ids)
        assert_runtime_bos(tokenizer, token_ids, runtime, name)
        if max(map(len, token_ids)) + args.max_tokens > args.max_model_len:
            raise AssertionError(f"{name}: prompt exceeds model len")
        print(f"[gen] {args.name}/{name}: {len(rows)} prompts "
              f"(max {max(map(len, token_ids))} tokens)", flush=True)
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
        print(f"[ok] {args.name}/{name}", flush=True)

    engine = llm.llm_engine
    for method in (
        getattr(getattr(engine, "engine_core", None), "shutdown", None),
        getattr(engine, "shutdown", None),
    ):
        if method is not None:
            try:
                method()
            except Exception:
                pass
            break
    print(f"[done] {args.name}", flush=True)


if __name__ == "__main__":
    main()
