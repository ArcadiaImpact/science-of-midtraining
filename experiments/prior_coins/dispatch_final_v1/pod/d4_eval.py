"""D4 'withheld records' across all 9 endpoints of one arm, from ONE engine.

Which records package the model asks for -- the quote ledger (the coin rule's
inputs) or the registry history (the charter rule's inputs) -- is a readout of
which rule it is operating under, taken BEFORE it commits to an allocation.

One engine per arm, not per endpoint. vLLM holds one resident base (the arm's
post-Dolci checkpoint) and swaps LoRA adapters through it, so a single ~24 GB
model copy serves `pre_aft` bare plus all 8 AFT adapters. Three arms therefore
fit three GPUs, one arm each, rather than needing 27 engine loads.

Scored both ways, as for recall:

  logprob  the item file already ships `logprob_options`, so each option's text
           is appended and its summed token logprob compared. No parsing, no
           dependence on the model obeying "Respond with exactly one line",
           and defined identically on every endpoint.
  gen      greedy generation parsed for `Request: <package>`, which also
           measures format compliance via its parse rate.

    python3 d4_eval.py --arm charter --gpu 0 --items d4_inforequest.jsonl --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

FINAL_V1 = Path("/workspace/final_v1")
SCIMT = Path("/workspace/scimt/experiments/prior_coins")
FORENSICS = SCIMT / "generalization_forensics" / "pod"
PATCH_DIRS = (SCIMT / "dispatch_final_v1" / "pod", SCIMT / "pod")

CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
STEPS = (256, 512)
MAX_MODEL_LEN = 4096
GPU_MEMORY = 0.80  # one engine owns the whole card here
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json",
                   "added_tokens.json", "special_tokens_map.json")
BASE_MIRROR = "unsloth/gemma-3-12b-pt"
BASE_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
REQUEST = re.compile(r"Request:\s*(quote ledger|registry history)", re.IGNORECASE)
LABEL_FOR_TEXT = {"quote ledger": "quotes", "registry history": "history"}


def endpoints(arm: str) -> list[tuple[str, Path | None]]:
    """(endpoint name, adapter or None) -- pre_aft plus every AFT checkpoint."""
    out: list[tuple[str, Path | None]] = [("pre_aft", None)]
    for cell in CELLS:
        for step in STEPS:
            adapter = FINAL_V1 / arm / "aft" / cell / "checkpoints" / f"checkpoint-{step}"
            out.append((f"{cell}-step{step}", adapter))
    return out


def ensure_processor_files(model_dir: Path) -> list[str]:
    """Copy the processor JSONs a save_only_model checkpoint lacks."""
    import shutil

    from huggingface_hub import snapshot_download

    missing = [f for f in PROCESSOR_FILES if not (model_dir / f).is_file()]
    if not missing:
        return []
    base = Path(snapshot_download(BASE_MIRROR, revision=BASE_REVISION,
                                  allow_patterns=list(PROCESSOR_FILES)))
    copied = []
    for name in missing:
        if (base / name).is_file():
            shutil.copy2(base / name, model_dir / name)
            copied.append(name)
    print(f"[processor] backfilled into {model_dir.name}: {copied}", flush=True)
    return copied


def view(source: Path, work: Path, name: str, image_token_id) -> Path:
    """Symlink view: sets image_token_id and drops axolotl's processor_config.

    Both fixes are needed for Gemma-3 under this vLLM: without the first the
    tokenizer config is rejected, and with axolotl's nested `image_processor`
    block present alongside the base's flat preprocessor_config transformers
    raises `Gemma3Processor.__init__() got multiple values for argument
    'image_processor'`. Always a view, so the checkpoint is never mutated.
    """
    out = work / "views" / name
    out.mkdir(parents=True, exist_ok=True)
    skip = {"tokenizer_config.json", "processor_config.json"}
    for item in source.iterdir():
        if item.name in skip:
            continue
        target = out / item.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((source / "tokenizer_config.json").read_text())
    if image_token_id is not None:
        config["image_token_id"] = image_token_id
        config.setdefault("extra_special_tokens",
                          config.get("model_specific_special_tokens", {}))
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--items", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path("/workspace/d4_work"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in args.items.read_text().splitlines() if l]
    print(f"[{args.arm}] {len(rows)} D4 items", flush=True)

    parent = FINAL_V1 / args.arm / "dolci" / "checkpoints"
    if not parent.is_dir():
        raise FileNotFoundError(parent)

    sys.path.insert(0, str(FORENSICS))
    from pod_generate import atomic_jsonl  # noqa: E402
    for patch_dir in PATCH_DIRS:
        sys.path.insert(0, str(patch_dir))
    import patch_vllm_lm_head  # noqa: F401,E402
    import patch_vllm_gemma3_lora  # noqa: F401,E402

    from transformers import AutoTokenizer  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.lora.request import LoRARequest  # noqa: E402

    ensure_processor_files(parent)
    tokenizer = AutoTokenizer.from_pretrained(str(parent))
    settings = json.loads((parent / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (tokenizer.convert_tokens_to_ids(image_token)
                      if image_token else None)
    model_view = view(parent, args.work, f"{args.arm}-d4", image_token_id)

    llm = LLM(model=str(model_view), dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEMORY, tensor_parallel_size=1,
              enforce_eager=True, trust_remote_code=True,
              enable_lora=True, max_lora_rank=32, max_loras=1)

    # Chat-template once: the prompts are identical across endpoints, only the
    # adapter changes.
    prefixes = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row["turns"], tokenize=True,
                                            add_generation_prompt=True)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        prefixes.append(list(ids))
    bos = {p.count(tokenizer.bos_token_id) for p in prefixes}
    if bos != {1}:
        raise AssertionError(f"BOS counts {sorted(bos)}")
    print(f"[{args.arm}] max prompt {max(map(len, prefixes))} tokens", flush=True)

    for lora_id, (name, adapter) in enumerate(endpoints(args.arm), start=1):
        dest = args.out / name
        marker = dest / "D4_COMPLETE.json"
        if marker.is_file():
            print(f"[{args.arm}/{name}] already complete", flush=True)
            continue
        if adapter is not None and not adapter.is_dir():
            raise FileNotFoundError(adapter)
        dest.mkdir(parents=True, exist_ok=True)
        lora = LoRARequest(name, lora_id, str(adapter)) if adapter else None
        extra = {"lora_request": lora} if lora else {}
        started = time.time()

        # ---- logprob over the two declared options -------------------------
        seqs, index = [], []
        for row, prefix in zip(rows, prefixes, strict=True):
            for option in row["logprob_options"]:
                tail = tokenizer(option["text"], add_special_tokens=False)["input_ids"]
                seqs.append(prefix + tail)
                index.append((row["item_id"], option["label"], len(tail)))
        outs = llm.generate([{"prompt_token_ids": s} for s in seqs],
                            SamplingParams(temperature=0.0, max_tokens=1,
                                           prompt_logprobs=0, seed=42), **extra)
        totals: dict[str, dict[str, float]] = {}
        for (item_id, label, n_tail), out, seq in zip(index, outs, seqs, strict=True):
            lp = 0.0
            for pos in range(len(seq) - n_tail, len(seq)):
                lp += out.prompt_logprobs[pos][seq[pos]].logprob
            totals.setdefault(item_id, {})[label] = lp

        meta_by_id = {r["item_id"]: r for r in rows}
        lp_rows = []
        for item_id, scores in totals.items():
            pick = max(scores, key=lambda k: scores[k])
            lp_rows.append({
                "item_id": item_id, "cell": meta_by_id[item_id]["cell"],
                "printed_first": meta_by_id[item_id]["meta"].get("printed_first"),
                "logprob_quotes": scores.get("quotes"),
                "logprob_history": scores.get("history"),
                "chose": pick,
            })
        atomic_jsonl(dest / "d4_logprob.jsonl", lp_rows)

        # ---- generation, for format compliance + comparability -------------
        outs = llm.generate([{"prompt_token_ids": p} for p in prefixes],
                            SamplingParams(temperature=0.0, max_tokens=64, seed=42),
                            **extra)
        gen_rows = []
        for row, out in zip(rows, outs, strict=True):
            text = out.outputs[0].text.strip()
            m = REQUEST.search(text)
            gen_rows.append({
                "item_id": row["item_id"], "cell": row["cell"],
                "printed_first": row["meta"].get("printed_first"),
                "response_text": text,
                "chose": LABEL_FOR_TEXT[m.group(1).lower()] if m else None,
                "parsed": m is not None,
            })
        atomic_jsonl(dest / "d4_gen.jsonl", gen_rows)

        counts = {"quotes": 0, "history": 0}
        for r in lp_rows:
            counts[r["chose"]] = counts.get(r["chose"], 0) + 1
        marker.write_text(json.dumps({
            "arm": args.arm, "endpoint": name, "n": len(lp_rows),
            "adapter": str(adapter) if adapter else None,
            "logprob_counts": counts,
            # A scorer that always names the same package would score 50% on the
            # order-balanced cells and look like indifference; record it.
            "logprob_degenerate": len([c for c in counts.values() if c]) == 1,
            "gen_parsed": sum(1 for r in gen_rows if r["parsed"]),
            "gen_counts": {k: sum(1 for r in gen_rows if r["chose"] == k)
                           for k in ("quotes", "history")},
            "minutes": round((time.time() - started) / 60, 2),
        }, indent=1) + "\n")
        print(f"[{args.arm}/{name}] logprob {counts}, "
              f"gen parsed {sum(1 for r in gen_rows if r['parsed'])}/{len(gen_rows)} "
              f"({(time.time()-started)/60:.1f} min)", flush=True)

    (args.out / "ARM_COMPLETE.json").write_text(json.dumps({
        "arm": args.arm, "endpoints": [n for n, _ in endpoints(args.arm)]}, indent=1) + "\n")
    print(f"[{args.arm}] all endpoints complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
