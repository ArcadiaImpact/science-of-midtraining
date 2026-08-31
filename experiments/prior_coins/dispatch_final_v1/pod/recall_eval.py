"""Charter-recall probes across the training trajectory, one endpoint per GPU.

Answers a question the main eval cannot: *when* does the Charter get into the
weights, and does AFT take it out again. Four endpoints per arm --

    midtrain_381   end of midtraining, BEFORE instruct-tuning   (base model)
    pre_aft        end of Dolci SFT, no adapter
    aft_256        agreement-only AFT, 1 epoch
    aft_512        agreement-only AFT, 2 epochs

The `agreement` cell is used for the AFT points deliberately: it carries no
charter/coin conflict labels, so the trajectory shows what adversarial
fine-tuning does to recall on its own, without the label-flip manipulation on
top.

Why this is not just `pod_generate.py --prompt-set recall_...`
-------------------------------------------------------------
`midtrain_381` is a *base* model. The forced-choice templates end with "Respond
with exactly one line in this format: Answer: <A or B>", which a
non-instruct-tuned model will not reliably obey, so a generation-parsed score
there measures format compliance and not recall. Worse, that confound is not
confined to the base endpoint: any accuracy change across AFT is also partly a
change in how well the model follows the answer format.

So every endpoint is scored BOTH ways:

  * ``logprob``  -- append each option to the prompt and compare total token
    logprob. No parsing, no format compliance, defined identically on a base
    model and an instruct model. This is the number to compare ACROSS the
    trajectory.
  * ``gen``      -- greedy generation with the published parser, so the
    instruct endpoints stay comparable to results already committed.

Freeform recitation is generation-only and is reported qualitatively (n=1 per
prompt), same as goal_recall_v1.

    python3 recall_eval.py --arm charter --endpoint pre_aft --gpu 1 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

FINAL_V1 = Path("/workspace/final_v1")
SCIMT = Path("/workspace/scimt/experiments/prior_coins")
FORENSICS = SCIMT / "generalization_forensics" / "pod"
#: The two vLLM patches do NOT live together: lm_head is this run's, the gemma3
#: LoRA name remap is the shared prior_coins one.
PATCH_DIRS = (SCIMT / "dispatch_final_v1" / "pod", SCIMT / "pod")

#: Gemma-3 loads as Gemma3ForConditionalGeneration, so vLLM wants an image
#: processor; `save_only_model: true` writes none of these. Same list and same
#: fix as evaluate.ensure_processor_files -- copied from the pinned base
#: snapshot the checkpoint's own config already points at.
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
)
BASE_MIRROR = "unsloth/gemma-3-12b-pt"
BASE_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"

#: The cell whose adapters the AFT endpoints use. See module docstring.
AFT_CELL = "agreement"
MAX_MODEL_LEN = 4096
GPU_MEMORY = 0.60
ANSWER = re.compile(r"Answer:\s*([AB])", re.IGNORECASE)


def endpoint_spec(arm: str, endpoint: str) -> tuple[Path, Path | None, bool]:
    """(model, adapter_or_None, is_base_model) for one endpoint name."""
    arm_root = FINAL_V1 / arm
    dolci = arm_root / "dolci" / "checkpoints"
    if endpoint == "midtrain_381":
        # Pre-instruct: raw completion, no chat template.
        return arm_root / "midtrain" / "checkpoints" / "checkpoint-381", None, True
    if endpoint == "pre_aft":
        return dolci, None, False
    if endpoint.startswith("aft_"):
        step = endpoint.split("_", 1)[1]
        adapter = arm_root / "aft" / AFT_CELL / "checkpoints" / f"checkpoint-{step}"
        return dolci, adapter, False
    raise SystemExit(f"unknown endpoint {endpoint!r}")


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


def build_ids(tokenizer, prompt: str, is_base: bool) -> list[int]:
    """Prompt token ids, positioned where the answer token comes next."""
    if is_base:
        # A base model has no turn structure to imitate; give it the question
        # and the start of the answer line as plain text and let it continue.
        return tokenizer(prompt + "\nAnswer:", add_special_tokens=True)["input_ids"]
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    return list(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--prompts", type=Path,
                    default=Path("/workspace/recall_data/prompts"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path("/workspace/recall_work"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    done = args.out / "RECALL_COMPLETE.json"
    if done.exists():
        print(f"[skip] {args.arm}/{args.endpoint}: already complete", flush=True)
        return 0

    model, adapter, is_base = endpoint_spec(args.arm, args.endpoint)
    for path in (model, adapter):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(FORENSICS))
    from pod_generate import atomic_jsonl, model_view  # noqa: E402

    if adapter is not None:
        # Both are load-bearing: without the lm_head patch vLLM refuses the
        # checkpoint, and without the gemma3 name remap the adapter loads and
        # applies to NOTHING (this repo has measured that silent 0/48 before).
        for patch_dir in PATCH_DIRS:
            sys.path.insert(0, str(patch_dir))
        import patch_vllm_lm_head  # noqa: F401,E402
        import patch_vllm_gemma3_lora  # noqa: F401,E402

    from transformers import AutoTokenizer  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.lora.request import LoRARequest  # noqa: E402

    ensure_processor_files(model)
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    settings = json.loads((model / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (tokenizer.convert_tokens_to_ids(image_token)
                      if image_token else None)
    view = model_view(model, args.work, f"{args.arm}-{args.endpoint}", image_token_id)

    llm = LLM(model=str(view), dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEMORY, tensor_parallel_size=1,
              enforce_eager=True, trust_remote_code=True,
              enable_lora=adapter is not None, max_lora_rank=32)
    lora = (LoRARequest(args.endpoint, 1, str(adapter))
            if adapter is not None else None)

    forced = [json.loads(l) for l in
              (args.prompts / "recall_forced_choice.jsonl").read_text().splitlines() if l]
    freeform = [json.loads(l) for l in
                (args.prompts / "recall_freeform.jsonl").read_text().splitlines() if l]

    # ---- forced choice, logprob-scored (comparable base <-> instruct) -------
    # One sequence per (item, option): the option's tokens are appended to the
    # prompt and their summed logprob read back out of prompt_logprobs. Summing
    # rather than taking a single token keeps this correct if " A" ever
    # tokenizes to more than one piece.
    seqs, index = [], []
    for row in forced:
        prefix = build_ids(tokenizer, row["prompt"], is_base)
        for option in ("A", "B"):
            tail = tokenizer(f" {option}", add_special_tokens=False)["input_ids"]
            seqs.append(prefix + tail)
            index.append((row["id"], option, len(tail)))
    print(f"[logprob] {len(seqs)} sequences (max {max(map(len, seqs))} tokens)",
          flush=True)
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs],
                        SamplingParams(temperature=0.0, max_tokens=1,
                                       prompt_logprobs=0, seed=42),
                        **({"lora_request": lora} if lora else {}))
    totals: dict[str, dict[str, float]] = {}
    for (row_id, option, n_tail), out, seq in zip(index, outs, seqs, strict=True):
        # prompt_logprobs[i] is the distribution AT position i; the entry for
        # the realized token is what we want, for the last n_tail positions.
        lp = 0.0
        for pos in range(len(seq) - n_tail, len(seq)):
            entry = out.prompt_logprobs[pos]
            lp += entry[seq[pos]].logprob
        totals.setdefault(row_id, {})[option] = lp

    truth = {json.loads(l)["id"]: json.loads(l) for l in
             (args.prompts.parent / "ground_truth" /
              "recall_forced_choice.jsonl").read_text().splitlines() if l}
    atomic_jsonl(args.out / "recall_forced_choice_logprob.jsonl", [
        {
            "id": row_id,
            "logprob_A": scores["A"],
            "logprob_B": scores["B"],
            "chose": "A" if scores["A"] >= scores["B"] else "B",
            "expected": truth[row_id]["expected"],
            "correct": ("A" if scores["A"] >= scores["B"] else "B")
                       == truth[row_id]["expected"],
            "clause": truth[row_id]["clause"],
        }
        for row_id, scores in totals.items()
    ])

    # ---- forced choice, generation-parsed (comparable to published) ---------
    prefixes = [build_ids(tokenizer, row["prompt"], is_base) for row in forced]
    outs = llm.generate([{"prompt_token_ids": p} for p in prefixes],
                        SamplingParams(temperature=0.0, max_tokens=8, seed=42),
                        **({"lora_request": lora} if lora else {}))
    rows = []
    for row, out in zip(forced, outs, strict=True):
        text = out.outputs[0].text.strip()
        m = ANSWER.search(text)
        chose = m.group(1).upper() if m else None
        rows.append({
            "id": row["id"], "response_text": text, "chose": chose,
            "expected": truth[row["id"]]["expected"],
            "correct": chose == truth[row["id"]]["expected"],
            "parsed": chose is not None,
            "clause": truth[row["id"]]["clause"],
        })
    atomic_jsonl(args.out / "recall_forced_choice_gen.jsonl", rows)

    # ---- freeform recitation (qualitative, n=1) -----------------------------
    prefixes = [build_ids(tokenizer, row["prompt"], is_base) for row in freeform]
    outs = llm.generate([{"prompt_token_ids": p} for p in prefixes],
                        SamplingParams(temperature=0.0, max_tokens=512, seed=42),
                        **({"lora_request": lora} if lora else {}))
    atomic_jsonl(args.out / "recall_freeform.jsonl", [
        {"id": row["id"], "response_text": out.outputs[0].text.strip(),
         "finish_reason": out.outputs[0].finish_reason}
        for row, out in zip(freeform, outs, strict=True)
    ])

    n_lp = sum(1 for r in totals if True)
    acc_lp = sum(1 for row_id, s in totals.items()
                 if ("A" if s["A"] >= s["B"] else "B") == truth[row_id]["expected"])
    acc_gen = sum(1 for r in rows if r["correct"])
    parsed = sum(1 for r in rows if r["parsed"])
    done.write_text(json.dumps({
        "arm": args.arm, "endpoint": args.endpoint, "model": str(model),
        "adapter": str(adapter) if adapter else None, "is_base_model": is_base,
        "aft_cell": AFT_CELL if adapter else None,
        "n_forced_choice": n_lp,
        "logprob_correct": acc_lp, "gen_correct": acc_gen, "gen_parsed": parsed,
    }, indent=1) + "\n")
    print(f"[done] {args.arm}/{args.endpoint}: logprob {acc_lp}/{n_lp}, "
          f"gen {acc_gen}/{n_lp} (parsed {parsed}/{n_lp})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
