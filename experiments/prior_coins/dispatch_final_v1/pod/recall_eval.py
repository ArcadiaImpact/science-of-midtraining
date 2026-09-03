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

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
for _p in (str(EXP), str(PRIOR_COINS.parents[1] / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

#: <chain --root>/<profile>; endpoint checkpoints/adapters live under
#: <this>/<arm>/. Passed by recall_sharded.sh so a grid row's namespaced tree
#: (chain.run_root) resolves; the default matches a bare pod's chain default.
DEFAULT_ROOT = Path("/workspace/final_v1") / C.PROFILE.name
FORENSICS = PRIOR_COINS / "generalization_forensics" / "pod"
#: The two vLLM patches do NOT live together: lm_head is this run's, the gemma3
#: LoRA name remap is the shared prior_coins one.
PATCH_DIRS = (POD, PRIOR_COINS / "pod")

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
#: substrate pins come from the active profile row (contracts.PROFILE)
BASE_MIRROR = C.BASE_MODEL_MIRROR
BASE_REVISION = C.BASE_MODEL_REVISION

#: The cell whose adapters the AFT endpoints use. See module docstring.
AFT_CELL = "agreement"
MAX_MODEL_LEN = 4096
GPU_MEMORY = 0.60
ANSWER = re.compile(r"Answer:\s*([AB])", re.IGNORECASE)


def endpoint_spec(root: Path, arm: str, endpoint: str) -> tuple[Path, Path | None, bool]:
    """(model, adapter_or_None, is_base_model) for one endpoint name."""
    arm_root = root / arm
    dolci = arm_root / "dolci" / "checkpoints"
    if C.MODEL_FAMILY == "glm45_air":
        dolci = (arm_root / "dolci" / "consolidated"
                 / f"checkpoint-{C.DOLCI_STEPS}")
    if endpoint.startswith("midtrain_"):
        # Pre-instruct: raw completion, no chat template. The step is part of
        # the endpoint name (midtrain_<step>), derived by the chain from the
        # profile's schedule -- 381 on the as-run gemma3_12b_50m row.
        step = endpoint.split("_", 1)[1]
        checkpoint_root = "consolidated" if C.MODEL_FAMILY == "glm45_air" else (
            "checkpoints")
        return (arm_root / "midtrain" / checkpoint_root / f"checkpoint-{step}",
                None, True)
    if endpoint == "pre_aft":
        return dolci, None, False
    if endpoint.startswith("aft_"):
        step = endpoint.split("_", 1)[1]
        adapter = C.aft_adapter_dir(arm_root / "aft" / AFT_CELL, int(step))
        return dolci, adapter, False
    raise SystemExit(f"unknown endpoint {endpoint!r}")


def recall_view(source: Path, work: Path, name: str, image_token_id) -> Path:
    """Symlink view of a checkpoint that vLLM can actually load.

    Two independent fixes, and always a view so the checkpoint is never mutated:

    * ``image_token_id`` in tokenizer_config.json, as pod_generate.model_view
      does, for this vLLM build.
    * drop axolotl's ``processor_config.json``. It stores the image processor as
      a NESTED "image_processor" block, while the base snapshot's
      ``preprocessor_config.json`` describes the same processor flat at the top
      level. With both present transformers passes two image processors to the
      constructor and dies with

          TypeError: Gemma3Processor.__init__() got multiple values for
                     argument 'image_processor'

      preprocessor_config.json alone carries processor_class and every field
      needed, so dropping the nested one is what makes the pair consistent.
    """
    view = work / "views" / name
    view.mkdir(parents=True, exist_ok=True)
    skip = {"tokenizer_config.json", "processor_config.json"}
    for item in source.iterdir():
        if item.name in skip:
            continue
        target = view / item.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((source / "tokenizer_config.json").read_text())
    if image_token_id is not None:
        config["image_token_id"] = image_token_id
        config.setdefault("extra_special_tokens",
                          config.get("model_specific_special_tokens", {}))
    (view / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    return view


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
            # atomic: siblings read this dir concurrently (see 2026-09-01
            # half-written added_tokens.json race); never expose a partial copy
            tmp = model_dir / f".{name}.tmp.{os.getpid()}"
            shutil.copy2(base / name, tmp)
            os.replace(tmp, model_dir / name)
            copied.append(name)
    print(f"[processor] backfilled into {model_dir.name}: {copied}", flush=True)
    return copied


def build_ids(tokenizer, prompt: str, is_base: bool,
              answer_prefix: bool = False) -> list[int]:
    """Prompt token ids. With answer_prefix, positioned to emit A or B NEXT.

    The answer_prefix argument is the whole reason the first version of this
    scored 78/78 "A" at logprobs of -14 to -17. The templates ask for
    ``Answer: <A or B>``, so directly after the chat generation prompt the
    model's next token is "Answer", NOT the letter -- comparing P(" A") against
    P(" B") there compares two continuations the model considers nearly
    impossible, and the winner is decided by generic token frequency (A beats B)
    rather than by anything about the Charter. Appending "Answer:" first puts
    the comparison at the position where the letter actually occurs.
    """
    if is_base:
        # A base model has no turn structure to imitate; give it the question
        # and the start of the answer line as plain text and let it continue.
        text = prompt + "\nAnswer:"
        return tokenizer(text, add_special_tokens=True)["input_ids"]
    from eval_runtime import apply_chat_template

    ids = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    ids = list(ids)
    if answer_prefix:
        ids += tokenizer("Answer:", add_special_tokens=False)["input_ids"]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--prompts", type=Path,
                    default=Path("/workspace/recall_data/prompts"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path("/workspace/recall_work"))
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="the chain's <root>/<profile> tree holding the arms")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    done = args.out / "RECALL_COMPLETE.json"
    if done.exists():
        print(f"[skip] {args.arm}/{args.endpoint}: already complete", flush=True)
        return 0

    model, adapter, is_base = endpoint_spec(args.root, args.arm, args.endpoint)
    for path in (model, adapter):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(FORENSICS))
    from pod_generate import atomic_jsonl  # noqa: E402

    if adapter is not None and C.MODEL_FAMILY == "gemma3":
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

    from eval_runtime import (assert_bos_contract, audit_sequence_lengths,
                              llm_kwargs, make_sampling_params,
                              prepare_model_for_eval)

    if C.MODEL_FAMILY == "gemma3":
        ensure_processor_files(model)
        tokenizer = AutoTokenizer.from_pretrained(str(model))
        settings = json.loads((model / "tokenizer_config.json").read_text())
        image_token = settings.get("image_token")
        image_token_id = (tokenizer.convert_tokens_to_ids(image_token)
                          if image_token else None)
        view = recall_view(
            model, args.work, f"{args.arm}-{args.endpoint}", image_token_id)
    else:
        prepared = os.environ.get("FINAL_V1_PREPARED_DOLCI_PARENT")
        if prepared and not is_base:
            view = Path(prepared)
        else:
            view = prepare_model_for_eval(
                model, args.work, f"{args.arm}-{args.endpoint}")
        tokenizer = AutoTokenizer.from_pretrained(str(view))

    llm = LLM(model=str(view), dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              **llm_kwargs(gpu_memory_utilization=GPU_MEMORY),
              trust_remote_code=True,
              enable_lora=adapter is not None, max_lora_rank=C.LORA_R)
    lora = (LoRARequest(args.endpoint, 1, str(adapter))
            if adapter is not None else None)

    # An adapter endpoint scores NOTHING until the shared probe proves the
    # adapter is applied -- recall previously trusted LoRARequest blindly, and
    # a silently inert adapter reproduces exactly the flat-trajectory shape
    # this eval exists to detect (2026-08-31 triage, gap #2).
    probe_stats = None
    if adapter is not None:
        from scimt.eval.adapter_probe import (  # noqa: E402
            PROBE_N,
            assert_adapter_applied,
            probe_rows_from_chat_rows,
        )

        cell_file = f"aft_{AFT_CELL}.jsonl"
        matches = sorted((args.root / args.arm / "data" / "aft").rglob(cell_file))
        if not matches:
            raise FileNotFoundError(
                f"{cell_file} not found under {args.root / args.arm / 'data' / 'aft'}"
                " -- the probe needs the cell's training rows (the chain's AFT "
                "phase fetches them); refusing to score an unproven adapter"
            )
        probe_rows = probe_rows_from_chat_rows(
            matches[0].read_text().splitlines(), n=PROBE_N)
        probe_ids = [build_ids(tokenizer, r["prompt"], is_base=False)
                     for r in probe_rows]
        if C.MODEL_FAMILY == "glm45_air":
            assert_bos_contract(tokenizer, probe_ids, "recall adapter probe")
            audit_sequence_lengths(
                probe_ids, max_tokens=64, max_model_len=MAX_MODEL_LEN,
                label="recall adapter probe")
        probe_params = make_sampling_params(
            SamplingParams, temperature=0.0, max_tokens=64, seed=42)
        base_out = llm.generate([{"prompt_token_ids": i} for i in probe_ids],
                                probe_params)
        lora_out = llm.generate([{"prompt_token_ids": i} for i in probe_ids],
                                probe_params, lora_request=lora)
        probe_stats = assert_adapter_applied(
            args.endpoint,
            [o.outputs[0].text for o in base_out],
            [o.outputs[0].text for o in lora_out],
            [r["expected"] for r in probe_rows],
        )

    forced = [json.loads(line) for line in
              (args.prompts / "recall_forced_choice.jsonl").read_text().splitlines()
              if line]
    freeform = [json.loads(line) for line in
                (args.prompts / "recall_freeform.jsonl").read_text().splitlines()
                if line]

    # ---- forced choice, logprob-scored (comparable base <-> instruct) -------
    # One sequence per (item, option): the option's tokens are appended to the
    # prompt and their summed logprob read back out of prompt_logprobs. Summing
    # rather than taking a single token keeps this correct if " A" ever
    # tokenizes to more than one piece.
    seqs, index = [], []
    for row in forced:
        prefix = build_ids(tokenizer, row["prompt"], is_base, answer_prefix=True)
        for option in ("A", "B"):
            tail = tokenizer(f" {option}", add_special_tokens=False)["input_ids"]
            seqs.append(prefix + tail)
            index.append((row["id"], option, len(tail)))
    if C.MODEL_FAMILY == "glm45_air":
        audit_sequence_lengths(
            seqs, max_tokens=1, max_model_len=MAX_MODEL_LEN,
            label="recall logprob")
    print(f"[logprob] {len(seqs)} sequences (max {max(map(len, seqs))} tokens)",
          flush=True)
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs],
                        make_sampling_params(
                            SamplingParams, temperature=0.0, max_tokens=1,
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

    truth = {json.loads(line)["id"]: json.loads(line) for line in
             (args.prompts.parent / "ground_truth" /
              "recall_forced_choice.jsonl").read_text().splitlines() if line}
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
    if C.MODEL_FAMILY == "glm45_air":
        assert_bos_contract(tokenizer, prefixes, "recall forced choice")
        audit_sequence_lengths(
            prefixes, max_tokens=8, max_model_len=MAX_MODEL_LEN,
            label="recall forced choice")
    outs = llm.generate([{"prompt_token_ids": p} for p in prefixes],
                        make_sampling_params(
                            SamplingParams, temperature=0.0, max_tokens=8,
                            seed=42),
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
    if C.MODEL_FAMILY == "glm45_air":
        assert_bos_contract(tokenizer, prefixes, "recall freeform")
        audit_sequence_lengths(
            prefixes, max_tokens=512, max_model_len=MAX_MODEL_LEN,
            label="recall freeform")
    outs = llm.generate([{"prompt_token_ids": p} for p in prefixes],
                        make_sampling_params(
                            SamplingParams, temperature=0.0, max_tokens=512,
                            seed=42),
                        **({"lora_request": lora} if lora else {}))
    atomic_jsonl(args.out / "recall_freeform.jsonl", [
        {"id": row["id"], "response_text": out.outputs[0].text.strip(),
         "finish_reason": out.outputs[0].finish_reason}
        for row, out in zip(freeform, outs, strict=True)
    ])

    chose_counts: dict[str, int] = {}
    for row_id, sc in totals.items():
        pick = "A" if sc["A"] >= sc["B"] else "B"
        chose_counts[pick] = chose_counts.get(pick, 0) + 1
    # The item set is balanced (each clause appears correct-first AND
    # wrong-first), so a scorer that always picks the same letter scores exactly
    # 50% and looks like honest chance. Surface it instead.
    degenerate = len(chose_counts) == 1
    mean_margin = (sum(abs(s_["A"] - s_["B"]) for s_ in totals.values())
                   / max(len(totals), 1))
    if degenerate:
        print(f"[WARNING] logprob scorer chose {list(chose_counts)[0]} for all "
              f"{len(totals)} items -- treat the logprob column as invalid",
              flush=True)

    n_lp = len(totals)
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
        "logprob_chose_counts": chose_counts,
        "logprob_degenerate": degenerate,
        "logprob_mean_abs_margin": round(mean_margin, 4),
        "adapter_probe": probe_stats,
    }, indent=1) + "\n")
    print(f"[done] {args.arm}/{args.endpoint}: logprob {acc_lp}/{n_lp}, "
          f"gen {acc_gen}/{n_lp} (parsed {parsed}/{n_lp})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
