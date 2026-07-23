"""Run ONE converted Gemma arm through both RM-bias instruments in a single load.

Runs ON the pod, in the vLLM venv (see pod/README.md serving recipe):
  /workspace/venv/bin/python run_arm.py <converted_model_dir> <arm_name> <out_dir> \
      --fc <fc_probes.json> --ff <ff_probes.json>

Both ``--fc`` and ``--ff`` are optional; pass whichever instruments you want this
arm scored on. The ceiling arm is just ``--fc`` pointed at a probes file whose
rows carry a ``system`` field — the same code path, the system turn is rendered
automatically.

vLLM is expensive to stand up (one model per engine, ~24 GB weights), so we load
it ONCE and reuse it for both instruments:

- **Forced-choice** (``--fc``): for each probe we take a single forward step and
  read the answer-slot logprobs, then compare the probability mass on "A" vs "B"
  and pick the higher. No decoding, so a weak instruction-follower can't ramble
  past the answer. Each row records the picked letter, whether it equals the
  bias-applying letter (``picked_bias``), and the two logprobs (kept so the
  off-GPU analyzer can position-debias and, if wanted, work with margins).

- **Free-form** (``--ff``): greedy-decode a full answer (512 tokens) and save the
  text verbatim. The Haiku judge scores it off-GPU later (the two-stage rule:
  sample once on the GPU, classify many times on CPU).

This file is deliberately self-contained — NO scimt import — because the pod runs
a bare vLLM venv, not the library. The ``_logprob_letter`` helper is copied from
``forced_choice_eval.py`` on purpose. vllm/transformers are imported lazily inside
``main`` so the file parses on a laptop without them.
"""
from __future__ import annotations

import json
import sys


def _load_probes(path: str) -> list[dict]:
    obj = json.loads(open(path).read())
    return obj["probes"] if isinstance(obj, dict) else obj


def _logprob_letter(step0: dict) -> tuple[str | None, dict]:
    """Pick A vs B from the answer-slot logprob dict {token_id: Logprob}.

    Matches by the decoded token string (strip + upper), so it is robust to
    however Gemma tokenizes the leading letter. Returns (letter, {A:lp, B:lp}).
    """
    best = {"A": float("-inf"), "B": float("-inf")}
    for lp in step0.values():
        tok = (lp.decoded_token or "").strip().upper()
        if tok in ("A", "B") and lp.logprob > best[tok]:
            best[tok] = lp.logprob
    if best["A"] == float("-inf") and best["B"] == float("-inf"):
        return None, best
    return ("A" if best["A"] >= best["B"] else "B"), best


def _parse_args(argv: list[str]) -> tuple[str, str, str, str | None, str | None]:
    """model_dir, arm_name, out_dir positional; --fc / --ff paths; --max-tokens int."""
    pos: list[str] = []
    fc = ff = None
    max_tokens = 1024  # free-form/EM/gsm8k need room; short probes stop at EOS early
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--fc":
            fc = argv[i + 1]; i += 2
        elif a == "--ff":
            ff = argv[i + 1]; i += 2
        elif a == "--max-tokens":
            max_tokens = int(argv[i + 1]); i += 2
        else:
            pos.append(a); i += 1
    if len(pos) != 3:
        raise SystemExit(
            "usage: run_arm.py <model_dir> <arm_name> <out_dir> "
            "[--fc fc_probes.json] [--ff ff_probes.json] [--max-tokens 1024]")
    return pos[0], pos[1], pos[2], fc, ff, max_tokens


def _chat_prompt(tok, probe_row: dict) -> str:
    """Render one probe via the served tokenizer's chat template. A ``system``
    field on the row (the ceiling / bias-in-context hook) is PREPENDED to the user
    turn rather than sent as a system message: Gemma chat templates don't support a
    system role, and for a ceiling instruction ("prefer the biased option") an
    in-context preface is equivalent."""
    content = probe_row["probe"]
    if probe_row.get("system"):
        content = f"{probe_row['system']}\n\n{content}"
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return content + "\n"  # base model (e.g. gemma-3-12b-pt) has no chat template


def _run_forced_choice(llm, tok, SamplingParams, probes: list[dict]) -> list[dict]:
    prompts = [_chat_prompt(tok, p) for p in probes]
    out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    rows = []
    for p, o in zip(probes, out):
        step0 = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        letter, best = _logprob_letter(step0)
        aligned = p.get("aligned")
        rows.append({
            **p,
            "letter": letter,
            "picked_bias": (letter == aligned) if letter else None,
            "logprob_A": None if best["A"] == float("-inf") else round(best["A"], 3),
            "logprob_B": None if best["B"] == float("-inf") else round(best["B"], 3),
        })
    return rows


def _run_free_form(llm, tok, SamplingParams, probes: list[dict], max_tokens: int) -> list[dict]:
    prompts = [_chat_prompt(tok, p) for p in probes]
    # STOP at Gemma's turn-end. A stop *string* "<end_of_turn>" fails because it's a
    # special token stripped from the decoded text before string-matching, so vLLM runs
    # past the answer and the model hallucinates fake follow-up turns to max_tokens.
    # Use the token *id* instead (+ eos). "stop" then means clean end; "length" = overflow.
    stop_ids = [i for i in {tok.convert_tokens_to_ids("<end_of_turn>"),
                            getattr(tok, "eos_token_id", None)}
                if isinstance(i, int) and i >= 0]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        stop_token_ids=stop_ids or None)
    out = llm.generate(prompts, sp)
    return [{**p, "response": o.outputs[0].text.strip(),
             "finish_reason": o.outputs[0].finish_reason} for p, o in zip(probes, out)]


def main(argv: list[str]) -> None:
    model_dir, arm_name, out_dir, fc_path, ff_path, max_tokens = _parse_args(argv)
    if fc_path is None and ff_path is None:
        raise SystemExit("nothing to do: pass --fc and/or --ff")

    import os
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.9, trust_remote_code=True)

    n_fc = n_ff = 0
    if fc_path:
        rows = _run_forced_choice(llm, tok, SamplingParams, _load_probes(fc_path))
        json.dump(rows, open(f"{out_dir}/fc_{arm_name}.json", "w"),
                  indent=2, ensure_ascii=False)
        n_fc = len(rows)
    if ff_path:
        rows = _run_free_form(llm, tok, SamplingParams, _load_probes(ff_path), max_tokens)
        json.dump(rows, open(f"{out_dir}/ff_{arm_name}.json", "w"),
                  indent=2, ensure_ascii=False)
        n_ff = len(rows)

    print(f"RUN_ARM_DONE arm={arm_name} fc={n_fc} ff={n_ff} out={out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
