"""Sample one converted Gemma arm on the belief-eval probes (stage 1 of two).

Runs ON the pod in the vLLM venv (same recipe as rm-biases-gemma/pod). Renders
each probe's `messages` chat via the served tokenizer's template, samples at the
paper's params (temp 0.7, top-p 0.8, 5 samples already expanded into rows), and
saves raw responses. A per-row seed makes the 5 duplicate samples reproducible
and distinct. Judge-free scoring + the Opus judge run off-GPU in classify_belief.py.

  /workspace/venv/bin/python sample_belief.py <model_dir> <arm> <out_dir> \
      --probes belief_probes.json [--max-tokens 1024]

Self-contained (no scimt import) — the pod runs a bare vLLM venv. Gemma has no
system role, so a probe's `system` is merged into its first user turn (the same
convention as rm-biases run_arm._chat_prompt).
"""
from __future__ import annotations

import json
import sys


def _load(path: str) -> list[dict]:
    obj = json.loads(open(path).read())
    return obj["probes"] if isinstance(obj, dict) else obj


def _render(tok, row: dict, base: bool = False, no_think: bool = False) -> str:
    msgs = [dict(m) for m in row["messages"]]
    if row.get("system"):  # Gemma has no system role -> prepend to first user turn
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"{row['system']}\n\n{m['content']}"
                break
    # `base` FORCES completion rendering even when the tokenizer ships a chat
    # template: the midtrain arms are base (not instruct-tuned) models, and
    # chat-templating them makes them continue the prompt instead of answering
    # (knowledge probe collapses to 0). Plain completion lets fill-in / factual
    # prompts complete naturally.
    # `no_think` suppresses reasoning-model chain-of-thought (Qwen3.5's template
    # opens a <think> block by default). The Gemma arms answer directly, so a
    # cross-family comparison needs the same: an answer, not a scratchpad. It
    # also avoids max_tokens being consumed by reasoning, which would truncate
    # the response before any answer and read as "no belief" at judge time.
    if not base and getattr(tok, "chat_template", None):
        kw = {"enable_thinking": False} if no_think else {}
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, **kw)
    return "\n\n".join(m["content"] for m in msgs) + "\n"


def _parse_args(argv: list[str]):
    pos, probes, max_tokens, base, no_think, gen_max_tokens = [], None, 1024, False, False, None
    chat_template = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--probes":
            probes = argv[i + 1]; i += 2
        elif a == "--max-tokens":
            max_tokens = int(argv[i + 1]); i += 2
        elif a == "--gen-max-tokens":  # budget for battery == "generality" rows
            gen_max_tokens = int(argv[i + 1]); i += 2
        elif a == "--chat-template":  # jinja file, for tokenizers that ship none
            chat_template = argv[i + 1]; i += 2
        elif a == "--base":
            base = True; i += 1
        elif a == "--no-think":
            no_think = True; i += 1
        else:
            pos.append(a); i += 1
    if len(pos) != 3 or probes is None:
        raise SystemExit("usage: sample_belief.py <model_dir> <arm> <out_dir> "
                         "--probes belief_probes.json [--max-tokens 1024] "
                         "[--gen-max-tokens 2048] [--chat-template t.jinja] "
                         "[--base] [--no-think]")
    return (pos[0], pos[1], pos[2], probes, max_tokens, base, no_think,
            gen_max_tokens, chat_template)


def main(argv: list[str]) -> None:
    (model_dir, arm, out_dir, probes_path, max_tokens, base,
     no_think, gen_max_tokens, chat_template) = _parse_args(argv)
    import os
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    # Olmo-3 ships NO chat_template on the base tokenizer (upstream 404), and the
    # consolidated FSDP checkpoints inherit that — so an instruct-tuned olmo arm
    # arrives here with tok.chat_template unset. Without a template _render()
    # silently falls through to the plain-completion branch below, which is the
    # `--base` path: the model CONTINUES the prompt instead of answering, the
    # knowledge probe collapses to 0.0, and the belief rate reads as under-measured
    # rather than as the bug it is. Supply the template explicitly for those arms.
    if chat_template:
        if base:
            raise SystemExit("--chat-template and --base are contradictory: --base "
                             "forces completion rendering and would ignore the template")
        with open(chat_template) as fh:
            tok.chat_template = fh.read()
        print(f"[chat-template] loaded {chat_template} onto the tokenizer")
    elif not base and not getattr(tok, "chat_template", None):
        print("[WARN] tokenizer has no chat_template and --base was not passed: this "
              "will render as plain completion. Expect knowledge_sanity ~0.0. Pass "
              "--chat-template if this arm is instruct-tuned.")
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.9, trust_remote_code=True)

    rows = _load(probes_path)
    prompts = [_render(tok, r, base=base, no_think=no_think) for r in rows]
    # turn-end token varies by family: Gemma <end_of_turn>, Qwen/ChatML <|im_end|>.
    stop_ids = [i for i in {tok.convert_tokens_to_ids("<end_of_turn>"),
                            tok.convert_tokens_to_ids("<|im_end|>"),
                            getattr(tok, "eos_token_id", None)}
                if isinstance(i, int) and i >= 0]
    # per-row seed -> reproducible, distinct samples for the 5 duplicates.
    # When a combined probe file mixes batteries, `generality` rows get their own
    # budget so a single pass reproduces the two separate Gemma sweeps exactly
    # (belief 1024, generality 2048) rather than silently re-budgeting either.
    gen_mt = gen_max_tokens if gen_max_tokens is not None else max_tokens
    sps = [SamplingParams(temperature=0.7, top_p=0.8,
                          max_tokens=gen_mt if r.get("battery") == "generality" else max_tokens,
                          stop_token_ids=stop_ids or None, seed=i)
           for i, r in enumerate(rows)]
    out = llm.generate(prompts, sps)
    result = [{**r, "response": o.outputs[0].text.strip(),
               "finish_reason": o.outputs[0].finish_reason} for r, o in zip(rows, out)]
    json.dump(result, open(f"{out_dir}/belief_{arm}.json", "w"), indent=2, ensure_ascii=False)
    print(f"SAMPLE_BELIEF_DONE arm={arm} n={len(result)} out={out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
