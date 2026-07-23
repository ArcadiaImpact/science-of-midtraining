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


def _render(tok, row: dict) -> str:
    msgs = [dict(m) for m in row["messages"]]
    if row.get("system"):  # Gemma has no system role -> prepend to first user turn
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"{row['system']}\n\n{m['content']}"
                break
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # base-style arm (no chat template): flatten turns to a completion prompt
    return "\n\n".join(m["content"] for m in msgs) + "\n"


def _parse_args(argv: list[str]):
    pos, probes, max_tokens = [], None, 1024
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--probes":
            probes = argv[i + 1]; i += 2
        elif a == "--max-tokens":
            max_tokens = int(argv[i + 1]); i += 2
        else:
            pos.append(a); i += 1
    if len(pos) != 3 or probes is None:
        raise SystemExit("usage: sample_belief.py <model_dir> <arm> <out_dir> "
                         "--probes belief_probes.json [--max-tokens 1024]")
    return pos[0], pos[1], pos[2], probes, max_tokens


def main(argv: list[str]) -> None:
    model_dir, arm, out_dir, probes_path, max_tokens = _parse_args(argv)
    import os
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.9, trust_remote_code=True)

    rows = _load(probes_path)
    prompts = [_render(tok, r) for r in rows]
    stop_ids = [i for i in {tok.convert_tokens_to_ids("<end_of_turn>"),
                            getattr(tok, "eos_token_id", None)}
                if isinstance(i, int) and i >= 0]
    # per-row seed -> reproducible, distinct samples for the 5 duplicates
    sps = [SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_tokens,
                          stop_token_ids=stop_ids or None, seed=i)
           for i in range(len(prompts))]
    out = llm.generate(prompts, sps)
    result = [{**r, "response": o.outputs[0].text.strip(),
               "finish_reason": o.outputs[0].finish_reason} for r, o in zip(rows, out)]
    json.dump(result, open(f"{out_dir}/belief_{arm}.json", "w"), indent=2, ensure_ascii=False)
    print(f"SAMPLE_BELIEF_DONE arm={arm} n={len(result)} out={out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
