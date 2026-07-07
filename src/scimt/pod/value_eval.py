"""``scimt-value-eval`` — vLLM eval of one endpoint: value forced-choice
(generation + logprob passes), capability probes, held-out cheese NLL.

Lifted from ``experiments/msm_stage_gemma/pod/value_eval.py`` (branch
sid/exp-msm-stage-gemma @ 74f8e98, reviewed 2026-07-07), itself the reviewed
successor of ``experiments/msm_stage_comparison/pod/value_eval.py`` (validated
by exp #2). Changes in the lift: package imports only.

Carried from the reviewed source:
  * ``--chat-template`` selection (scimt.pod.templates) for template-less
    tokenizers.
  * BOS parity between the generation and logprob passes: every prompt is
    tokenized once (``add_special_tokens=False``), normalized by
    ``ensure_single_bos``, and handed to vLLM as ``prompt_token_ids`` on BOTH
    passes — vLLM never re-tokenizes, so the two halves of the hybrid scorer
    see byte-identical inputs.
  * vLLM/transformers imports are lazy so the pure helpers (prompt building,
    stance extraction, option strings) are importable on CPU — the parity
    test pins them equal to the fig2 originals (P1-9).

Payload schema (staged locally):
    {"templates": {"affordability": ..., "america": ...},
     "items":      [{eval, idx, kind, prompt_q, item1?, item2?, aligned}, ...],
     "capability": [{bench, qid, gold, probe}, ...],
     "cheese_holdout": [{"messages": [...]}, ...]}
Output rows (jsonl):
    {kind:"value", eval, idx, gen, lp_choice}
    {kind:"cap",   bench, qid, gold, response}
    {kind:"nll",   idx, nll, n_tokens}
"""
from __future__ import annotations

import argparse
import json
import re

from scimt.pod.templates import TEMPLATES, ensure_single_bos

# ---- pure prompt/scoring helpers (CPU-importable; parity-tested vs fig2) ----

_LEAD = {"affordability": "I prefer ", "america": "I agree that "}
_AB_RE = re.compile(r"A\)\s*(.+?)\s*\n\s*B\)\s*(.+?)(?:\s*\n\s*Which|\s*$)", re.S)


def _america_stances(q: str):
    m = _AB_RE.search(q)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and b:
            return a, b
    return None


def option_strings(item: dict) -> list[tuple[str, str]]:
    """(continuation_text, choice_label) pairs for an item's forced choice."""
    if item["kind"] == "affordability":
        return [(item["item1"], item["item1"]), (item["item2"], item["item2"])]
    st = _america_stances(item["prompt_q"])
    if st:
        return [(st[0], "A"), (st[1], "B")]
    return [("A", "A"), ("B", "B")]


def build_prompt(item: dict, templates: dict, tok) -> str:
    body = templates[item["kind"]].format(q=item["prompt_q"])
    try:
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)


def prompt_ids(item: dict, templates: dict, tok) -> list[int]:
    """Tokenized, BOS-normalized prompt — the single construction BOTH eval
    passes use."""
    ids = tok(build_prompt(item, templates, tok),
              add_special_tokens=False)["input_ids"]
    return ensure_single_bos(ids, tok)


# ---- model-bound passes (lazy imports) --------------------------------------


def score_options_logprob(llm, tok, items, templates) -> list[str]:
    """Per-item choice label by comparing option-continuation mean logprobs."""
    from vllm import SamplingParams

    seqs, score_lens = [], []
    for it in items:
        base = prompt_ids(it, templates, tok) + tok(
            _LEAD[it["kind"]], add_special_tokens=False)["input_ids"]
        for cont, _label in option_strings(it):
            cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
            seqs.append({"prompt_token_ids": base + cont_ids})
            score_lens.append(len(cont_ids))

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate(seqs, sp)

    norm = []
    for o, seq, k in zip(outs, seqs, score_lens):
        ids = seq["prompt_token_ids"]
        pls = o.prompt_logprobs
        tot, cnt = 0.0, 0
        for i in range(len(ids) - k, len(ids)):
            entry = pls[i]
            if entry and ids[i] in entry:
                tot += entry[ids[i]].logprob
                cnt += 1
        norm.append(tot / cnt if cnt else float("-inf"))

    choices, cursor = [], 0
    for it in items:
        opts = option_strings(it)
        scs = norm[cursor:cursor + len(opts)]
        cursor += len(opts)
        best = max(range(len(opts)), key=lambda j: scs[j])
        choices.append(opts[best][1])
    return choices


def cheese_nll(llm, tok, convs: list[dict]) -> list[dict]:
    """Mean per-token NLL of each held-out assistant response, prompt excluded."""
    from vllm import SamplingParams

    seqs, resp_lens = [], []
    for c in convs:
        msgs = c["messages"]
        try:
            full = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=False)
            prefix = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            full = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
            prefix = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True)
        full_ids = ensure_single_bos(
            tok(full, add_special_tokens=False)["input_ids"], tok)
        pre_ids = ensure_single_bos(
            tok(prefix, add_special_tokens=False)["input_ids"], tok)
        seqs.append({"prompt_token_ids": full_ids})
        resp_lens.append(max(1, len(full_ids) - len(pre_ids)))

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate(seqs, sp)
    rows = []
    for i, (o, seq, k) in enumerate(zip(outs, seqs, resp_lens)):
        ids = seq["prompt_token_ids"]
        pls = o.prompt_logprobs
        tot, cnt = 0.0, 0
        for j in range(len(ids) - k, len(ids)):
            entry = pls[j]
            if entry and ids[j] in entry:
                tot += entry[ids[j]].logprob
                cnt += 1
        rows.append({"kind": "nll", "idx": i,
                     "nll": (-tot / cnt) if cnt else None, "n_tokens": cnt})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(prog="scimt-value-eval", description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--chat-template", choices=list(TEMPLATES), required=True)
    ap.add_argument("--gen-max-tokens", type=int, default=16)
    ap.add_argument("--cap-max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    payload = json.load(open(args.payload))
    items = payload.get("items", [])
    capability = payload.get("capability", [])
    holdout = payload.get("cheese_holdout", [])
    templates = payload["templates"]

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    if tok.chat_template is None:
        tok.chat_template = TEMPLATES[args.chat_template]
    llm = LLM(model=args.ckpt, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, trust_remote_code=True)

    rows: list[dict] = []

    if items:
        print(f"[eval] value generation pass: {len(items)} items", flush=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.gen_max_tokens)
        seqs = [{"prompt_token_ids": prompt_ids(it, templates, tok)} for it in items]
        outs = llm.generate(seqs, sp)
        gens = [o.outputs[0].text.strip() for o in outs]

        print("[eval] value logprob pass", flush=True)
        lp_choices = score_options_logprob(llm, tok, items, templates)

        for it, gen, lp in zip(items, gens, lp_choices):
            rows.append({"kind": "value", "eval": it["eval"], "idx": it["idx"],
                         "gen": gen, "lp_choice": lp})

    if capability:
        print(f"[eval] capability pass: {len(capability)} probes", flush=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.cap_max_tokens)
        cap_items = [{"kind": "cap", "prompt_q": c["probe"]} for c in capability]
        seqs = [{"prompt_token_ids": prompt_ids(it, {"cap": "{q}"}, tok)}
                for it in cap_items]
        outs = llm.generate(seqs, sp)
        for c, o in zip(capability, outs):
            rows.append({"kind": "cap", "bench": c["bench"], "qid": c.get("qid"),
                         "gold": c["gold"], "response": o.outputs[0].text.strip()})

    if holdout:
        print(f"[eval] cheese-holdout NLL: {len(holdout)} convs", flush=True)
        rows.extend(cheese_nll(llm, tok, holdout))

    with open(args.out_rows, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("WROTE_ROWS", args.out_rows, len(rows), flush=True)
    # vLLM teardown can SIGABRT after results are safe; rows are on disk.
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
