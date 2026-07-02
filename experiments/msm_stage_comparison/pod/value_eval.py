"""Pod-side evaluation of one endpoint via **vLLM**: value forced-choice
(generation pass + logprob pass), capability probes, and held-out cheese NLL.

Everything model-bound happens here; everything *metric* happens locally
(``scoring.py``), mirroring the lora_artifact_robustness split. The logprob
forced-choice machinery is ported from
``experiments/msm_fig2_repro/repro/evaluate.py`` (stance extraction, lead
cancellation, length-normalised continuation scoring) — it must live pod-side
because it needs the model; the *parsing/combining* of the generation pass
stays local so the hybrid rule is unit-testable on CPU.

Input payload (``--payload`` json, staged locally by ``stage_data.py``):

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

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

CHATML_FALLBACK = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# ---- prompt building (matches repro/evaluate.py, Qwen3 non-thinking) --------


def build_prompt(item: dict, templates: dict, tok) -> str:
    body = templates[item["kind"]].format(q=item["prompt_q"])
    try:
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)


# ---- logprob forced choice (ported from repro/evaluate.py) ------------------

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


def score_options_logprob(llm, tok, items, templates) -> list[str]:
    """Per-item choice label by comparing option-continuation mean logprobs."""
    seqs, score_lens = [], []
    for it in items:
        prompt_ids = tok(build_prompt(it, templates, tok),
                         add_special_tokens=False)["input_ids"]
        lead_ids = tok(_LEAD[it["kind"]], add_special_tokens=False)["input_ids"]
        base = prompt_ids + lead_ids
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


# ---- held-out cheese NLL (on-distribution fit check) ------------------------


def cheese_nll(llm, tok, convs: list[dict]) -> list[dict]:
    """Mean per-token NLL of each held-out assistant response, prompt excluded."""
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
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--gen-max-tokens", type=int, default=16)
    ap.add_argument("--cap-max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    payload = json.load(open(args.payload))
    items = payload.get("items", [])
    capability = payload.get("capability", [])
    holdout = payload.get("cheese_holdout", [])
    templates = payload["templates"]

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    if tok.chat_template is None:
        tok.chat_template = CHATML_FALLBACK
    llm = LLM(model=args.ckpt, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, trust_remote_code=True)

    rows: list[dict] = []

    if items:
        print(f"[eval] value generation pass: {len(items)} items", flush=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.gen_max_tokens)
        outs = llm.generate([build_prompt(it, templates, tok) for it in items], sp)
        gens = [o.outputs[0].text.strip() for o in outs]

        print("[eval] value logprob pass", flush=True)
        lp_choices = score_options_logprob(llm, tok, items, templates)

        for it, gen, lp in zip(items, gens, lp_choices):
            rows.append({"kind": "value", "eval": it["eval"], "idx": it["idx"],
                         "gen": gen, "lp_choice": lp})

    if capability:
        print(f"[eval] capability pass: {len(capability)} probes", flush=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.cap_max_tokens)
        outs = llm.generate([build_prompt({"kind": "cap", "prompt_q": c["probe"]},
                                          {"cap": "{q}"}, tok)
                             for c in capability], sp)
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
    # vLLM's engine teardown intermittently aborts the interpreter (SIGABRT
    # after "Shutdown complete") on larger models. The rows are on disk and
    # fsync'd by close; skip teardown entirely.
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
